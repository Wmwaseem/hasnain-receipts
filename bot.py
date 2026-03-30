import os, json, time, threading, logging
from datetime import datetime
from collections import deque
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot_state = {
    "running": False, "paper_mode": True, "testnet": True,
    "api_key": "", "api_secret": "",
    "balance": 1000.0, "initial_balance": 1000.0,
    "open_trades": {}, "closed_trades": [], "signals": [],
    "logs": deque(maxlen=300),
    "stats": {
        "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0, "total_pnl": 0, "drawdown": 0,
        "peak_balance": 1000.0, "scans": 0, "signals_found": 0
    },
    "market_regime": "UNKNOWN", "btc_trend": "NEUTRAL",
    "config": {
        "max_drawdown": 5.0, "risk_per_trade": 1.5,
        "max_trades": 5, "min_rr": 2.0,
        "max_hold_seconds": 300, "scan_interval": 30,
        "min_volume_usdt": 3.0, "min_confluence_score": 62,
        "atr_sl_multiplier": 1.5
    }
}

WATCHLIST = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT",
    "MATICUSDT","LTCUSDT","UNIUSDT","ATOMUSDT","NEARUSDT",
    "FTMUSDT","SANDUSDT","AXSUSDT","APEUSDT","INJUSDT"
]

client = None

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] [{level}] {msg}"
    bot_state["logs"].appendleft(entry)
    logger.info(msg)

def init_client():
    global client
    if not BINANCE_AVAILABLE:
        log("python-binance not installed! Run: pip install python-binance", "ERROR")
        return False
    try:
        ak = bot_state["api_key"]; sk = bot_state["api_secret"]
        client = Client(ak, sk, testnet=bot_state["testnet"]) if ak and sk else Client("", "", testnet=bot_state["testnet"])
        client.ping()
        log(f"Connected to Binance {'Testnet' if bot_state['testnet'] else 'Live'}")
        if ak and sk and not bot_state["paper_mode"]:
            acc = client.get_account()
            usdt = next((float(a['free']) for a in acc['balances'] if a['asset']=='USDT'), 0)
            bot_state["balance"] = usdt
            log(f"Real balance: ${usdt:.2f} USDT")
        return True
    except Exception as e:
        log(f"Connection failed: {e}", "ERROR")
        return False

def get_klines(symbol, interval, limit=100):
    try:
        raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume','ct','qv','trades','tbb','tbq','ignore'])
        return df.astype({'open':float,'high':float,'low':float,'close':float,'volume':float,'qv':float})
    except: return None

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d=s.diff(); g=d.where(d>0,0); l=-d.where(d<0,0)
    ag=g.ewm(com=n-1,adjust=False).mean(); al=l.ewm(com=n-1,adjust=False).mean()
    return 100-(100/(1+(ag/al)))
def macd(s,f=12,sl=26,sig=9):
    ml=ema(s,f)-ema(s,sl); sl_=ema(ml,sig); return ml,sl_,ml-sl_
def bollinger(s,n=20,k=2):
    m=s.rolling(n).mean(); sd=s.rolling(n).std()
    up=m+k*sd; lo=m-k*sd
    return up,m,lo,(s-lo)/(up-lo)
def atr(df,n=14):
    h,l,c=df['high'],df['low'],df['close']
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
    return tr.ewm(com=n-1,adjust=False).mean()
def adx(df,n=14):
    h,l,c=df['high'],df['low'],df['close']
    pdm=h.diff(); ndm=l.diff().abs()
    pdm=pdm.where((pdm>ndm)&(pdm>0),0)
    ndm=ndm.where((ndm>pdm.abs())&(ndm>0),0)
    at=atr(df,n)
    pdi=100*(pdm.ewm(com=n-1,adjust=False).mean()/at)
    ndi=100*(ndm.ewm(com=n-1,adjust=False).mean()/at)
    dx=100*((pdi-ndi).abs()/(pdi+ndi))
    return dx.ewm(com=n-1,adjust=False).mean(),pdi,ndi
def vwap(df):
    tp=(df['high']+df['low']+df['close'])/3
    return (tp*df['volume']).cumsum()/df['volume'].cumsum()
def obv(df):
    return (np.sign(df['close'].diff())*df['volume']).fillna(0).cumsum()
def rsi_divergence(df, rsi_s, lb=10):
    if len(df)<lb: return "none"
    p=df['close'].values[-lb:]; r=rsi_s.values[-lb:]
    pt=p[-1]-p[0]; rt=r[-1]-r[0]
    if pt<0 and rt>2: return "bullish"
    if pt>0 and rt<-2: return "bearish"
    return "none"

def market_regime(df):
    try:
        c=df['close']
        e20=ema(c,20).iloc[-1]; e50=ema(c,50).iloc[-1]
        e200=ema(c,min(200,len(c)//2)).iloc[-1]
        adx_,pdi,ndi=adx(df); av=adx_.iloc[-1]
        at=atr(df); ap=(at.iloc[-1]/c.iloc[-1])*100
        cur=c.iloc[-1]
        if av>28:
            if e20>e50>e200 and cur>e20: return "TRENDING_UP",av
            if e20<e50<e200 and cur<e20: return "TRENDING_DOWN",av
        if ap>2.5: return "VOLATILE",av
        return "RANGING",av
    except: return "UNKNOWN",0

def btc_trend():
    try:
        df=get_klines("BTCUSDT","15m",50)
        if df is None: return "NEUTRAL"
        c=df['close']
        e20=ema(c,20).iloc[-1]; e50=ema(c,50).iloc[-1]; cur=c.iloc[-1]
        r=rsi(c).iloc[-1]
        if cur>e20>e50 and r>50: return "BULLISH"
        if cur<e20<e50 and r<50: return "BEARISH"
        return "NEUTRAL"
    except: return "NEUTRAL"

def analyze_tf(symbol, interval):
    df=get_klines(symbol,interval,100)
    if df is None or len(df)<50: return None
    c=df['close']; res={}
    e20=ema(c,20); e50=ema(c,50)
    res['price']=c.iloc[-1]; res['ema20']=e20.iloc[-1]; res['ema50']=e50.iloc[-1]
    res['ema_bull']=e20.iloc[-1]>e50.iloc[-1]
    res['above_ema20']=c.iloc[-1]>e20.iloc[-1]
    r=rsi(c); rv=r.iloc[-1]
    res['rsi']=rv; res['rsi_bull']=40<rv<70
    res['rsi_os']=rv<35; res['rsi_ob']=rv>70
    res['rsi_div']=rsi_divergence(df,r)
    ml,sl,hist=macd(c)
    res['macd']=ml.iloc[-1]; res['macd_sig']=sl.iloc[-1]; res['macd_hist']=hist.iloc[-1]
    res['macd_bull']=hist.iloc[-1]>0 and hist.iloc[-1]>hist.iloc[-2]
    _,_,_,bbp=bollinger(c)
    res['bb_pct']=bbp.iloc[-1]
    adx_,pdi,ndi=adx(df)
    res['adx']=adx_.iloc[-1]; res['pdi']=pdi.iloc[-1]; res['ndi']=ndi.iloc[-1]
    res['trend_strong']=adx_.iloc[-1]>25; res['di_bull']=pdi.iloc[-1]>ndi.iloc[-1]
    vw=vwap(df); res['vwap']=vw.iloc[-1]; res['above_vwap']=c.iloc[-1]>vw.iloc[-1]
    at=atr(df); res['atr']=at.iloc[-1]; res['atr_pct']=(at.iloc[-1]/c.iloc[-1])*100
    vol_avg=df['volume'].rolling(20).mean().iloc[-1]
    res['vol_ratio']=df['volume'].iloc[-1]/vol_avg if vol_avg>0 else 1
    res['vol_surge']=res['vol_ratio']>1.5
    ob=obv(df); ob_ema=ema(ob,10)
    res['obv_up']=ob.iloc[-1]>ob_ema.iloc[-1]
    return res

def confluence_score(tf1,tf5,tf15,btc):
    bs=0; ss=0; br=[]; sr=[]
    # BTC Master Filter (15pts)
    if btc=="BULLISH": bs+=15; br.append("BTC trend bullish ✅")
    elif btc=="BEARISH": ss+=15; sr.append("BTC trend bearish ✅")
    # 15m bias (30pts)
    if tf15:
        if tf15['ema_bull']: bs+=8; br.append("15m EMA stack bullish")
        else: ss+=8; sr.append("15m EMA stack bearish")
        if tf15['above_vwap']: bs+=7; br.append("15m price above VWAP")
        else: ss+=7; sr.append("15m price below VWAP")
        if tf15['macd_bull']: bs+=8; br.append("15m MACD momentum ↑")
        else: ss+=8; sr.append("15m MACD momentum ↓")
        if tf15['di_bull']: bs+=7; br.append("15m +DI dominant")
        else: ss+=7; sr.append("15m -DI dominant")
    # 5m entry timing (30pts)
    if tf5:
        if tf5['ema_bull'] and tf5['above_ema20']: bs+=8; br.append("5m price above EMA20")
        else: ss+=8; sr.append("5m price below EMA20")
        if tf5['rsi_bull']: bs+=5; br.append("5m RSI healthy zone")
        if tf5['rsi_div']=='bullish': bs+=8; br.append("5m Bullish RSI divergence 🔥")
        elif tf5['rsi_div']=='bearish': ss+=8; sr.append("5m Bearish RSI divergence 🔥")
        if tf5['vol_surge']: bs+=3; ss+=3; br.append("Volume surge 📈"); sr.append("Volume surge 📈")
        if tf5['obv_up']: bs+=4; br.append("OBV rising (smart money)")
        else: ss+=4; sr.append("OBV falling (smart money)")
    # 1m trigger (20pts)
    if tf1:
        if tf1['macd_bull']: bs+=10; br.append("1m MACD crossover up ⚡")
        else: ss+=10; sr.append("1m MACD crossover down ⚡")
        if tf1['rsi_os']: bs+=7; br.append("1m RSI oversold bounce")
        elif tf1['rsi_ob']: ss+=7; sr.append("1m RSI overbought rejection")
    if bs>ss: return bs,"BUY",br
    return ss,"SELL",sr

def position_size(balance, entry, at, direction):
    cfg=bot_state["config"]
    risk_amt=balance*(cfg["risk_per_trade"]/100)
    sl_dist=at*cfg["atr_sl_multiplier"]
    sl_pct=(sl_dist/entry)*100
    pos=risk_amt/(sl_pct/100); pos=min(pos,balance*0.25)
    if direction=="BUY":
        sl=entry-sl_dist; tp=entry+(sl_dist*cfg["min_rr"])
    else:
        sl=entry+sl_dist; tp=entry-(sl_dist*cfg["min_rr"])
    return round(pos,2),round(sl,8),round(tp,8)

def scan_symbol(symbol):
    try:
        tf15=analyze_tf(symbol,"15m")
        tf5=analyze_tf(symbol,"5m")
        tf1=analyze_tf(symbol,"1m")
        if not tf5 or not tf1: return None
        df5=get_klines(symbol,"5m",5)
        if df5 is None: return None
        vol_m=df5['qv'].mean()/1e6
        if vol_m<bot_state["config"]["min_volume_usdt"]: return None
        sc,action,reasons=confluence_score(tf1,tf5,tf15,bot_state["btc_trend"])
        if sc<bot_state["config"]["min_confluence_score"]: return None
        entry=tf1['price']; at=tf1['atr']
        pos,sl,tp=position_size(bot_state["balance"],entry,at,action)
        if action=="BUY": rr=(tp-entry)/(entry-sl) if (entry-sl)>0 else 0
        else: rr=(entry-tp)/(sl-entry) if (sl-entry)>0 else 0
        if rr<bot_state["config"]["min_rr"]: return None
        df15=get_klines(symbol,"15m",100)
        regime,adx_v=market_regime(df15) if df15 is not None else ("UNKNOWN",0)
        return {
            "symbol":symbol,"action":action,"score":round(sc),
            "accuracy":min(round(sc*0.93),94),
            "entry":entry,"stop_loss":sl,"take_profit":tp,
            "position_size":pos,"rsi":round(tf5['rsi'],1),
            "macd":round(tf5['macd'],8),"adx":round(adx_v,1),
            "bb_pct":round(tf5['bb_pct']*100,1),
            "vol_ratio":round(tf5['vol_ratio'],2),
            "regime":regime,"rr":round(rr,2),
            "reasons":reasons[:5],
            "timestamp":datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        log(f"Scan error {symbol}: {e}","ERROR"); return None

def open_trade(signal):
    if signal['symbol'] in bot_state["open_trades"]: return
    if len(bot_state["open_trades"])>=bot_state["config"]["max_trades"]: return
    pk=bot_state["stats"]["peak_balance"]; bal=bot_state["balance"]
    dd=((pk-bal)/pk)*100 if pk>0 else 0
    if dd>=bot_state["config"]["max_drawdown"]:
        log(f"Max drawdown {dd:.1f}% — no new trades","WARN"); return
    trade={
        "symbol":signal['symbol'],"action":signal['action'],
        "entry":signal['entry'],"current":signal['entry'],
        "stop_loss":signal['stop_loss'],"take_profit":signal['take_profit'],
        "trailing_sl":signal['stop_loss'],"position_size":signal['position_size'],
        "score":signal['score'],"reasons":signal['reasons'],
        "pnl":0,"pnl_pct":0,
        "open_time":time.time(),"open_time_str":datetime.now().strftime("%H:%M:%S"),
        "highest":signal['entry'],"lowest":signal['entry']
    }
    bot_state["open_trades"][signal['symbol']]=trade
    bot_state["stats"]["total_trades"]+=1
    log(f"OPEN {signal['action']} {signal['symbol']} @ {signal['entry']} | Score:{signal['score']} | RR:{signal['rr']}")

def update_trades():
    to_close=[]
    for sym,trade in bot_state["open_trades"].items():
        try:
            tk=client.get_symbol_ticker(symbol=sym)
            cp=float(tk['price']); trade['current']=cp
            if trade['action']=="BUY":
                pnl_pct=((cp-trade['entry'])/trade['entry'])*100
                if cp>trade['highest']:
                    trade['highest']=cp
                    pd_=(cp-trade['entry'])*0.5
                    nts=trade['entry']+pd_
                    if nts>trade['trailing_sl']: trade['trailing_sl']=nts
            else:
                pnl_pct=((trade['entry']-cp)/trade['entry'])*100
                if cp<trade['lowest']:
                    trade['lowest']=cp
                    pd_=(trade['entry']-cp)*0.5
                    nts=trade['entry']-pd_
                    if nts<trade['trailing_sl']: trade['trailing_sl']=nts
            trade['pnl']=round((pnl_pct/100)*trade['position_size'],6)
            trade['pnl_pct']=round(pnl_pct,4)
            reason=None
            if trade['action']=="BUY":
                if cp>=trade['take_profit']: reason="TP Hit 🎯"
                elif cp<=trade['trailing_sl']: reason="Trailing SL 🔒" if trade['trailing_sl']>trade['stop_loss'] else "SL Hit"
            else:
                if cp<=trade['take_profit']: reason="TP Hit 🎯"
                elif cp>=trade['trailing_sl']: reason="Trailing SL 🔒" if trade['trailing_sl']<trade['stop_loss'] else "SL Hit"
            if time.time()-trade['open_time']>bot_state["config"]["max_hold_seconds"]: reason="Time Limit ⏱"
            if reason: to_close.append((sym,reason))
        except Exception as e: log(f"Update error {sym}: {e}","ERROR")
    for sym,reason in to_close: close_trade(sym,reason)

def close_trade(symbol, reason):
    if symbol not in bot_state["open_trades"]: return
    trade=bot_state["open_trades"].pop(symbol)
    pnl=trade['pnl']
    bot_state["balance"]+=pnl; bot_state["stats"]["total_pnl"]+=pnl
    if bot_state["balance"]>bot_state["stats"]["peak_balance"]: bot_state["stats"]["peak_balance"]=bot_state["balance"]
    dd=((bot_state["stats"]["peak_balance"]-bot_state["balance"])/bot_state["stats"]["peak_balance"])*100
    bot_state["stats"]["drawdown"]=max(0,round(dd,2))
    if pnl>0: bot_state["stats"]["wins"]+=1
    else: bot_state["stats"]["losses"]+=1
    t=bot_state["stats"]["wins"]+bot_state["stats"]["losses"]
    bot_state["stats"]["win_rate"]=round((bot_state["stats"]["wins"]/t)*100,1) if t>0 else 0
    closed={**trade,"close_price":trade['current'],"close_reason":reason,
            "close_time":datetime.now().strftime("%H:%M:%S"),
            "final_pnl":round(pnl,6),"final_pnl_pct":round(trade['pnl_pct'],2)}
    bot_state["closed_trades"].insert(0,closed)
    if len(bot_state["closed_trades"])>100: bot_state["closed_trades"]=bot_state["closed_trades"][:100]
    e="✅" if pnl>0 else "❌"
    log(f"{e} CLOSE {trade['action']} {symbol} | {reason} | P&L: ${pnl:+.6f} ({trade['pnl_pct']:+.2f}%)")

def bot_loop():
    log("🚀 CryptoSense ELITE Bot v2.0 started!")
    log(f"Mode: {'📄 Paper' if bot_state['paper_mode'] else '💸 Live'} | Risk/trade: {bot_state['config']['risk_per_trade']}%")
    si=0
    while bot_state["running"]:
        try:
            if si%5==0:
                old=bot_state["btc_trend"]
                bot_state["btc_trend"]=btc_trend()
                if old!=bot_state["btc_trend"]: log(f"BTC trend: {old} → {bot_state['btc_trend']}")
            if bot_state["open_trades"]: update_trades()
            bot_state["stats"]["scans"]+=1
            log(f"🔍 Scan #{bot_state['stats']['scans']} — scanning {len(WATCHLIST)} coins...")
            signals=[]
            for sym in WATCHLIST:
                if not bot_state["running"]: break
                sig=scan_symbol(sym)
                if sig:
                    signals.append(sig); bot_state["stats"]["signals_found"]+=1
                    log(f"⚡ {sig['action']} {sym} | Score:{sig['score']} | {sig['reasons'][0] if sig['reasons'] else ''}")
                    if sig['score']>=bot_state["config"]["min_confluence_score"]: open_trade(sig)
            signals.sort(key=lambda x:x['score'],reverse=True)
            bot_state["signals"]=signals[:20]
            log(f"✅ Done. {len(signals)} signals | {len(bot_state['open_trades'])} open trades | Balance: ${bot_state['balance']:.2f}")
            si+=1
            for _ in range(bot_state["config"]["scan_interval"]):
                if not bot_state["running"]: break
                time.sleep(1)
                if bot_state["open_trades"]: update_trades()
        except Exception as e:
            log(f"Loop error: {e}","ERROR"); time.sleep(5)
    log("🛑 Bot stopped.")

@app.route('/')
def index(): return send_from_directory('.','index.html')

@app.route('/api/status')
def get_status():
    roi=((bot_state["balance"]-bot_state["initial_balance"])/bot_state["initial_balance"])*100
    return jsonify({
        "running":bot_state["running"],"paper_mode":bot_state["paper_mode"],
        "testnet":bot_state["testnet"],"balance":round(bot_state["balance"],2),
        "initial_balance":bot_state["initial_balance"],"roi":round(roi,2),
        "open_trades":list(bot_state["open_trades"].values()),
        "closed_trades":bot_state["closed_trades"][:25],
        "signals":bot_state["signals"][:12],"stats":bot_state["stats"],
        "btc_trend":bot_state["btc_trend"],"logs":list(bot_state["logs"])[:60],
        "config":bot_state["config"]
    })

@app.route('/api/start',methods=['POST'])
def start_bot():
    if bot_state["running"]: return jsonify({"success":False,"message":"Already running"})
    if not init_client(): return jsonify({"success":False,"message":"Binance connection failed"})
    bot_state["running"]=True
    threading.Thread(target=bot_loop,daemon=True).start()
    return jsonify({"success":True,"message":"Bot started!"})

@app.route('/api/stop',methods=['POST'])
def stop_bot():
    bot_state["running"]=False
    return jsonify({"success":True,"message":"Bot stopped"})

@app.route('/api/reset',methods=['POST'])
def reset_bot():
    bot_state["running"]=False; bot_state["balance"]=1000.0
    bot_state["initial_balance"]=1000.0; bot_state["open_trades"]={}
    bot_state["closed_trades"]=[]; bot_state["signals"]=[]
    bot_state["logs"]=deque(maxlen=300)
    bot_state["stats"]={"total_trades":0,"wins":0,"losses":0,"win_rate":0,
                        "total_pnl":0,"drawdown":0,"peak_balance":1000.0,"scans":0,"signals_found":0}
    bot_state["btc_trend"]="NEUTRAL"; log("🔄 Bot reset")
    return jsonify({"success":True})

@app.route('/api/config',methods=['POST'])
def save_config():
    d=request.json
    if 'config' in d:
        for k,v in d['config'].items():
            if k in bot_state["config"]: bot_state["config"][k]=v
    for key in ['api_key','api_secret','paper_mode','testnet']:
        if key in d: bot_state[key]=d[key]
    log("⚙️ Config saved"); return jsonify({"success":True})

@app.route('/api/close_trade',methods=['POST'])
def manual_close():
    sym=request.json.get('symbol')
    if sym: close_trade(sym,"Manual Close 👤"); return jsonify({"success":True})
    return jsonify({"success":False})

if __name__=='__main__':
    log("CryptoSense ELITE v2.0 — Dashboard: http://0.0.0.0:5000")
    app.run(host='0.0.0.0',port=5000,debug=False)
