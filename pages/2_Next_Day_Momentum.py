import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, io, math, time

st.set_page_config(page_title="Next-Day Momentum", page_icon="🚀", layout="wide")
st.title("🚀 Next-Day Momentum Scanner")
st.caption("Ranks NSE stocks using end-of-day momentum, trend, volume and volatility signals. It does not guarantee tomorrow's return.")

@st.cache_data(ttl=86400)
def universe():
    try:
        r=requests.get("https://archives.nseindia.com/content/indices/ind_nifty500list.csv",timeout=15,headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status(); x=pd.read_csv(io.BytesIO(r.content))
        return x[["Symbol","Company Name","Industry"]].drop_duplicates("Symbol")
    except Exception:
        return pd.DataFrame({"Symbol":"RELIANCE TCS HDFCBANK ICICIBANK INFY BHARTIARTL ITC LT SBIN AXISBANK KOTAKBANK TITAN BAJFINANCE SUNPHARMA MARUTI M&M NTPC POWERGRID TATAMOTORS TATASTEEL HCLTECH WIPRO ADANIENT CIPLA DRREDDY EICHERMOT JSWSTEEL ONGC TECHM ULTRACEMCO".split()})

def rsi(c,p=14):
    d=c.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/p,adjust=False).mean(); ad=dn.ewm(alpha=1/p,adjust=False).mean()
    return 100-(100/(1+au/ad.replace(0,np.nan)))

def adx(h,l,c,p=14):
    up=h.diff(); dn=-l.diff()
    plus=np.where((up>dn)&(up>0),up,0); minus=np.where((dn>up)&(dn>0),dn,0)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/p,adjust=False).mean()
    pdi=100*pd.Series(plus,index=c.index).ewm(alpha=1/p,adjust=False).mean()/atr
    mdi=100*pd.Series(minus,index=c.index).ewm(alpha=1/p,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/p,adjust=False).mean()

def metrics(x):
    x=x.dropna()
    if len(x)<60: return None
    c=x["Close"].astype(float); h=x["High"].astype(float); l=x["Low"].astype(float); v=x["Volume"].astype(float)
    ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean()
    mac=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); sig=mac.ewm(span=9,adjust=False).mean()
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=tr.ewm(alpha=1/14,adjust=False).mean()
    avgv=v.rolling(20).mean(); r=rsi(c).iloc[-1]; a=adx(h,l,c).iloc[-1]; last=c.iloc[-1]
    upper=c.rolling(20).mean()+2*c.rolling(20).std(); lower=c.rolling(20).mean()-2*c.rolling(20).std()
    bb=(last-lower.iloc[-1])/(upper.iloc[-1]-lower.iloc[-1])*100 if upper.iloc[-1]!=lower.iloc[-1] else np.nan
    d20=(last/c.tail(20).max()-1)*100
    vals={"Price":last,"RSI":r,"MACD":mac.iloc[-1],"MACD Signal":sig.iloc[-1],"ADX":a,"Volume Ratio":v.iloc[-1]/avgv.iloc[-1],"1D %":(last/c.iloc[-2]-1)*100,"5D %":(last/c.iloc[-6]-1)*100,"20D %":(last/c.iloc[-21]-1)*100,"EMA20 %":(last/ema20.iloc[-1]-1)*100,"EMA50 %":(last/ema50.iloc[-1]-1)*100,"20D High %":d20,"ATR %":atr.iloc[-1]/last*100,"BB Position %":bb}
    if any(pd.isna(vals[k]) for k in ["RSI","ADX","Volume Ratio","EMA20 %","EMA50 %"]): return None
    s=0
    s+=15 if vals["EMA20 %"]>0 else 0
    s+=10 if vals["EMA50 %"]>0 else 0
    s+=15 if vals["MACD"]>vals["MACD Signal"] else 0
    s+=15 if vals["Volume Ratio"]>=1.2 else 0
    s+=10 if vals["ADX"]>=20 else 0
    s+=10 if 52<=vals["RSI"]<=68 else 0
    s+=10 if -2<=vals["20D High %"]<=0.5 else 0
    s+=5 if vals["5D %"]>0 else 0
    s+=5 if -1<=vals["1D %"]<=3 else 0
    s+=5 if 1<=vals["ATR %"]<=5 else 0
    vals["Setup Score"]=s
    return vals

u=universe()
col1,col2,col3=st.columns(3)
with col1: n=st.selectbox("Stocks to scan",[50,100,200,300,500],index=1)
with col2: threshold=st.slider("Minimum score",40,90,65,5)
with col3: minvol=st.slider("Minimum volume ratio",0.8,3.0,1.2,0.1)

symbols=u.Symbol.astype(str).head(n).tolist()
run=st.button("🔄 Scan for tomorrow",type="primary",use_container_width=True)

if run or "nextday" not in st.session_state:
    results=[]; progress=st.progress(0); status=st.empty()
    for start in range(0,len(symbols),50):
        chunk=symbols[start:start+50]; tickers=[s+".NS" for s in chunk]
        status.write(f"Downloading {start+1}–{start+len(chunk)} of {len(symbols)}…")
        try:
            data=yf.download(tickers=tickers,period="6mo",interval="1d",auto_adjust=True,group_by="ticker",threads=True,progress=False)
            for sym in chunk:
                try:
                    key=sym+".NS"
                    if isinstance(data.columns,pd.MultiIndex) and key in data.columns.get_level_values(0): x=data[key]
                    elif len(tickers)==1: x=data
                    else: continue
                    m=metrics(x)
                    if m: m["Symbol"]=sym; results.append(m)
                except Exception: continue
        except Exception: pass
        progress.progress(min(1,(start+len(chunk))/len(symbols))); time.sleep(.1)
    progress.empty(); status.empty(); st.session_state.nextday=pd.DataFrame(results)

df=st.session_state.nextday.copy()
if df.empty:
    st.warning("No technical data was returned. Try Scan again in a minute.")
else:
    df=df[df["Volume Ratio"]>=minvol]
    df=df[df["Setup Score"]>=threshold].sort_values(["Setup Score","Volume Ratio"],ascending=False)
    st.success(f"Found {len(df)} candidates meeting the selected setup rules.")
    st.dataframe(df.head(25),use_container_width=True,height=650)
    st.download_button("⬇️ Download candidates",df.to_csv(index=False).encode(),"next_day_momentum.csv","text/csv")
    st.markdown("### 🎯 How to use this")
    st.write("For a possible +2–3% next-day move, focus on higher Setup Score, volume above its 20-day average, RSI around 52–68, bullish MACD, price above EMA20/EMA50 and a close near the 20-day high. These are screening signals—not a forecast.")
    st.warning("A 2–3% move cannot be reliably guaranteed from technical indicators alone. Check fresh news/results, market direction, liquidity and circuit limits before taking any trade. This is educational research, not investment advice.")
