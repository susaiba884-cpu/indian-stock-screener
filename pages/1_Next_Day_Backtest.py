import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, io, time

st.set_page_config(page_title="Next-Day Backtest", page_icon="🧪", layout="wide")
st.title("🧪 Next-Day Momentum Backtest")
st.caption("Tests the same momentum-style setup historically. It measures what happened on the next trading day; it does not predict or guarantee future returns.")

@st.cache_data(ttl=86400)
def universe():
    try:
        r=requests.get("https://archives.nseindia.com/content/indices/ind_nifty500list.csv",timeout=15,headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status(); x=pd.read_csv(io.BytesIO(r.content))
        return x[["Symbol","Company Name","Industry"]].drop_duplicates("Symbol")
    except Exception:
        syms="RELIANCE TCS HDFCBANK ICICIBANK INFY BHARTIARTL ITC LT SBIN AXISBANK KOTAKBANK TITAN BAJFINANCE SUNPHARMA MARUTI M&M NTPC POWERGRID TATAMOTORS TATASTEEL HCLTECH WIPRO ADANIENT CIPLA DRREDDY EICHERMOT JSWSTEEL ONGC TECHM ULTRACEMCO".split()
        return pd.DataFrame({"Symbol":syms,"Company Name":syms,"Industry":""})

def rsi(c,p=14):
    d=c.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/p,adjust=False).mean(); ad=dn.ewm(alpha=1/p,adjust=False).mean()
    return 100-100/(1+au/ad.replace(0,np.nan))

def adx(h,l,c,p=14):
    up=h.diff(); dn=-l.diff()
    plus=np.where((up>dn)&(up>0),up,0); minus=np.where((dn>up)&(dn>0),dn,0)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/p,adjust=False).mean()
    pdi=100*pd.Series(plus,index=c.index).ewm(alpha=1/p,adjust=False).mean()/atr
    mdi=100*pd.Series(minus,index=c.index).ewm(alpha=1/p,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/p,adjust=False).mean()

def daily_setup(x):
    x=x.dropna().copy()
    if len(x)<60: return pd.DataFrame()
    c=x["Close"].astype(float); h=x["High"].astype(float); l=x["Low"].astype(float); v=x["Volume"].astype(float)
    ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean()
    mac=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); sig=mac.ewm(span=9,adjust=False).mean()
    a=adx(h,l,c); rr=rsi(c); avgv=v.rolling(20).mean(); tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=tr.ewm(alpha=1/14,adjust=False).mean()
    high20=c.rolling(20).max()
    score=(ema20.gt(0)*0).astype(float)
    score += np.where(c>ema20,15,0)+np.where(c>ema50,10,0)+np.where(mac>sig,15,0)+np.where(v/avgv>=1.2,15,0)+np.where(a>=20,10,0)+np.where((rr>=52)&(rr<=68),10,0)+np.where(((c/high20-1)*100>=-2)&((c/high20-1)*100<=0.5),10,0)+np.where(c.pct_change(5)>0,5,0)+np.where((c.pct_change()>-0.01)&(c.pct_change()<0.03),5,0)+np.where(((atr/c)*100>=1)&((atr/c)*100<=5),5,0)
    out=pd.DataFrame(index=c.index)
    out["Score"]=score
    out["Volume Ratio"]=v/avgv
    out["Next Day %"]=c.shift(-1)/c-1
    out["Next Day +2%"]=(out["Next Day %"]>=0.02)
    out["Next Day +3%"]=(out["Next Day %"]>=0.03)
    out["Next Day +1%"]=(out["Next Day %"]>=0.01)
    out["Next Day <=-2%"]=(out["Next Day %"]<=-0.02)
    return out.dropna()

u=universe()
c1,c2,c3=st.columns(3)
with c1: n=st.selectbox("Stocks",[25,50,100,200,500],index=2)
with c2: years=st.selectbox("History",[1,2,3,5],index=2)
with c3: minscore=st.slider("Minimum setup score",40,90,65,5)

symbols=u.Symbol.astype(str).head(n).tolist()
run=st.button("🧪 Run Backtest",type="primary",use_container_width=True)

if run or "bt" not in st.session_state:
    allrows=[]; progress=st.progress(0); status=st.empty()
    for start in range(0,len(symbols),50):
        chunk=symbols[start:start+50]; tickers=[s+".NS" for s in chunk]
        status.write(f"Testing {start+1}–{start+len(chunk)} of {len(symbols)}…")
        try:
            data=yf.download(tickers=tickers,period=f"{years}y",interval="1d",auto_adjust=True,group_by="ticker",threads=True,progress=False)
            for sym in chunk:
                try:
                    key=sym+".NS"
                    if isinstance(data.columns,pd.MultiIndex) and key in data.columns.get_level_values(0): x=data[key]
                    elif len(tickers)==1: x=data
                    else: continue
                    z=daily_setup(x)
                    if z.empty: continue
                    z=z[z["Score"]>=minscore].copy()
                    if z.empty: continue
                    z["Symbol"]=sym; allrows.append(z.reset_index())
                except Exception: continue
        except Exception: pass
        progress.progress(min(1,(start+len(chunk))/len(symbols))); time.sleep(.05)
    progress.empty(); status.empty()
    st.session_state.bt=pd.concat(allrows,ignore_index=True) if allrows else pd.DataFrame()

df=st.session_state.bt.copy()
if df.empty:
    st.warning("No qualifying historical setups were returned. Try a lower score or fewer/shorter history settings.")
else:
    nextret=pd.to_numeric(df["Next Day %"],errors="coerce")*100
    st.subheader(f"Historical setups tested: {len(df):,}")
    a,b,c,d=st.columns(4)
    a.metric("Avg next-day return",f"{nextret.mean():.2f}%")
    b.metric("Median next-day return",f"{nextret.median():.2f}%")
    c.metric("Hit rate ≥ +2%",f"{(nextret>=2).mean()*100:.1f}%")
    d.metric("Hit rate ≥ +3%",f"{(nextret>=3).mean()*100:.1f}%")
    e,f=st.columns(2)
    e.metric("Hit rate ≥ +1%",f"{(nextret>=1).mean()*100:.1f}%")
    f.metric("Downside ≤ −2%",f"{(nextret<=-2).mean()*100:.1f}%")
    st.markdown("### Score bucket results")
    df["Score Bucket"]=pd.cut(df["Score"],bins=[39,49,59,69,79,89,100],labels=["40–49","50–59","60–69","70–79","80–89","90–100"])
    g=df.groupby("Score Bucket",observed=False).agg(Setups=("Next Day %","size"),Avg_Next_Day=("Next Day %","mean"),Median_Next_Day=("Next Day %","median"),Hit_2pct=("Next Day +2%","mean"),Hit_3pct=("Next Day +3%","mean"),Downside_2pct=("Next Day <=-2%","mean")).reset_index()
    g["Avg_Next_Day"]=g["Avg_Next_Day"]*100; g["Median_Next_Day"]=g["Median_Next_Day"]*100; g["Hit_2pct"]=g["Hit_2pct"]*100; g["Hit_3pct"]=g["Hit_3pct"]*100; g["Downside_2pct"]=g["Downside_2pct"]*100
    st.dataframe(g.rename(columns={"Avg_Next_Day":"Avg next-day %","Median_Next_Day":"Median next-day %","Hit_2pct":"≥+2% %","Hit_3pct":"≥+3% %","Downside_2pct":"≤−2% %"}),use_container_width=True)
    st.markdown("### Best historical candidates by observed setup")
    cols=[c for c in ["Date","Symbol","Score","Volume Ratio","Next Day %"] if c in df]
    st.dataframe(df.sort_values(["Score","Volume Ratio"],ascending=False)[cols].head(50),use_container_width=True,height=500)
    st.download_button("⬇️ Download backtest observations",df.to_csv(index=False).encode(),"next_day_backtest.csv","text/csv")
    st.info("Important: this is a historical test of the rules, not a prediction. Results can change with market regime, slippage, taxes, liquidity, corporate actions and the exact data source. Do not treat a high hit rate as a guarantee.")
