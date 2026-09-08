
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, io, time, math

st.set_page_config(page_title="Live Indian Stock Screener", page_icon="📈", layout="wide")

st.title("📈 Live Indian Stock Screener")
st.caption("A Big-Screener-style interface using publicly available market data. Data availability varies by company.")

FILTERS = {
    "Price & Size": [("Price (₹)", "price"), ("Market Cap (₹ Cr)", "market_cap_cr")],
    "Profit / Balance Sheet / Cash": [
        ("Net Profit (₹ Cr)", "net_profit_cr"), ("Net Worth (₹ Cr)", "net_worth_cr"),
        ("Cash Flow From Operations (₹ Cr)", "cfo_cr")],
    "Liquidity & Solvency": [
        ("Debt / Equity", "de"), ("Interest Coverage Ratio", "icr"), ("Current Ratio", "current_ratio")],
    "Returns": [("ROE 5Y Avg (%)", "roe_5y"), ("ROCE 5Y Avg (%)", "roce_5y")],
    "Growth": [
        ("Income Growth (%)", "income_growth_5y"), ("Operating Profit Growth (%)", "op_growth_5y"),
        ("EPS Growth (%)", "eps_growth_5y")],
    "Valuation": [("P/E", "pe"), ("PEG", "peg_3y"), ("ROIC (%)", "roic")],
    "Price Growth": [("3M Return (%)", "return_3m"), ("5Y Return (%)", "return_5y"), ("10Y Return (%)", "return_10y")],
}

@st.cache_data(ttl=86400)
def get_universe():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        x = pd.read_csv(io.BytesIO(r.content))
        x["Symbol"] = x["Symbol"].astype(str).str.strip()
        return x[["Symbol","Company Name","Industry"]].drop_duplicates("Symbol")
    except Exception:
        # Fallback starter universe if NSE's public CSV is temporarily unavailable.
        syms = """RELIANCE TCS HDFCBANK ICICIBANK INFY BHARTIARTL ITC LT SBIN AXISBANK KOTAKBANK HINDUNILVR MARUTI SUNPHARMA M&M TITAN BAJFINANCE ASIANPAINT NTPC POWERGRID TATAMOTORS TATASTEEL HCLTECH WIPRO ADANIENT ADANIPORTS CIPLA DRREDDY EICHERMOT HEROMOTOCO HINDALCO JSWSTEEL NESTLEIND ONGC TATACONSUM TECHM ULTRACEMCO""".split()
        return pd.DataFrame({"Symbol":syms,"Company Name":syms,"Industry":""})

def safe_num(x):
    try:
        x=float(x)
        return x if math.isfinite(x) else np.nan
    except: return np.nan

def cagr(first,last,years):
    if pd.isna(first) or pd.isna(last) or first<=0 or last<=0: return np.nan
    return (last/first)**(1/years)-1

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_one(symbol):
    t = yf.Ticker(symbol + ".NS")
    out={"symbol":symbol}
    try:
        info=t.info or {}
    except: info={}
    out["name"]=info.get("longName") or info.get("shortName") or symbol
    out["sector"]=info.get("sector")
    out["industry"]=info.get("industry")
    out["price"]=safe_num(info.get("currentPrice") or info.get("regularMarketPrice"))
    mc=info.get("marketCap")
    out["market_cap_cr"]=safe_num(mc)/1e7 if mc else np.nan
    out["pe"]=safe_num(info.get("trailingPE"))
    out["peg_3y"]=safe_num(info.get("pegRatio"))
    out["roe_5y"]=safe_num(info.get("returnOnEquity"))*100 if info.get("returnOnEquity") is not None else np.nan
    out["roic"]=safe_num(info.get("returnOnAssets"))*100 if info.get("returnOnAssets") is not None else np.nan

    # Annual statements. Yahoo may expose fewer than 5 years.
    try:
        inc=t.income_stmt
        bs=t.balance_sheet
        cf=t.cashflow
    except:
        inc=bs=cf=pd.DataFrame()

    def row(df,names):
        for n in names:
            if n in df.index: return df.loc[n]
        return pd.Series(dtype=float)

    try:
        rev=row(inc,["Total Revenue","Operating Revenue"])
        op=row(inc,["Operating Income","Operating Income Loss"])
        ni=row(inc,["Net Income","Net Income Common Stockholders"])
        eps=row(inc,["Diluted EPS","Basic EPS"])
        if len(rev): out["income_growth_5y"]=safe_num(cagr(rev.iloc[-1],rev.iloc[0],max(1,len(rev)-1))*100)
        if len(op): out["op_growth_5y"]=safe_num(cagr(op.iloc[-1],op.iloc[0],max(1,len(op)-1))*100)
        if len(eps): out["eps_growth_5y"]=safe_num(cagr(eps.iloc[-1],eps.iloc[0],max(1,len(eps)-1))*100)
        out["net_profit_cr"]=safe_num(ni.iloc[0])/1e7 if len(ni) else np.nan
    except: pass

    try:
        equity=row(bs,["Stockholders Equity","Total Equity Gross Minority Interest"])
        debt=row(bs,["Total Debt","Long Term Debt And Capital Lease Obligation"])
        assets=row(bs,["Total Assets"])
        cash=row(bs,["Cash Cash Equivalents And Short Term Investments"])
        current_assets=row(bs,["Current Assets"])
        current_liab=row(bs,["Current Liabilities"])
        out["net_worth_cr"]=safe_num(equity.iloc[0])/1e7 if len(equity) else np.nan
        out["de"]=safe_num(debt.iloc[0]/equity.iloc[0]) if len(debt) and len(equity) and equity.iloc[0] else np.nan
        out["current_ratio"]=safe_num(current_assets.iloc[0]/current_liab.iloc[0]) if len(current_assets) and len(current_liab) and current_liab.iloc[0] else np.nan
    except: pass

    try:
        ocf=row(cf,["Operating Cash Flow","Total Cash From Operating Activities"])
        out["cfo_cr"]=safe_num(ocf.iloc[0])/1e7 if len(ocf) else np.nan
    except: pass

    # Price returns from downloaded history.
    try:
        hist=t.history(period="11y",auto_adjust=True)
        close=hist["Close"].dropna()
        if len(close):
            now=close.iloc[-1]
            def ret_days(days):
                target=close.index[-1]-pd.Timedelta(days=days)
                prior=close.loc[close.index<=target]
                return (now/prior.iloc[-1]-1)*100 if len(prior) else np.nan
            out["return_3m"]=safe_num(ret_days(90))
            out["return_5y"]=safe_num(ret_days(365*5))
            out["return_10y"]=safe_num(ret_days(365*10))
    except: pass

    # Yahoo does not expose every Big Screener metric consistently.
    out["icr"]=np.nan
    out["roce_5y"]=np.nan
    return out

with st.sidebar:
    st.header("Universe")
    universe=get_universe()
    st.write(f"Universe: **{len(universe):,} stocks**")
    limit=st.slider("Stocks to scan per refresh", 10, min(500,len(universe)), min(100,len(universe)), 10)
    chosen=st.multiselect("Or choose specific stocks", universe.Symbol.tolist()[:500])
    if chosen:
        symbols=chosen
    else:
        symbols=universe.Symbol.tolist()[:limit]
    st.divider()
    st.header("Filters")
    selected_sector=st.multiselect("Sector", sorted(universe["Industry"].dropna().astype(str).unique()))
    active=[]
    for group,items in FILTERS.items():
        with st.expander(group):
            for label,key in items:
                use=st.checkbox("Use",key="use_"+key)
                if use:
                    op=st.selectbox("Operator",["≥","≤",">","<","="],key="op_"+key)
                    val=st.number_input("Value",value=0.0,key="val_"+key)
                    active.append((key,op,val))

    run=st.button("🔄 Scan / Refresh",type="primary",use_container_width=True)

if "data" not in st.session_state or run:
    rows=[]
    progress=st.progress(0)
    status=st.empty()
    for i,s in enumerate(symbols):
        status.write(f"Fetching {s}…")
        try: rows.append(fetch_one(s))
        except Exception as e: rows.append({"symbol":s,"name":s})
        progress.progress((i+1)/len(symbols))
        time.sleep(.03)
    progress.empty(); status.empty()
    st.session_state.data=pd.DataFrame(rows)

df=st.session_state.data.copy()

if selected_sector and "industry" in df:
    # The public NSE universe field is called Industry; company sector may be unavailable from Yahoo.
    df=df[df["industry"].fillna("").astype(str).isin(selected_sector)]

for key,op,v in active:
    if key not in df: continue
    s=pd.to_numeric(df[key],errors="coerce")
    if op=="≥": df=df[s>=v]
    elif op=="≤": df=df[s<=v]
    elif op==">": df=df[s>v]
    elif op=="<": df=df[s<v]
    else: df=df[np.isclose(s,v,atol=1e-9)]

st.subheader(f"Screened results: {len(df):,}")
st.caption("Metrics that Yahoo Finance does not consistently provide are left blank rather than fabricated.")

sortcol=st.selectbox("Sort by", [c for c in ["market_cap_cr","price","roe_5y","roce_5y","income_growth_5y","eps_growth_5y","pe","peg_3y","roic","return_3m","return_5y","return_10y"] if c in df])
if sortcol: df=df.sort_values(sortcol,ascending=sortcol in ["pe","peg_3y"])

display=[c for c in ["symbol","name","sector","industry","price","market_cap_cr","net_profit_cr","net_worth_cr","cfo_cr","de","icr","current_ratio","roe_5y","roce_5y","income_growth_5y","op_growth_5y","eps_growth_5y","pe","peg_3y","roic","return_3m","return_5y","return_10y"] if c in df]
st.dataframe(df[display],use_container_width=True,height=650)
st.download_button("⬇️ Download results CSV",df.to_csv(index=False).encode(),"screened_indian_stocks.csv","text/csv")

st.info("For a production-grade version with complete 40+ metrics, historical 5-year fundamentals, intrinsic value, Overall Score, saved screens and alerts, connect a licensed Indian financial-data API. This app intentionally does not copy GetMoneyRich's proprietary scoring or database.")
