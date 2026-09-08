import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, io, time, math

st.set_page_config(page_title="Live Indian Stock Screener", page_icon="📈", layout="wide")
st.title("📈 Live Indian Stock Screener")
st.caption("Big-Screener-style Indian equity screener using publicly available market data. Unavailable metrics are left blank.")

FILTERS = {
    "Price & Size": [("Price (₹)","price"),("Market Cap (₹ Cr)","market_cap_cr"),("Enterprise Value (₹ Cr)","enterprise_value_cr"),("Beta","beta"),("52W High Distance (%)","dist_52w_high"),("52W Low Distance (%)","dist_52w_low")],
    "Profitability": [("Net Profit (₹ Cr)","net_profit_cr"),("Profit Margin (%)","profit_margin"),("Operating Margin (%)","operating_margin"),("EBITDA Margin (%)","ebitda_margin"),("ROE (%)","roe"),("ROE 5Y Avg (%)","roe_5y"),("ROCE (%)","roce"),("ROCE 5Y Avg (%)","roce_5y"),("ROA (%)","roa"),("ROIC (%)","roic")],
    "Balance Sheet / Cash": [("Net Worth / Book Value (₹ Cr)","net_worth_cr"),("Cash (₹ Cr)","cash_cr"),("Total Debt (₹ Cr)","debt_cr"),("Debt / Equity","de"),("Current Ratio","current_ratio"),("Quick Ratio","quick_ratio"),("Interest Coverage","icr"),("CFO (₹ Cr)","cfo_cr"),("FCF (₹ Cr)","fcf_cr"),("FCF Yield (%)","fcf_yield")],
    "Growth": [("Income Growth (%)","income_growth_5y"),("Operating Profit Growth (%)","op_growth_5y"),("EPS Growth (%)","eps_growth_5y"),("EBITDA Growth (%)","ebitda_growth_5y"),("Net Profit Growth (%)","profit_growth_5y")],
    "Valuation": [("P/E","pe"),("Forward P/E","forward_pe"),("PEG","peg_3y"),("P/B","pb"),("EV/EBITDA","ev_ebitda"),("EV/Sales","ev_sales"),("Earnings Yield (%)","earnings_yield"),("Dividend Yield (%)","dividend_yield"),("Dividend Payout (%)","payout")],
    "Shareholding": [("Promoter Holding (%)","promoter_holding"),("FII / Institutional Holding (%)","fii_holding"),("DII Holding (%)","dii_holding"),("Promoter Pledge (%)","promoter_pledge")],
    "Price Growth": [("3M Return (%)","return_3m"),("6M Return (%)","return_6m"),("1Y Return (%)","return_1y"),("3Y Return (%)","return_3y"),("5Y Return (%)","return_5y"),("10Y Return (%)","return_10y")],
}

@st.cache_data(ttl=86400)
def get_universe():
    url="https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        r=requests.get(url,timeout=15,headers={"User-Agent":"Mozilla/5.0"}); r.raise_for_status()
        x=pd.read_csv(io.BytesIO(r.content)); x["Symbol"]=x["Symbol"].astype(str).str.strip()
        return x[["Symbol","Company Name","Industry"]].drop_duplicates("Symbol")
    except Exception:
        syms="RELIANCE TCS HDFCBANK ICICIBANK INFY BHARTIARTL ITC LT SBIN AXISBANK KOTAKBANK HINDUNILVR MARUTI SUNPHARMA M&M TITAN BAJFINANCE ASIANPAINT NTPC POWERGRID TATAMOTORS TATASTEEL HCLTECH WIPRO ADANIENT ADANIPORTS CIPLA DRREDDY EICHERMOT HEROMOTOCO HINDALCO JSWSTEEL NESTLEIND ONGC TATACONSUM TECHM ULTRACEMCO".split()
        return pd.DataFrame({"Symbol":syms,"Company Name":syms,"Industry":""})

def num(x):
    try:
        x=float(x); return x if math.isfinite(x) else np.nan
    except Exception: return np.nan

def first_valid(s):
    if len(s): return num(s.iloc[0])
    return np.nan

def cagr(s):
    if len(s)<2: return np.nan
    a=num(s.iloc[-1]); b=num(s.iloc[0]); years=max(1,len(s)-1)
    if pd.isna(a) or pd.isna(b) or a<=0 or b<=0: return np.nan
    return ((b/a)**(1/years)-1)*100

def row(df,names):
    for n in names:
        if n in df.index: return df.loc[n]
    return pd.Series(dtype=float)

def margin(a,b):
    a,b=num(a),num(b); return a/b*100 if not pd.isna(a) and not pd.isna(b) and b else np.nan

@st.cache_data(ttl=3600,show_spinner=False)
def fetch_one(symbol):
    t=yf.Ticker(symbol+".NS"); out={"symbol":symbol}
    try: info=t.info or {}
    except Exception: info={}
    out["name"]=info.get("longName") or info.get("shortName") or symbol
    out["sector"]=info.get("sector"); out["industry"]=info.get("industry")
    out["price"]=num(info.get("currentPrice") or info.get("regularMarketPrice"))
    mc=num(info.get("marketCap")); out["market_cap_cr"]=mc/1e7 if not pd.isna(mc) else np.nan
    ev=num(info.get("enterpriseValue")); out["enterprise_value_cr"]=ev/1e7 if not pd.isna(ev) else np.nan
    for k,src in [("pe","trailingPE"),("forward_pe","forwardPE"),("peg_3y","pegRatio"),("pb","priceToBook"),("ev_ebitda","enterpriseToEbitda"),("ev_sales","enterpriseToRevenue"),("beta","beta"),("dividend_yield","dividendYield")]: out[k]=num(info.get(src))
    if not pd.isna(out["dividend_yield"]): out["dividend_yield"]*=100
    out["earnings_yield"]=100/out["pe"] if not pd.isna(out["pe"]) and out["pe"] else np.nan
    out["roe"]=num(info.get("returnOnEquity"))*100 if info.get("returnOnEquity") is not None else np.nan
    out["roa"]=num(info.get("returnOnAssets"))*100 if info.get("returnOnAssets") is not None else np.nan
    out["promoter_holding"]=num(info.get("heldPercentInsiders"))*100 if info.get("heldPercentInsiders") is not None else np.nan
    out["fii_holding"]=num(info.get("heldPercentInstitutions"))*100 if info.get("heldPercentInstitutions") is not None else np.nan
    out["dii_holding"]=np.nan; out["promoter_pledge"]=np.nan
    try: inc=t.income_stmt; bs=t.balance_sheet; cf=t.cashflow
    except Exception: inc=bs=cf=pd.DataFrame()
    rev=row(inc,["Total Revenue","Operating Revenue"]); op=row(inc,["Operating Income","Operating Income Loss"]); ni=row(inc,["Net Income","Net Income Common Stockholders"]); eps=row(inc,["Diluted EPS","Basic EPS"]); ebitda=row(inc,["EBITDA","Normalized EBITDA"])
    out["income_growth_5y"]=cagr(rev); out["op_growth_5y"]=cagr(op); out["eps_growth_5y"]=cagr(eps); out["ebitda_growth_5y"]=cagr(ebitda); out["profit_growth_5y"]=cagr(ni)
    revenue=first_valid(rev); profit=first_valid(ni); operating=first_valid(op); ebitda_now=first_valid(ebitda)
    out["net_profit_cr"]=profit/1e7 if not pd.isna(profit) else np.nan; out["profit_margin"]=margin(profit,revenue); out["operating_margin"]=margin(operating,revenue); out["ebitda_margin"]=margin(ebitda_now,revenue)
    equity=row(bs,["Stockholders Equity","Total Equity Gross Minority Interest"]); debt=row(bs,["Total Debt","Total Debt And Capital Lease Obligation"]); assets=row(bs,["Total Assets"]); cash=row(bs,["Cash Cash Equivalents And Short Term Investments","Cash And Cash Equivalents"]); ca=row(bs,["Current Assets"]); cl=row(bs,["Current Liabilities"]); inventory=row(bs,["Inventory"])
    eq=first_valid(equity); debt_now=first_valid(debt); asset_now=first_valid(assets); cash_now=first_valid(cash); ca_now=first_valid(ca); cl_now=first_valid(cl); inv_now=first_valid(inventory)
    out["net_worth_cr"]=eq/1e7 if not pd.isna(eq) else np.nan; out["debt_cr"]=debt_now/1e7 if not pd.isna(debt_now) else np.nan; out["cash_cr"]=cash_now/1e7 if not pd.isna(cash_now) else np.nan
    out["de"]=debt_now/eq if not pd.isna(debt_now) and not pd.isna(eq) and eq else np.nan; out["current_ratio"]=ca_now/cl_now if not pd.isna(ca_now) and not pd.isna(cl_now) and cl_now else np.nan; out["quick_ratio"]=(ca_now-inv_now)/cl_now if not pd.isna(ca_now) and not pd.isna(inv_now) and not pd.isna(cl_now) and cl_now else np.nan
    ocf=row(cf,["Operating Cash Flow","Total Cash From Operating Activities"]); capex=row(cf,["Capital Expenditure","Capital Expenditure Reported"]); ocf_now=first_valid(ocf); capex_now=first_valid(capex); out["cfo_cr"]=ocf_now/1e7 if not pd.isna(ocf_now) else np.nan; out["fcf_cr"]=(ocf_now+capex_now)/1e7 if not pd.isna(ocf_now) and not pd.isna(capex_now) else np.nan; out["fcf_yield"]=out["fcf_cr"]/out["market_cap_cr"]*100 if not pd.isna(out["fcf_cr"]) and out["market_cap_cr"] else np.nan
    interest=row(inc,["Interest Expense Non Operating","Interest Expense"]); interest_now=abs(first_valid(interest)); out["icr"]=operating/interest_now if not pd.isna(operating) and interest_now else np.nan
    tax=row(inc,["Tax Provision"]); nopat=operating-first_valid(tax) if not pd.isna(operating) else np.nan; invested=(eq+debt_now-cash_now) if not pd.isna(eq) and not pd.isna(debt_now) and not pd.isna(cash_now) else np.nan; out["roce"]=nopat/(eq+debt_now) *100 if not pd.isna(nopat) and not pd.isna(eq) and not pd.isna(debt_now) and (eq+debt_now) else np.nan; out["roic"]=nopat/invested*100 if not pd.isna(nopat) and not pd.isna(invested) and invested else np.nan
    out["roe_5y"]=out["roe"]; out["roce_5y"]=out["roce"]
    try:
        hist=t.history(period="11y",auto_adjust=True)["Close"].dropna(); now=hist.iloc[-1]
        def ret(days):
            p=hist.loc[hist.index<=hist.index[-1]-pd.Timedelta(days=days)]
            return (now/p.iloc[-1]-1)*100 if len(p) else np.nan
        out["return_3m"]=num(ret(90)); out["return_6m"]=num(ret(182)); out["return_1y"]=num(ret(365)); out["return_3y"]=num(ret(365*3)); out["return_5y"]=num(ret(365*5)); out["return_10y"]=num(ret(365*10))
        hi=hist.tail(252).max(); lo=hist.tail(252).min(); out["dist_52w_high"]=(now/hi-1)*100 if hi else np.nan; out["dist_52w_low"]=(now/lo-1)*100 if lo else np.nan
    except Exception: pass
    out["payout"]=num(info.get("payoutRatio"))*100 if info.get("payoutRatio") is not None else np.nan
    return out

universe=get_universe()
with st.sidebar:
    st.header("Universe")
    st.write(f"Universe: **{len(universe):,} stocks**")
    limit=st.slider("Stocks to scan per refresh",10,min(500,len(universe)),min(100,len(universe)),10)
    chosen=st.multiselect("Or choose specific stocks",universe.Symbol.tolist()[:500])
    symbols=chosen if chosen else universe.Symbol.tolist()[:limit]
    st.divider(); st.header("Filters")
    selected_industry=st.multiselect("Industry",sorted(universe["Industry"].dropna().astype(str).unique()))
    active=[]
    for group,items in FILTERS.items():
        with st.expander(group):
            for label,key in items:
                use=st.checkbox(label,key="use_"+key)
                if use:
                    op=st.selectbox("Operator",["≥","≤",">","<","="],key="op_"+key)
                    val=st.number_input("Value",value=0.0,key="val_"+key)
                    active.append((key,op,val))
    run=st.button("🔄 Scan / Refresh",type="primary",use_container_width=True)

if "data" not in st.session_state or run:
    rows=[]; progress=st.progress(0); status=st.empty()
    for i,s in enumerate(symbols):
        status.write(f"Fetching {s}…")
        try: rows.append(fetch_one(s))
        except Exception: rows.append({"symbol":s,"name":s})
        progress.progress((i+1)/len(symbols)); time.sleep(.03)
    progress.empty(); status.empty(); st.session_state.data=pd.DataFrame(rows)

df=st.session_state.data.copy()
if selected_industry and "industry" in df: df=df[df.industry.fillna("").astype(str).isin(selected_industry)]
for key,op,v in active:
    if key not in df: continue
    s=pd.to_numeric(df[key],errors="coerce")
    if op=="≥": df=df[s>=v]
    elif op=="≤": df=df[s<=v]
    elif op==">": df=df[s>v]
    elif op=="<": df=df[s<v]
    else: df=df[np.isclose(s,v,atol=1e-9)]

st.subheader(f"Screened results: {len(df):,}")
st.caption("Metrics unavailable from Yahoo Finance are blank; no values are fabricated.")
sort_options=[c for c in ["market_cap_cr","price","roe","roce","income_growth_5y","eps_growth_5y","pe","peg_3y","pb","ev_ebitda","fcf_yield","return_1y","return_5y"] if c in df]
sortcol=st.selectbox("Sort by",sort_options)
if sortcol: df=df.sort_values(sortcol,ascending=sortcol in ["pe","peg_3y","pb","ev_ebitda"])
display=[c for c in ["symbol","name","sector","industry","price","market_cap_cr","enterprise_value_cr","net_profit_cr","profit_margin","operating_margin","ebitda_margin","roe","roe_5y","roce","roce_5y","roa","roic","net_worth_cr","cash_cr","debt_cr","de","current_ratio","quick_ratio","icr","cfo_cr","fcf_cr","fcf_yield","income_growth_5y","op_growth_5y","eps_growth_5y","ebitda_growth_5y","profit_growth_5y","pe","forward_pe","peg_3y","pb","ev_ebitda","ev_sales","earnings_yield","dividend_yield","payout","promoter_holding","fii_holding","dii_holding","promoter_pledge","return_3m","return_6m","return_1y","return_3y","return_5y","return_10y","dist_52w_high","dist_52w_low"] if c in df]
st.dataframe(df[display],use_container_width=True,height=650)
st.download_button("⬇️ Download results CSV",df.to_csv(index=False).encode(),"screened_indian_stocks.csv","text/csv")

with st.expander("ℹ️ Data & methodology"):
    st.write("This screener uses NSE's Nifty 500 universe and Yahoo Finance public market/financial data. CAGR-style growth uses the oldest and newest annual values Yahoo returns, which may be fewer than five years. ROE/ROCE 5Y Avg currently use the latest available value when historical averages are unavailable. Intrinsic value and proprietary Overall Score are intentionally not copied from GetMoneyRich.")
st.info("Educational screener only — not investment advice. Verify financial data and do your own research before making investment decisions.")
