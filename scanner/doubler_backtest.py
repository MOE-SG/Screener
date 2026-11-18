#!/usr/bin/env python3
"""
Doubler backtest
================
Replays every trading day of the last 10 years (or any range) on the Doubler watchlist,
using the same entry rules as the Doubler app, and lists every date where at least
one stock met them. Each signal is then traded with the Plan tab's exit rules.

Requirements (once):   pip install yfinance pandas numpy openpyxl
Run:                   python doubler_backtest.py
Options:               python doubler_backtest.py --start-year 2017 --end-year 2026   (default: last 10 years)
                       python doubler_backtest.py --year 2026            (one year only)
                       python doubler_backtest.py --account 20000 --risk 0.5 --compound
                       python doubler_backtest.py --check NVDA           (adds NVDA's daily rule columns to verify)
                       python doubler_backtest.py --no-earnings      (faster, skips earnings dates)
                       python doubler_backtest.py --tickers mylist.txt
                       python doubler_backtest.py --demo             (synthetic data, for testing)

Data: Yahoo Finance via yfinance (free, no key). Prices are split-adjusted.

What is tested (technical rules only)
-------------------------------------
The app's 0-100 score also uses revenue/EPS growth and market cap from Finnhub, which
are only available as "today's" values, not as they were on past dates. Using them
here would leak future information, so this backtest uses the technical entry rules:

  Trend 4/4        close > 50-day SMA, 50-day rising (vs 10 days ago),
                   close > 200-day SMA, 200-day rising (vs 20 days ago)
  Breakout         close above the highest high of the previous 25 days,
                   that 25-day range is <= 20% deep (a sideways base),
                   volume >= 1.5x the 50-day average volume
  Not stretched    close is 0% to +10% above the 50-day SMA
  Earnings         no earnings report in the next 7 calendar days

  TRADE SIGNAL   = all four on the same day (the breakout day)
  SETUP 4/4 DAY  = the app's "Setup 4/4" tag: same, but the breakout may be up to
                   2 days old (still holding), or price may still be inside the base
                   within 5% of the pivot (a watch, not yet a trigger)

Also recorded per signal: market direction (SPY/QQQ, same rules as the app banner)
and relative strength vs the S&P 500.

Trade simulation (Plan tab defaults)
------------------------------------
  Entry      next day's open after the signal
  Stop       10-day low -0.5% if that is 2%..8% below entry, else 8% below entry
  Target 1   +25%: sell 1/3, raise stop to entry (breakeven)
  Double     +100%: sell another 1/3
  Trailing   after Target 1: sell the rest on a close below the 50-day SMA
             or 20% below the highest close since entry
  Time stop  after 10 weeks, if Target 1 not reached and close < entry +5%: sell
  Portfolio  a $10,000 account (--account), risking 1% per trade (--risk), max
             25% of the account in one stock (--maxpos). Signals are taken in date
             order; on busy days the strongest relative strength goes first. If the
             account has no cash left, the signal is reduced or skipped
             (--no-cash-limit to take all). The account is valued at market prices
             every day, so positions open at 31 Dec carry into the next year at that
             day's value, and each year's % is the true change in the account.
  If the stop and a target fall on the same day, the stop is assumed hit first
  (conservative). Trades still open at the end are marked to the last close.

Not financial advice. Past signals do not guarantee future results.
"""
import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WATCHLIST = """
PLTR SOFI HOOD RKLB IONQ HIMS APP AFRM UPST CELH ASTS OKLO SMCI CRWD NET DDOG SNOW SHOP MELI NU
TMDX AXON DUOL CAVA ELF ONON RDDT ALAB VRT ANET CRDO NVDA AMD MU TSLA COIN MSTR SOUN BBAI ACHR
NBIS CRWV IREN AVGO MRVL ARM TSM SNDK LITE COHR CLS AMAT ASML LRCX INTC APLD GOOGL META AMZN MSFT
ORCL PANW PATH UBER NFLX LLY NVO ABBV JNJ MRK ISRG BSX LMT RTX NOC GD LHX KTOS XOM CVX OXY SLB EOG
LNG CEG VST GEV FCX NEM B AA NUE CDE CAT DE ETN HON PH CMI COST WMT PG KO PEP CL PLD EQIX DLR O
AMT CCI JPM GS MS BAC C BRK.B PGR ABNB BKNG DAL UAL MAR CCL LUNR JOBY RIVN QBTS POET CRCL OPEN AAPL
QCOM WDC TTWO FSLR SE SBUX CVS STRL
""".split()
LEVERAGED = {"TQQQ", "SOXL"}          # not ranked in the app, so not tested
INDEXES = ["SPY", "QQQ"]

# Rule settings: same defaults as the app
BASE_DAYS, BASE_MAX, VOL_X = 25, 20.0, 1.5
EXT_MAX, EARN_DAYS = 10.0, 7
STOP_PCT, T1_PCT, TRAIL_PCT, TIME_WEEKS = 8.0, 25.0, 20.0, 10


def yahoo_symbol(t):
    return t.replace(".", "-")        # BRK.B -> BRK-B


# ---------------------------------------------------------------- data
def download(tickers, start, end):
    import yfinance as yf
    print(f"Downloading daily prices for {len(tickers)} symbols from Yahoo Finance...")
    ysyms = [yahoo_symbol(t) for t in tickers]
    raw = yf.download(ysyms, start=start, end=end, auto_adjust=True, group_by="ticker",
                      progress=True, threads=True)
    out = {}
    for t, y in zip(tickers, ysyms):
        try:
            df = raw[y][["Open", "High", "Low", "Close", "Volume"]].dropna()
        except KeyError:
            continue
        if len(df) > 60:
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            out[t] = df
    missing = [t for t in tickers if t not in out]
    if missing:
        print(f"  No data for: {' '.join(missing)}")
    return out


def earnings_dates(tickers, limit=16):
    import yfinance as yf
    print(f"Fetching past and upcoming earnings dates for {len(tickers)} stocks...")
    out = {}
    for i, t in enumerate(tickers, 1):
        try:
            ed = yf.Ticker(yahoo_symbol(t)).get_earnings_dates(limit=limit)
            if ed is not None and len(ed):
                out[t] = sorted({pd.Timestamp(x).tz_localize(None).normalize()
                                 if pd.Timestamp(x).tzinfo else pd.Timestamp(x).normalize()
                                 for x in ed.index})
        except Exception:
            pass
        if i % 20 == 0:
            print(f"  {i}/{len(tickers)}")
    print(f"  Earnings dates found for {len(out)} of {len(tickers)} stocks")
    return out


def demo_data(tickers, start, end, seed=7):
    """Synthetic prices with trends, bases and breakouts, for testing the script."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    out, earn = {}, {}
    for k, t in enumerate(tickers):
        n = len(days)
        drift = rng.normal(0.0006, 0.0012, n)
        regime = np.repeat(rng.normal(0, 0.003, n // 40 + 1), 40)[:n]
        r = drift + regime + rng.normal(0, 0.018 if t not in INDEXES else 0.009, n)
        close = 50 * np.exp(np.cumsum(r)) * (1 + k % 7)
        opn = close * (1 + rng.normal(0, 0.006, n))
        high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
        low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
        vol = rng.lognormal(14, 0.35, n) * (1 + 1.5 * (r > 0.03))
        out[t] = pd.DataFrame({"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol}, index=days)
        if t not in INDEXES:
            first = days[0] + pd.Timedelta(days=int(rng.integers(0, 90)))
            earn[t] = list(pd.date_range(first, days[-1] + pd.Timedelta(days=90), freq="91D"))
    return out, earn


# ---------------------------------------------------------------- indicators
def add_indicators(df):
    d = df.copy()
    c, h, l, v = d.Close, d.High, d.Low, d.Volume
    d["s50"] = c.rolling(50).mean()
    d["s200"] = c.rolling(200).mean()
    a50, r50 = c > d.s50, d.s50 > d.s50.shift(10)
    a200, r200 = c > d.s200, d.s200 > d.s200.shift(20)
    d["trend"] = a50.astype(int) + r50.astype(int) + a200.astype(int) + r200.astype(int)
    # base of the previous 25 days (excluding today), as in the app
    d["pivot_px"] = h.shift(1).rolling(BASE_DAYS).max()
    base_lo = l.shift(1).rolling(BASE_DAYS).min()
    d["depth"] = (d.pivot_px - base_lo) / d.pivot_px * 100
    d["volx"] = v / v.shift(1).rolling(50).mean()
    d["brk"] = (d.depth <= BASE_MAX) & (c > d.pivot_px)
    d["brk_vol"] = d.brk & (d.volx >= VOL_X)
    d["ext"] = (c / d.s50 - 1) * 100
    # app check 2: breakout on volume in the last 3 sessions and still above its pivot,
    # or (no recent breakout) inside a tight base within 5% of the pivot
    recent = pd.Series(False, index=d.index)
    any_brk = pd.Series(False, index=d.index)
    for k in range(3):
        any_brk |= d.brk.shift(k, fill_value=False)
        recent |= d.brk_vol.shift(k, fill_value=False) & (c > d.pivot_px.shift(k))
    hi_in = h.rolling(BASE_DAYS).max()
    lo_in = l.rolling(BASE_DAYS).min()
    near = (~any_brk) & ((hi_in - lo_in) / hi_in * 100 <= BASE_MAX) & ((c / hi_in - 1) * 100 >= -5)
    d["chk2"] = recent | near
    d["near_pivot"] = near
    d["low10"] = l.rolling(10).min()
    return d


def add_rs(d, spy_close):
    s = spy_close.reindex(d.index)
    w = {63: .4, 126: .2, 189: .2, 252: .2}
    ws = sum(wt * (d.Close / d.Close.shift(k) - 1) for k, wt in w.items())
    wm = sum(wt * (s / s.shift(k) - 1) for k, wt in w.items())
    d["rs"] = ((1 + ws) / (1 + wm) - 1) * 100
    ratio = d.Close / s
    d["rs_newhigh"] = ratio >= ratio.rolling(252, min_periods=120).max() * 0.99
    return d


def market_series(data):
    parts = []
    for t in INDEXES:
        if t not in data:
            continue
        d = add_indicators(data[t])
        pts = ((d.Close > d.s50).astype(int) + (d.s50 > d.s50.shift(10)).astype(int)
               + (d.Close > d.s200).astype(int))
        parts.append(pts.rename(t))
    if not parts:
        return None
    pts = pd.concat(parts, axis=1).dropna()
    total, mx = pts.sum(axis=1), 3 * pts.shape[1]
    base = data["SPY"] if "SPY" in data else data[INDEXES[1]]
    sell_day = (base.Close < base.Close.shift(1) * 0.998) & (base.Volume > base.Volume.shift(1))
    dd = sell_day.astype(int).rolling(25).sum().reindex(total.index)
    state = np.where(total >= mx - 1, "Uptrend", np.where(total <= mx // 3, "Downtrend", "Caution"))
    state = pd.Series(state, index=total.index)
    state[(state == "Uptrend") & (dd >= 5)] = "Caution"
    return pd.DataFrame({"market": state, "selling_days": dd})


def days_to_next_earnings(dates, day):
    if dates is None:
        return None
    fut = [e for e in dates if e >= day]
    return (fut[0] - day).days if fut else None


# ---------------------------------------------------------------- trade simulation
def simulate(d, i):
    """Trade a signal on row i with the Plan tab's default rules.
    Also returns the trade's daily mark-to-market P&L per share (_path), so open
    positions can be valued on any day, including year-ends."""
    if i + 1 >= len(d):
        return {"status": "No entry yet"}
    o, h, l, c, s50 = (d[x].values for x in ("Open", "High", "Low", "Close", "s50"))
    j0 = i + 1
    E = o[j0]
    pct_stop = E * (1 - STOP_PCT / 100)
    low10 = d.low10.values[i] * 0.995
    stop = low10 if (low10 < E * 0.98 and low10 > pct_stop) else pct_stop
    risk = E - stop
    T1, T2 = E * (1 + T1_PCT / 100), E * 2
    pos, realized, stage, high, cur_stop = 1.0, 0.0, 0, E, stop
    status, exit_day, exit_px, hit_t1, hit_t2 = "Open", None, None, None, None
    path_d, path_v = [], []
    for j in range(j0, len(d)):
        done = False
        if l[j] <= cur_stop:                                   # stop first (conservative)
            px = o[j] if o[j] <= cur_stop else cur_stop
            realized += pos * (px - E); pos = 0
            status = "Stopped" if stage == 0 else "Breakeven stop"
            exit_day, exit_px, done = d.index[j], px, True
        else:
            if stage < 1 and h[j] >= T1:
                realized += (1 / 3) * (T1 - E); pos -= 1 / 3; stage = 1
                cur_stop = max(cur_stop, E); hit_t1 = d.index[j]
            if stage < 2 and h[j] >= T2:
                realized += (1 / 3) * (T2 - E); pos -= 1 / 3; stage = 2; hit_t2 = d.index[j]
            high = max(high, c[j])
            if stage >= 1 and (c[j] < s50[j] or c[j] <= high * (1 - TRAIL_PCT / 100)):
                realized += pos * (c[j] - E); pos = 0
                status, exit_day, exit_px, done = "Trailing exit", d.index[j], c[j], True
            elif stage == 0 and (j - j0) >= TIME_WEEKS * 5 and c[j] < E * 1.05:
                realized += pos * (c[j] - E); pos = 0
                status, exit_day, exit_px, done = "Time stop", d.index[j], c[j], True
        path_d.append(d.index[j]); path_v.append(realized + pos * (c[j] - E))
        if done:
            break
    pnl = path_v[-1]
    held = ((exit_day if exit_day is not None else d.index[-1]) - d.index[j0]).days
    return {
        "entry_date": d.index[j0].date(), "entry": E, "stop": stop, "stop_pct": (1 - stop / E) * 100,
        "target1": T1, "hit_target1": hit_t1.date() if hit_t1 is not None else None,
        "hit_double": hit_t2.date() if hit_t2 is not None else None,
        "status": status, "exit_date": exit_day.date() if exit_day is not None else None,
        "exit_price": exit_px, "days_held": held,
        "return_pct": pnl / E * 100, "R": pnl / risk if risk > 0 else np.nan,
        "_risk_ps": risk, "_path": (pd.DatetimeIndex(path_d), np.array(path_v)),
    }


def fwd(d, i, k):
    return (d.Close.values[i + k] / d.Close.values[i] - 1) * 100 if i + k < len(d) else np.nan


def run_portfolio(sig, cal, account, risk_pct, maxpos_pct, compound, cash_limit):
    """Takes signals in date order (strongest relative strength first on busy days),
    sizes each by risk, and values the account every trading day at market prices."""
    n = len(cal)
    pnl = np.zeros(n)                     # account P&L, marked to market each day
    in_use = np.zeros(n)                  # cost of open positions
    n_open = np.zeros(n, dtype=int)
    shares, note = {}, {}
    order = sig[sig.entry.notna()].sort_values(["entry_date", "rs_vs_spx_pct"], ascending=[True, False],
                                               na_position="last")
    for k, r in order.iterrows():
        s_i = cal.searchsorted(pd.Timestamp(r.entry_date))
        if s_i >= n:
            note[k] = "After end of test"; continue
        e_i = cal.searchsorted(pd.Timestamp(r.exit_date)) if r.exit_date is not None and not pd.isna(r.exit_date) else n - 1
        e_i = min(e_i, n - 1)
        eq_prev = account + (pnl[s_i - 1] if s_i > 0 else 0.0)
        base = eq_prev if compound else account
        sh = math.floor(min(base * risk_pct / 100 / r._risk_ps, base * maxpos_pct / 100 / r.entry)) if r._risk_ps > 0 else 0
        if cash_limit:
            free = eq_prev - in_use[s_i]
            if sh * r.entry > free:
                sh = math.floor(max(free, 0) / r.entry)
                note[k] = "Reduced: not enough cash" if sh >= 1 else "Skipped: no cash free"
        if sh < 1:
            note.setdefault(k, "Skipped: position too small"); shares[k] = 0; continue
        pdates, pvals = r._path
        ser = pd.Series(pvals, index=pdates).reindex(cal[s_i:], method="ffill").fillna(0.0).values
        pnl[s_i:] += sh * ser
        in_use[s_i:e_i + 1] += sh * r.entry
        n_open[s_i:e_i + 1] += 1
        shares[k] = sh
        note.setdefault(k, "Taken")
    return pnl, in_use, n_open, shares, note


def check_dump(t, data, spy, earn, xw):
    """--check: one stock's full daily history with every rule column, to verify in Excel."""
    d = add_indicators(data[t])
    if spy is not None:
        d = add_rs(d, spy)
    cols = ["Open", "High", "Low", "Close", "Volume", "s50", "s200", "trend", "pivot_px", "depth", "volx",
            "brk", "brk_vol", "ext", "low10"] + (["rs", "rs_newhigh"] if "rs" in d else [])
    out = d[cols].copy()
    ed = [days_to_next_earnings(earn.get(t), x) for x in out.index]
    out["days_to_earnings"] = ed
    e_ok = np.array([e is None or e > EARN_DAYS for e in ed])
    out["TRADE_SIGNAL"] = (out.trend == 4) & out.brk_vol & (out.ext >= 0) & (out.ext <= EXT_MAX) & e_ok
    out = out.rename(columns={"s50": "SMA50", "s200": "SMA200", "trend": "trend_pts_0to4",
                              "pivot_px": "prev25_high", "depth": "base_depth_pct", "volx": "vol_x_50d_avg",
                              "brk": "close_above_prev25_high", "brk_vol": "...and_vol>=1.5x",
                              "ext": "pct_above_SMA50", "low10": "low_10d"})
    out.index.name = "date"
    out.reset_index().assign(date=lambda x: x.date.dt.date).round(4).to_excel(
        xw, sheet_name=f"Check {t}"[:31], index=False)


# ---------------------------------------------------------------- main
def main():
    today = dt.date.today()
    ap = argparse.ArgumentParser(description="Backtest Doubler's entry rules.")
    ap.add_argument("--start-year", type=int, default=today.year - 9, help="first year (default: 10 years ago)")
    ap.add_argument("--end-year", type=int, default=today.year, help="last year (default: this year)")
    ap.add_argument("--year", type=int, help="test a single year only")
    ap.add_argument("--tickers", help="text file with tickers (spaces/commas/new lines)")
    ap.add_argument("--out", help="output Excel file")
    ap.add_argument("--no-earnings", action="store_true", help="skip earnings dates (faster)")
    ap.add_argument("--account", type=float, default=10000, help="starting account size in $ (default 10000)")
    ap.add_argument("--risk", type=float, default=1.0, help="risk per trade, %% of account (default 1)")
    ap.add_argument("--maxpos", type=float, default=25.0, help="max position size, %% of account (default 25)")
    ap.add_argument("--compound", action="store_true", help="size trades from the current account value, not the start value")
    ap.add_argument("--no-cash-limit", action="store_true", help="take every signal even when the account has no cash left")
    ap.add_argument("--check", help="also write one ticker's full daily rule columns to a sheet, e.g. --check NVDA")
    ap.add_argument("--demo", action="store_true", help="use synthetic data to test the script")
    a = ap.parse_args()
    if a.year:
        a.start_year = a.end_year = a.year

    tickers = WATCHLIST
    if a.tickers:
        tickers = [x.strip().upper() for x in Path(a.tickers).read_text().replace(",", " ").split()]
    tickers = [t for t in dict.fromkeys(tickers) if t not in LEVERAGED]
    y0 = dt.date(a.start_year, 1, 1)
    y1 = min(dt.date(a.end_year, 12, 31), today)
    start = y0 - dt.timedelta(days=560)        # warm-up for 200-day SMA, 12-month RS
    end = today + dt.timedelta(days=1)
    n_years = a.end_year - a.start_year + 1
    label = str(a.start_year) if n_years == 1 else f"{a.start_year}-{a.end_year}"

    if a.demo:
        data, earn = demo_data(tickers + INDEXES, start, end)
        if a.no_earnings:
            earn = {}
    else:
        data = download(tickers + INDEXES, start, end)
        earn = {} if a.no_earnings else earnings_dates([t for t in tickers if t in data], limit=4 * n_years + 12)

    mkt = market_series(data)
    spy = data["SPY"].Close if "SPY" in data else None
    sig_rows, setup_rows = [], []
    funnel = dict(trend_brk_vol=0, ext_ok=0, earn_ok=0)
    late = []                                       # stocks listed after the test started

    for t in tickers:
        if t not in data:
            continue
        d = add_indicators(data[t])
        if d.index[0].date() > y0 - dt.timedelta(days=300):
            late.append(f"{t} ({d.index[0].date()})")
        if spy is not None:
            d = add_rs(d, spy)
        else:
            d["rs"], d["rs_newhigh"] = np.nan, False
        d = d[d.index.date <= y1]                   # trades never see prices after the test ends
        in_rng = (d.index.date >= y0) & (d.index.date <= y1)
        ext_ok = ((d.ext >= 0) & (d.ext <= EXT_MAX)).values
        sigmask = ((d.trend == 4) & d.brk_vol).values & in_rng
        setmask = ((d.trend == 4).values & ext_ok & d.chk2.values) & in_rng
        for i in np.where(sigmask | setmask)[0]:
            day = d.index[i]
            edays = days_to_next_earnings(earn.get(t), day)
            e_ok = edays is None or edays > EARN_DAYS
            row = d.iloc[i]
            if sigmask[i]:
                funnel["trend_brk_vol"] += 1
                if ext_ok[i]:
                    funnel["ext_ok"] += 1
                    funnel["earn_ok"] += int(e_ok)
            common = {
                "date": day.date(), "ticker": t, "close": row.Close,
                "50d_sma": row.s50, "200d_sma": row.s200, "pct_vs_50d": row.ext,
                "pivot": row.pivot_px, "base_depth_pct": row.depth, "volume_x_avg": row.volx,
                "rs_vs_spx_pct": row.rs, "rs_line_new_high": bool(row.rs_newhigh) if not pd.isna(row.rs_newhigh) else None,
                "market": mkt.market.get(day, "n/a") if mkt is not None else "n/a",
                "days_to_earnings": edays if edays is not None else ("n/a" if t not in earn else "none"),
            }
            if sigmask[i] and ext_ok[i] and e_ok:
                rec = dict(common)
                rec.update({"fwd_1m_pct": fwd(d, i, 21), "fwd_3m_pct": fwd(d, i, 63),
                            "max_gain_3m_pct": (d.High.values[i + 1:i + 64].max() / row.Close - 1) * 100 if i + 1 < len(d) else np.nan})
                rec.update(simulate(d, i))
                sig_rows.append(rec)
            if setmask[i] and e_ok:
                s = dict(common)
                s["kind"] = ("Breakout today" if sigmask[i] else
                             "Breakout, 1-2 days ago" if not d.near_pivot.iloc[i] else "In base, near pivot")
                setup_rows.append(s)

    # trading calendar = SPY's days (or the longest stock)
    ref = data["SPY"] if "SPY" in data else max(data.values(), key=len) if data else None
    if ref is None:
        print("No price data. Check your internet connection and that yfinance is up to date "
              "(pip install -U yfinance).")
        sys.exit(1)
    cal = ref.index[(ref.index.date >= y0) & (ref.index.date <= y1)]

    sig = pd.DataFrame(sig_rows)
    setups = pd.DataFrame(setup_rows)
    days = pd.DataFrame({"date": [x.date() for x in cal], "year": cal.year})
    if mkt is not None:
        days["market"] = [mkt.market.get(x, "n/a") for x in cal]
    days["signals"] = days.date.map(sig.groupby("date").size() if len(sig) else {}).fillna(0).astype(int)
    days["signal_tickers"] = days.date.map(sig.groupby("date").ticker.apply(lambda s: " ".join(sorted(s)))
                                           if len(sig) else {}).fillna("")
    days["setup_4of4_stocks"] = days.date.map(setups.groupby("date").size() if len(setups) else {}).fillna(0).astype(int)

    # ---------- portfolio, marked to market every day (open trades carry into the next year)
    cash_limit = not a.no_cash_limit
    if len(sig):
        pnl, in_use, n_open, shares, note = run_portfolio(sig, cal, a.account, a.risk, a.maxpos, a.compound, cash_limit)
        sig["shares"] = pd.Series(shares).reindex(sig.index).fillna(0).astype(int)
        sig["taken"] = pd.Series(note).reindex(sig.index).fillna("No entry yet")
        sig["cost_usd"] = sig.shares * sig.entry
        sig["pnl_usd"] = sig.shares * sig.return_pct / 100 * sig.entry
    else:
        pnl, in_use, n_open = np.zeros(len(cal)), np.zeros(len(cal)), np.zeros(len(cal), dtype=int)
    equity = a.account + pnl
    days["account_usd"] = equity.round(2)
    days["positions_open"] = n_open
    days["capital_in_use_usd"] = in_use.round(0)
    spy_px = spy.reindex(cal).ffill().values if spy is not None else None

    def max_dd(x):
        x = np.asarray(x, dtype=float)
        return ((x / np.maximum.accumulate(x)) - 1).min() * 100 if len(x) else 0.0

    # ---------- yearly table
    yrows = []
    prev_eq, prev_spy = a.account, (spy_px[0] if spy_px is not None else None)
    for y in range(a.start_year, a.end_year + 1):
        m = (cal.year == y)
        if not m.any():
            continue
        idx = np.where(m)[0]
        e_end = equity[idx[-1]]
        ys = sig[pd.to_datetime(sig.date).dt.year == y] if len(sig) else sig
        yt = ys[ys.get("shares", 0) > 0] if len(ys) else ys
        row = {
            "year": y, "trading_days": int(m.sum()),
            "days_with_signal": int((days.signals[m] > 0).sum()),
            "pct_days_with_signal": round((days.signals[m] > 0).mean() * 100, 1),
            "signals": len(ys), "trades_taken": len(yt),
            "win_rate_pct (trades opened this year)": round((yt.return_pct > 0).mean() * 100, 1) if len(yt) else None,
            "avg_days_held": round(yt.days_held.mean(), 1) if len(yt) else None,
            "account_start_usd": round(prev_eq, 2), "account_end_usd": round(e_end, 2),
            "pnl_usd": round(e_end - prev_eq, 2),
            "RETURN_PCT": round((e_end / prev_eq - 1) * 100, 2),
            "SPY_RETURN_PCT": round((spy_px[idx[-1]] / prev_spy - 1) * 100, 2) if spy_px is not None else None,
            "max_drop_in_year_pct": round(max_dd(np.concatenate([[prev_eq], equity[idx]])), 2),
            "positions_carried_to_next_year": int(n_open[idx[-1]]) if y < a.end_year else None,
            "value_carried_at_cost_usd": round(in_use[idx[-1]], 0) if y < a.end_year else None,
            "days_market_uptrend_pct": round((days.market[m] == "Uptrend").mean() * 100, 1) if "market" in days else None,
        }
        row["beat_SPY"] = ("Yes" if row["RETURN_PCT"] > row["SPY_RETURN_PCT"] else "No") if row["SPY_RETURN_PCT"] is not None else ""
        yrows.append(row)
        prev_eq = e_end
        if spy_px is not None:
            prev_spy = spy_px[idx[-1]]
    yearly = pd.DataFrame(yrows)

    traded = sig[sig.shares > 0] if len(sig) else sig
    closed = traded[traded.status.isin(["Stopped", "Breakeven stop", "Trailing exit", "Time stop"])] if len(traded) else traded

    def stats(df):
        if not len(df):
            return {"trades": 0}
        win = df.return_pct > 0
        return {"trades": len(df), "win_rate_pct": round(win.mean() * 100, 1),
                "avg_return_pct": round(df.return_pct.mean(), 2), "median_return_pct": round(df.return_pct.median(), 2),
                "avg_R": round(df.R.mean(), 2), "best_pct": round(df.return_pct.max(), 1),
                "worst_pct": round(df.return_pct.min(), 1),
                "reached_target1_pct": round(df.hit_target1.notna().mean() * 100, 1),
                "doubled_pct": round(df.hit_double.notna().mean() * 100, 1),
                "avg_days_held": round(df.days_held.mean(), 1), "median_days_held": round(df.days_held.median(), 1),
                "avg_days_winners": round(df.days_held[win].mean(), 1) if win.any() else "-",
                "avg_days_losers": round(df.days_held[~win].mean(), 1) if (~win).any() else "-"}

    n_days = len(days)
    d_sig = int((days.signals > 0).sum())
    d_setup = int((days.setup_4of4_stocks > 0).sum())
    yrs = max(n_days / 252, 1e-9)
    total_ret = (equity[-1] / a.account - 1) * 100
    spy_tot = (spy_px[-1] / spy_px[0] - 1) * 100 if spy_px is not None else None
    cagr = ((equity[-1] / a.account) ** (1 / yrs) - 1) * 100 if equity[-1] > 0 else -100
    spy_cagr = (((spy_px[-1] / spy_px[0]) ** (1 / yrs)) - 1) * 100 if spy_px is not None else None
    open_now = traded[traded.status == "Open"] if len(traded) else traded

    summary = [
        ("Period", f"{days.date.iloc[0]} to {days.date.iloc[-1]}  ({n_years} year{'s' if n_years > 1 else ''})"),
        ("Data", "SYNTHETIC DEMO DATA, not real prices" if a.demo else "Yahoo Finance daily, split-adjusted"),
        ("Stocks tested", sum(t in data for t in tickers)),
        ("Trading days", n_days),
        ("", ""),
        ("OVERALL RESULT", ""),
        ("   Account", f"${a.account:,.0f} -> ${equity[-1]:,.0f}"),
        ("   Total return", f"{total_ret:+.1f}%"),
        ("   Average per year (CAGR)", f"{cagr:+.1f}%"),
        ("   SPY buy-and-hold, same period", f"{spy_tot:+.1f}%  (CAGR {spy_cagr:+.1f}%)" if spy_tot is not None else "n/a"),
        ("   Years beating SPY", f"{(yearly.beat_SPY == 'Yes').sum()} of {len(yearly)}" if len(yearly) else "-"),
        ("   Best year", f"{int(yearly.loc[yearly.RETURN_PCT.idxmax(), 'year'])}: {yearly.RETURN_PCT.max():+.1f}%" if len(yearly) else "-"),
        ("   Worst year", f"{int(yearly.loc[yearly.RETURN_PCT.idxmin(), 'year'])}: {yearly.RETURN_PCT.min():+.1f}%" if len(yearly) else "-"),
        ("   Largest drop in account (peak to low, daily values)", f"{max_dd(equity):.1f}%"),
        ("   Positions still open now", f"{len(open_now)} (valued at last close)"),
        ("   Sizing", f"risk {a.risk:g}% per trade, max {a.maxpos:g}% per stock, "
                      f"{'compounding (sized from current account)' if a.compound else 'fixed (sized from starting account)'}, "
                      f"{'cash limit ON (skip/reduce when no cash; strongest RS first)' if cash_limit else 'no cash limit'}"),
        ("   Most positions open at once", int(n_open.max()) if len(n_open) else 0),
        ("   Most capital in use", f"${in_use.max():,.0f} ({in_use.max() / a.account * 100:.0f}% of start)" if len(in_use) else "-"),
        ("   Signals skipped or reduced for lack of cash", int(sig.taken.str.startswith(("Skipped: no", "Reduced")).sum()) if len(sig) else 0),
        ("   Average days a position is open", f"{traded.days_held.mean():.0f} calendar days (median {traded.days_held.median():.0f})" if len(traded) else "-"),
        ("   See the 'Yearly' sheet for each year. Open positions at 31 Dec are valued at that day's close and carried into the next year.", ""),
        ("", ""),
        ("HOW OFTEN SIGNALS HAPPEN", ""),
        ("   Days with at least 1 TRADE SIGNAL", f"{d_sig} of {n_days} ({d_sig / n_days * 100:.0f}%)"),
    ]
    if "market" in days:
        summary.append(("   ...of which the market was in an Uptrend", int(((days.signals > 0) & (days.market == "Uptrend")).sum())))
    summary += [
        ("   Days with at least 1 SETUP 4/4 stock (app tag)", f"{d_setup} of {n_days} ({d_setup / n_days * 100:.0f}%)"),
        ("   Total trade signals", len(sig)),
        ("   Stocks that gave at least one signal", sig.ticker.nunique() if len(sig) else 0),
        ("   Funnel: trend 4/4 + breakout on volume", funnel["trend_brk_vol"]),
        ("   ...and 0-10% above 50-day", funnel["ext_ok"]),
        ("   ...and no earnings within 7 days", funnel["earn_ok"]),
        ("", ""),
        ("TRADE STATISTICS (trades taken)", ""),
    ]
    for lab, df in [("All trades (open ones valued at last close)", traded),
                    ("Closed trades only", closed),
                    ("Trades in market Uptrend", traded[traded.market == "Uptrend"] if len(traded) else traded),
                    ("Trades in Caution/Downtrend", traded[traded.market.isin(["Caution", "Downtrend"])] if len(traded) else traded)]:
        summary.append((lab, ""))
        summary += [(f"   {k}", v) for k, v in stats(df).items()]
    summary += [("", ""), ("WARNINGS", "")]
    if len(late) and n_years > 1:
        summary.append(("   Survivorship bias", "This watchlist was picked in 2026 from stocks that have already done well. "
                        "Testing it on past years makes results look better than they would really have been. "
                        "Re-run with a broad list (e.g. the S&P 500 via --tickers) for a fairer test."))
        summary.append((f"   {len(late)} stocks started trading after the test began (only tested from their start)",
                        ", ".join(late)))
    summary.append(("   Rules", "Technical rules only; growth/size scores are not point-in-time, so not tested. "
                                 "No commissions or slippage. Not financial advice."))

    if len(sig):
        sig["month"] = pd.to_datetime(sig.date).dt.to_period("M").astype(str)
        monthly = traded.assign(month=pd.to_datetime(traded.date).dt.to_period("M").astype(str)).groupby("month").agg(
            trades=("ticker", "size"), win_rate_pct=("return_pct", lambda x: (x > 0).mean() * 100),
            avg_return_pct=("return_pct", "mean"), avg_R=("R", "mean"), pnl_usd=("pnl_usd", "sum")).round(2).reset_index()
        by_t = traded.groupby("ticker").agg(trades=("date", "size"), first=("date", "min"), last=("date", "max"),
                                            win_rate_pct=("return_pct", lambda x: (x > 0).mean() * 100),
                                            avg_return_pct=("return_pct", "mean"), best_pct=("return_pct", "max"),
                                            avg_R=("R", "mean"), pnl_usd=("pnl_usd", "sum")).round(2) \
            .sort_values("pnl_usd", ascending=False).reset_index()
        sig = sig.drop(columns=["month", "_risk_ps", "_path"])
    else:
        monthly = by_t = pd.DataFrame()

    out = Path(a.out or f"doubler_backtest_{label}{'_DEMO' if a.demo else ''}.xlsx")
    num = lambda df: df.round(2) if len(df) else df
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        pd.DataFrame(summary, columns=["Item", "Value"]).to_excel(xw, sheet_name="Summary", index=False)
        yearly.to_excel(xw, sheet_name="Yearly", index=False)
        num(sig).to_excel(xw, sheet_name="Trade signals", index=False)
        days.to_excel(xw, sheet_name="Daily", index=False)
        num(monthly).to_excel(xw, sheet_name="Monthly", index=False)
        num(by_t).to_excel(xw, sheet_name="By ticker", index=False)
        if len(setups) <= 60000:
            num(setups).to_excel(xw, sheet_name="Setup 4of4 days", index=False)
        if a.check:
            ct = a.check.upper()
            if ct in data:
                check_dump(ct, data, spy, earn, xw)
            else:
                print(f"--check: no data for {ct}")
        pd.DataFrame({"Rule": __doc__.strip().splitlines()}).to_excel(xw, sheet_name="Rules", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2" if ws.title != "Yearly" else "B2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                lim = 110 if ws.title in ("Rules", "Summary") else 60
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), lim)
    sig.to_csv(out.with_suffix(".csv"), index=False)
    if len(setups) > 60000:
        setups.to_csv(out.with_name(out.stem + "_setups.csv"), index=False)

    print("\n" + "\n".join(f"{k:<58} {v}" for k, v in summary if k))
    if len(yearly):
        print("\nYEARLY")
        print(yearly[["year", "days_with_signal", "trades_taken", "RETURN_PCT", "SPY_RETURN_PCT",
                      "max_drop_in_year_pct", "positions_carried_to_next_year"]].to_string(index=False))
    print(f"\nSaved {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
