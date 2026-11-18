#!/usr/bin/env python3
"""
Doubler nightly scan
====================
Runs on GitHub Actions after each US market close (see .github/workflows/nightly.yml).
Applies the app's READY rules to your watchlist (scanner/watchlist.txt) plus every
liquid US stock (NYSE + Nasdaq common stocks, price >= $5, >= $10M traded per day,
about 2,000-3,000 stocks; list refreshed weekly), and writes:

  data/scan.json          read by the app's Nightly tab
  data/ready_history.csv  one line per READY stock per day (your running record)

If the repository has the secrets TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, it also
sends the READY list to Telegram.

READY (same as the app and the backtest):
  1 uptrend 4/4: close above a rising 50-day and a rising 200-day SMA
  2 broke out of a 25-day base (<= 20% deep) in the last 3 sessions on >= 1.5x volume
  3 price still within the buy zone: <= 5% above the breakout level
  4 0-10% above the 50-day SMA
  5 no earnings within 7 days
  6 market (SPY/QQQ) not in a downtrend; Caution = half size

Run locally:   python scanner/nightly_scan.py                  (whole market)
               python scanner/nightly_scan.py --universe sp500 (watchlist + S&P 500)
Price history is cached in .cache/ so later runs only download recent days.
Test offline:  python scanner/nightly_scan.py --demo --dry-run
Not financial advice.
"""
import argparse
import datetime as dt
import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import doubler_backtest as B  # noqa: E402  (shared rule code, same as the backtest)

ROOT = HERE.parent
UA = {"User-Agent": "Mozilla/5.0 (Doubler nightly scan)"}


def read_watchlist(path):
    if not path.exists():
        return list(B.WATCHLIST)
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0]
        out += [x.strip().upper() for x in line.replace(",", " ").split() if x.strip()]
    return out


def sp500():
    try:
        req = urllib.request.Request("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA)
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
        tab = pd.read_html(io.StringIO(html))[0]
        syms = [str(s).strip().upper() for s in tab["Symbol"]]
        print(f"S&P 500 list: {len(syms)} symbols")
        return syms
    except Exception as e:
        print(f"Could not load the S&P 500 list ({e}); scanning the watchlist only")
        return []


# ---------------------------------------------------------------- whole-market universe
import re  # noqa: E402

CACHE = ROOT / ".cache"
SYMBOL_FILES = [("nasdaq", "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"),
                ("other", "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")]
BAD_NAME = re.compile(r"warrant|\bunits?\b|\brights?\b|preferred|depositary shares? represent|\bnotes? due|"
                      r"debenture|subordinated|% |when[- ]issued|acquisition corp", re.I)


def parse_symbol_file(text, kind):
    """Nasdaq Trader symbol directory -> common stocks only (no ETFs, test issues,
    warrants, units, rights, preferreds, notes, SPACs, deficient Nasdaq listings)."""
    lines = [x for x in text.splitlines() if x.strip() and not x.startswith("File Creation Time")]
    if not lines:
        return []
    hdr = lines[0].split("|")
    out = []
    for line in lines[1:]:
        r = dict(zip(hdr, line.split("|")))
        sym = (r.get("Symbol") or r.get("ACT Symbol") or "").strip()
        if not sym or r.get("ETF") == "Y" or r.get("Test Issue") == "Y":
            continue
        if kind == "nasdaq" and r.get("Financial Status", "N") not in ("N", ""):
            continue
        if any(ch in sym for ch in "$^ =+#"):
            continue
        if BAD_NAME.search(r.get("Security Name", "")):
            continue
        out.append(sym.replace("/", "."))
    return out


def us_listed():
    syms = []
    for kind, url in SYMBOL_FILES:
        try:
            txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "replace")
            got = parse_symbol_file(txt, kind)
            print(f"{url.rsplit('/', 1)[-1]}: {len(got)} common stocks")
            syms += got
        except Exception as e:
            print(f"Could not load {url} ({e})")
    return sorted(set(syms))


def yf_chunks(tickers, start, end, chunk=150, pause=2.0, label="prices"):
    """Download daily bars in batches, with retries, so ~3,000 symbols don't trip Yahoo's limits."""
    import yfinance as yf
    out, n = {}, len(tickers)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for k in range(0, n, chunk):
        part = tickers[k:k + chunk]
        ys = [B.yahoo_symbol(t) for t in part]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(ys, start=start, end=end, auto_adjust=True, group_by="ticker",
                                  progress=False, threads=True)
                break
            except Exception as e:
                wait = 30 * (attempt + 1)
                print(f"  {label} batch {k // chunk + 1}: {e}; retrying in {wait} s")
                time.sleep(wait)
        if raw is None or raw.empty:
            continue
        multi = isinstance(raw.columns, pd.MultiIndex)
        for t, y in zip(part, ys):
            try:
                df = (raw[y] if multi else raw)[cols].dropna()
            except KeyError:
                continue
            if len(df):
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                out[t] = df
        print(f"  {label}: {min(k + chunk, n)}/{n} requested, {len(out)} received")
        if k + chunk < n:
            time.sleep(pause)
    return out


def build_universe(min_price, min_dvol_m, path, today, max_age_days=7):
    """All liquid US common stocks, refreshed weekly and saved to data/universe.json."""
    if path.exists():
        try:
            u = json.loads(path.read_text())
            age = (today - dt.date.fromisoformat(u["date"])).days
            if age < max_age_days and u.get("minPrice") == min_price and u.get("minDollarVolM") == min_dvol_m and u["symbols"]:
                print(f"Universe: {u['count']} stocks (saved {u['date']}, refreshed weekly)")
                return u["symbols"]
        except Exception:
            pass
    syms = us_listed()
    if not syms:
        if path.exists():
            print("Symbol lists unavailable; using the last saved universe")
            return json.loads(path.read_text())["symbols"]
        return []
    print(f"Checking price and trading value for {len(syms)} listed stocks (weekly)...")
    px = yf_chunks(syms, today - dt.timedelta(days=45), today + dt.timedelta(days=1), chunk=200, label="universe")
    keep = []
    for t, df in px.items():
        if len(df) >= 15 and df.Close.iloc[-1] >= min_price and (df.Close * df.Volume).tail(20).mean() >= min_dvol_m * 1e6:
            keep.append(t)
    keep.sort()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": str(today), "minPrice": min_price, "minDollarVolM": min_dvol_m,
                                "listed": len(syms), "count": len(keep), "symbols": keep}, separators=(",", ":")))
    print(f"Universe: {len(keep)} of {len(syms)} listed stocks pass price >= ${min_price:g} "
          f"and >= ${min_dvol_m:g}M traded per day")
    return keep


# ---------------------------------------------------------------- price cache between runs
def _cfile(t):
    return CACHE / "prices" / (t.replace(".", "_").replace("/", "_") + ".pkl")


def cached_prices(tickers, start, today, fetch=None, full_every_days=30):
    """Keeps ~2 years of daily bars per stock in .cache (saved by GitHub between runs),
    so each night only the last couple of weeks are downloaded. A stock is fully
    re-downloaded if a split/dividend changed its adjusted history, if its cache is
    stale, or every 30 days."""
    fetch = fetch or yf_chunks
    (CACHE / "prices").mkdir(parents=True, exist_ok=True)
    meta_p = CACHE / "prices_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    end = today + dt.timedelta(days=1)
    have, full, inc = {}, [], []
    for t in tickers:
        f = _cfile(t)
        last_full = meta.get(t)
        if f.exists() and last_full and (today - dt.date.fromisoformat(last_full)).days < full_every_days:
            try:
                df = pd.read_pickle(f)
                if len(df) and (today - df.index[-1].date()).days <= 12:
                    have[t] = df
                    inc.append(t)
                    continue
            except Exception:
                pass
        full.append(t)
    if inc:
        print(f"Updating {len(inc)} cached stocks (last 15 days)...")
        new = fetch(inc, today - dt.timedelta(days=15), end, label="update")
        for t in inc:
            nw = new.get(t)
            if nw is None or not len(nw):
                continue                                    # keep cached bars (maybe a holiday or delisting)
            old = have[t]
            ov = old.index.intersection(nw.index)
            if len(ov) and abs(nw.Close.loc[ov[0]] / old.Close.loc[ov[0]] - 1) > 0.005:
                full.append(t)                              # adjusted history changed (split/dividend)
                continue
            merged = pd.concat([old[old.index < nw.index[0]], nw])
            have[t] = merged[merged.index >= pd.Timestamp(start)]
            have[t].to_pickle(_cfile(t))
    if full:
        print(f"Downloading full history for {len(full)} stocks...")
        got = fetch(full, start, end, label="history")
        for t, df in got.items():
            have[t] = df
            df.to_pickle(_cfile(t))
            meta[t] = str(today)
    meta_p.write_text(json.dumps(meta))
    return {t: have[t] for t in tickers if t in have}


def cached_earnings(tickers, today, max_age_days=5):
    p = CACHE / "earnings.json"
    store = json.loads(p.read_text()) if p.exists() else {}
    stale = [t for t in tickers if t not in store or (today - dt.date.fromisoformat(store[t]["fetched"])).days >= max_age_days]
    if stale:
        got = B.earnings_dates(stale, limit=6)
        for t in stale:
            store[t] = {"fetched": str(today), "dates": [str(x.date()) for x in got.get(t, [])]}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(store))
    out = {}
    for t in tickers:
        ds = store.get(t, {}).get("dates")
        out[t] = [pd.Timestamp(x) for x in ds] if ds else None
    return out


def sr_zones(h, l, c):
    """Port of the app's srZones(): swing highs/lows grouped into zones."""
    n = len(c)
    if n < 40:
        return []
    atr = np.mean([max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(n - 14, n)])
    W, piv = 5, []
    for i in range(W, n):
        R = min(W, n - 1 - i)
        if R < 2:
            break
        is_h = is_l = True
        for j in range(i - W, i + R + 1):
            if j == i:
                continue
            if h[j] > h[i]:
                is_h = False
            if l[j] < l[i]:
                is_l = False
        if is_h:
            piv.append((h[i], i))
        if is_l:
            piv.append((l[i], i))
    piv.sort()
    last, tol, cl = c[-1], max(c[-1] * 0.012, atr * 0.5), []
    for p, i in piv:
        if cl and p - cl[-1]["lo"] <= tol:
            cl[-1]["m"].append((p, i))
        else:
            cl.append({"lo": p, "m": [(p, i)]})
    zones = []
    for k in cl:
        ps = [x[0] for x in k["m"]]
        lo, hi = min(ps), max(ps)
        if hi - lo < atr * 0.25:
            mid = (lo + hi) / 2
            lo, hi = mid - atr * 0.125, mid + atr * 0.125
        age = n - 1 - max(x[1] for x in k["m"])
        if len(k["m"]) >= 2 or age < 40:
            zones.append({"lo": lo, "hi": hi, "touches": len(k["m"]), "age": age})
    return zones


def near_sr(zones, ref):
    sup = res = None
    for z in zones:
        mid = (z["lo"] + z["hi"]) / 2
        if mid < ref:
            if sup is None or z["hi"] > sup["hi"]:
                sup = z
        elif res is None or z["lo"] < res["lo"]:
            res = z
    pack = lambda z, p: None if z is None else {"lo": round(z["lo"], 2), "hi": round(z["hi"], 2), "touches": z["touches"], "pct": round(p, 2)}
    return (pack(sup, (min(sup["hi"], ref) / ref - 1) * 100 if sup else 0),
            pack(res, (max(res["lo"], ref) / ref - 1) * 100 if res else 0))


def num(x, nd=2):
    try:
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else round(x, nd)
    except (TypeError, ValueError):
        return None


def analyse(t, df, spy):
    """Latest-day values for one stock, using the exact rule code of the backtest."""
    d = B.add_indicators(df)
    if spy is not None:
        d = B.add_rs(d, spy)
    if len(d) < 60 or pd.isna(d.s50.iloc[-1]):
        return None
    n = len(d)
    c, h, l = d.Close.values, d.High.values, d.Low.values
    last = c[-1]
    row = d.iloc[-1]
    a50, a200 = bool(last > row.s50), bool(not pd.isna(row.s200) and last > row.s200)
    r50 = bool(n > 60 and row.s50 > d.s50.iloc[-11])
    r200 = bool(not pd.isna(row.s200) and n > 220 and row.s200 > d.s200.iloc[-21])
    brk = None
    for k in range(3):                                      # breakout today or in the last 2 sessions
        i = n - 1 - k
        if bool(d.brk.iloc[i]):
            brk = {"daysAgo": k, "pivot": num(d.pivot_px.iloc[i]), "depth": num(d.depth.iloc[i], 1),
                   "volx": num(d.volx.iloc[i]), "holding": bool(last > d.pivot_px.iloc[i])}
            break
    base = None
    if brk is None and n >= B.BASE_DAYS:
        hi, lo = h[-B.BASE_DAYS:].max(), l[-B.BASE_DAYS:].min()
        depth = (hi - lo) / hi * 100
        base = {"hi": num(hi), "lo": num(lo), "depth": num(depth, 1), "tight": bool(depth <= B.BASE_MAX),
                "toPivot": num((last / hi - 1) * 100)}
    sup, res = near_sr(sr_zones(h[-260:], l[-260:], c[-260:]), last)
    v = d.Volume.values
    vol_today = v[-1] / v[-51:-1].mean() if n > 51 else None
    return {
        "t": t, "price": num(last), "chg": num((last / c[-2] - 1) * 100) if n > 1 else None,
        "asOf": str(d.index[-1].date()),
        "s50": num(row.s50), "s200": num(row.s200), "s50chg": num((row.s50 / d.s50.iloc[-11] - 1) * 100),
        "a50": a50, "r50": r50, "a200": a200, "r200": r200, "trendPts": int(a50) + int(r50) + int(a200) + int(r200),
        "ext": num(row.ext), "brk": brk, "base": base, "volToday": num(vol_today),
        "low10": num(d.low10.iloc[-1]), "sup": sup, "res": res,
        "rs": num(row.rs if "rs" in d else None, 1),
        "rsNewHigh": bool(row.rs_newhigh) if "rs_newhigh" in d and not pd.isna(row.rs_newhigh) else False,
    }


def judge(s, earn_dates, market_state, today):
    """App checks: Setup x/4 and READY 6/6."""
    b = s["brk"]
    good_brk = bool(b and b["holding"] and (b["volx"] or 0) >= B.VOL_X)
    ext_ok = s["ext"] is not None and 0 <= s["ext"] <= B.EXT_MAX
    if earn_dates is None:
        s["earn"], e_ok = None, None
    else:
        fut = [e for e in earn_dates if e.date() >= today]
        if fut:
            days = (fut[0].date() - today).days
            s["earn"] = {"date": str(fut[0].date()), "days": days}
            e_ok = days > B.EARN_DAYS
        else:
            s["earn"], e_ok = {"none": True}, True
    near_pivot = bool(not b and s["base"] and s["base"]["tight"] and s["base"]["toPivot"] >= -5)
    setup = [s["trendPts"] == 4, good_brk or near_pivot, ext_ok, bool(e_ok)]
    s["setup"] = sum(setup)
    in_zone = bool(good_brk and s["price"] <= b["pivot"] * 1.05)
    checks = [s["trendPts"] == 4, good_brk, in_zone, ext_ok, e_ok,
              None if market_state is None else market_state != "Downtrend"]
    s["checks"] = checks
    s["passed"] = sum(1 for x in checks if x is True)
    s["ready"] = all(x is not False for x in checks)
    s["half"] = bool(s["ready"] and market_state == "Caution")
    if s["ready"]:                                    # Plan-tab style stop and zone, for the message
        E = s["price"]
        pct_stop = E * (1 - B.STOP_PCT / 100)
        low10 = (s["low10"] or 0) * 0.995
        s["stop"] = num(low10 if (pct_stop < low10 < E * 0.98) else pct_stop)
        s["buyZoneTop"] = num(b["pivot"] * 1.05)
    return s


def market_info(data):
    ms = B.market_series(data)
    out = {"state": None, "sellingDays": None, "indexes": {}}
    if ms is not None and len(ms):
        out["state"] = str(ms.market.iloc[-1])
        out["sellingDays"] = int(ms.selling_days.iloc[-1]) if not pd.isna(ms.selling_days.iloc[-1]) else None
    for t in B.INDEXES:
        if t in data:
            d = B.add_indicators(data[t])
            r = d.iloc[-1]
            out["indexes"][t] = {"price": num(r.Close), "ext": num(r.ext), "a50": bool(r.Close > r.s50),
                                 "r50": bool(r.s50 > d.s50.iloc[-11]), "a200": bool(r.Close > r.s200),
                                 "asOf": str(d.index[-1].date())}
    return out


def telegram(msg, dry):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if dry or not tok or not chat:
        print("\n--- Telegram message" + (" (dry run)" if dry else " (not sent: no TELEGRAM_* secrets)") + " ---\n" + msg)
        return
    body = urllib.parse.urlencode({"chat_id": chat, "text": msg, "disable_web_page_preview": "true"}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=body), timeout=30)
        print("Telegram message sent")
    except Exception as e:
        print(f"Telegram send failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="Doubler nightly scan")
    ap.add_argument("--universe", choices=["all", "sp500", "watchlist"], default="all",
                    help="all = every liquid US stock (default); sp500 = watchlist + S&P 500; watchlist = watchlist only")
    ap.add_argument("--sp500", action="store_true", help="same as --universe sp500 (kept for older workflows)")
    ap.add_argument("--min-price", type=float, default=5.0, help="universe filter: last price at least this (default 5)")
    ap.add_argument("--min-dollar-volume", type=float, default=10.0,
                    help="universe filter: average traded value per day, in $ millions (default 10)")
    ap.add_argument("--watchlist", default=str(HERE / "watchlist.txt"))
    ap.add_argument("--out", default=str(ROOT / "data" / "scan.json"))
    ap.add_argument("--dry-run", action="store_true", help="print the Telegram message instead of sending it")
    ap.add_argument("--demo", action="store_true", help="synthetic prices, for testing without internet")
    a = ap.parse_args()
    if a.sp500:
        a.universe = "sp500"
    t0 = time.time()
    today = dt.date.today()

    watch = [t for t in dict.fromkeys(read_watchlist(Path(a.watchlist))) if t not in B.LEVERAGED]
    start = today - dt.timedelta(days=560)
    if a.universe == "all" and not a.demo:
        market = build_universe(a.min_price, a.min_dollar_volume, ROOT / "data" / "universe.json", today)
        uname = f"all US stocks ≥ ${a.min_price:g}, ≥ ${a.min_dollar_volume:g}M/day"
        if not market:
            print("Universe unavailable; falling back to watchlist + S&P 500")
            market, uname = sp500(), "watchlist + S&P 500"
    elif a.universe == "sp500" and not a.demo:
        market, uname = sp500(), "watchlist + S&P 500"
    else:
        market, uname = [], "watchlist"
    universe = list(dict.fromkeys(watch + [t for t in market if t not in B.LEVERAGED]))
    print(f"Scanning {len(universe)} stocks ({uname})")

    if a.demo:
        data, earn_all = B.demo_data(universe + B.INDEXES, start, today)
    else:
        data = cached_prices(universe + B.INDEXES, start, today)
        print(f"Price history ready for {len(data)} of {len(universe) + len(B.INDEXES)} symbols")
        earn_all = None
    if not data or "SPY" not in data:
        print("No price data (Yahoo may be rate-limiting). Keeping the previous scan.json.")
        sys.exit(1)

    mkt = market_info(data)
    spy = data["SPY"].Close
    asof = data["SPY"].index[-1].date()
    rows = []
    for t in universe:
        if t in data:
            try:
                s = analyse(t, data[t], spy)
            except Exception as e:
                print(f"  {t}: {e}")
                s = None
            if s:
                rows.append(s)

    # earnings dates only for stocks that could qualify (keeps Yahoo calls low)
    cand = [s["t"] for s in rows if s["trendPts"] >= 3 and (s["brk"] or (s["base"] and s["base"]["tight"]))]
    if a.demo:
        earn = {t: earn_all.get(t) for t in cand}
    else:
        print(f"{len(cand)} candidates need earnings dates")
        earn = cached_earnings(cand, today) if cand else {}
    for s in rows:
        judge(s, earn.get(s["t"]) if s["t"] in cand else None, mkt["state"], asof)
        s["watch"] = s["t"] in watch

    # keep the JSON small: all watchlist stocks, plus market stocks that are close to a setup
    keep = [s for s in rows if s["watch"] or s["passed"] >= 4 or s["setup"] >= 3]
    keep.sort(key=lambda s: (not s["ready"], -s["passed"], -s["setup"], -(s["rs"] or -999)))
    ready = [s for s in keep if s["ready"]]
    near = [s for s in keep if not s["ready"] and s["passed"] >= 4]
    out = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "asOf": str(asof), "demo": bool(a.demo),
        "universe": len(universe), "universeName": uname, "scanned": len(rows), "watchlist": len(watch),
        "sp500": a.universe != "watchlist",
        "market": mkt,
        "counts": {"ready": len(ready), "near": len(near), "setup4": sum(1 for s in rows if s["setup"] == 4)},
        "stocks": keep,
    }
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, separators=(",", ":")))
    print(f"\nScanned {len(rows)} stocks in {time.time() - t0:.0f} s. Market: {mkt['state']}. "
          f"READY: {len(ready)}  near: {len(near)}. Wrote {outp} ({outp.stat().st_size // 1024} KB)")

    hist = outp.parent / "ready_history.csv"
    new = pd.DataFrame([{"date": str(asof), "ticker": s["t"], "price": s["price"], "pivot": s["brk"]["pivot"],
                         "volume_x": s["brk"]["volx"], "pct_vs_50d": s["ext"], "rs": s["rs"], "stop": s.get("stop"),
                         "market": mkt["state"], "half_size": s["half"]} for s in ready])
    if len(new):
        old = pd.read_csv(hist) if hist.exists() else pd.DataFrame()
        allh = pd.concat([old, new]).drop_duplicates(["date", "ticker"], keep="last")
        allh.to_csv(hist, index=False)

    # Telegram summary
    st = mkt["state"] or "unknown"
    sell = mkt["sellingDays"]
    lines = [f"Doubler nightly · US close {asof}", f"Scanned {len(rows)} stocks ({uname})",
             f"Market: {st}" + (f" ({sell} selling days)" if sell is not None else "")]
    if st == "Downtrend":
        lines.append("Most breakouts fail now: no READY signals in a downtrend.")
    if ready:
        lines.append(f"\nREADY ({len(ready)})" + (" · half size (market caution)" if st == "Caution" else ""))
        for s in ready[:12]:
            b = s["brk"]
            lines.append(f"• {s['t']} ${s['price']:.2f} · breakout ${b['pivot']:.2f} on {b['volx']:.1f}× vol · "
                         f"buy ≤ ${s['buyZoneTop']:.2f} · stop ~${s['stop']:.2f} ({(s['stop'] / s['price'] - 1) * 100:.1f}%)"
                         + (f" · RS {s['rs']:+.0f}%" if s['rs'] is not None else ""))
    else:
        lines.append("\nNo stocks READY today.")
    if near:
        lines.append(f"\nClose (4–5 of 6 checks): " + ", ".join(s["t"] for s in near[:10]))
    url = os.environ.get("APP_URL", "").strip()
    if url:
        lines.append(f"\n{url}")
    lines.append("Not financial advice.")
    telegram("\n".join(lines), a.dry_run or a.demo)


if __name__ == "__main__":
    main()
