# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "pandas"]
# ///
"""Premier League predictor: Dixon-Coles model, trends, tactics, season sim and value bets.

uv run epl.py fetch      download results, xG, fixtures and odds (run before the others)
uv run epl.py predict    next round: probabilities, likely scores, team styles
uv run epl.py value      bets whose best price beats the blended model/market probability
uv run epl.py props      player props (anytime scorer, assists, shots) vs AU books, logged as paper bets
uv run epl.py ucl        Champions League: cross-league model predictions and value bets
uv run epl.py ucl_backtest  walk-forward test of the cross-league model on European games since 2023/24 (slow)
uv run epl.py season     10,000 simulations of the rest of the season
uv run epl.py tactics    current-season team profiles (pressing, xG luck, box-score stats)
uv run epl.py trends     how the league has changed since 1993/94
uv run epl.py backtest   walk-forward test since 2019/20; retunes data/params.json (~20 s)
uv run epl.py report     everything above as report.html

Set ODDS_API_KEY (free at the-odds-api.com) to price bets with Australian bookmakers; otherwise
odds come from football-data.co.uk.
"""
import gzip, http.client, json, os, sys, urllib.error, urllib.request, warnings
from functools import cache
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

DATA = Path(__file__).parent / "data"
LOCAL_TZ = datetime.now().astimezone().tzinfo
NOW_UTC = pd.Timestamp.now(tz="UTC").tz_localize(None)  # Understat kickoffs are naive UTC
TODAY = pd.Timestamp.now().normalize()


def season_of(ts):
    """Season start year: Aug 2026 - Jul 2027 is 2026 (2019/20 finished in late July 2020)."""
    return ts.year if ts.month >= 8 else ts.year - 1


CURRENT = season_of(TODAY)
FIRST = 1993
FD = "https://football-data.co.uk/mmz4281/{code}/{div}.csv"
FIXTURES = "https://football-data.co.uk/fixtures.csv"
UNDERSTAT = "https://understat.com/getLeagueData/EPL/{y}"
# Player data for UCL props: Understat's other leagues, keyed to the ESPN league of the same clubs
UNDERSTAT_LEAGUES = {"EPL": "eng.1", "La_liga": "esp.1", "Bundesliga": "ger.1", "Serie_A": "ita.1", "Ligue_1": "fra.1"}
ODDS_API = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds?apiKey={key}&regions=au&markets=h2h,totals&oddsFormat=decimal"

# Every spelling seen in Understat / The Odds API -> the football-data name.
ALIASES = {
    "Manchester City": "Man City", "Manchester United": "Man United", "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves", "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham", "Brighton and Hove Albion": "Brighton", "Brighton & Hove Albion": "Brighton",
    "West Bromwich Albion": "West Brom", "Leicester City": "Leicester", "Leeds United": "Leeds",
    "AFC Bournemouth": "Bournemouth", "Ipswich Town": "Ipswich", "Luton Town": "Luton", "Norwich City": "Norwich",
    "Stoke City": "Stoke", "Swansea City": "Swansea", "Cardiff City": "Cardiff", "Hull City": "Hull",
    "Huddersfield Town": "Huddersfield", "Coventry City": "Coventry", "Queens Park Rangers": "QPR",
    "Sheffield Wednesday": "Sheffield Weds",
}


def canon(name):
    return ALIASES.get(name, name)


def code(y):
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def fetch_url(url, headers=None, tries=4):
    """(body, response headers), retrying dropped connections and 5xx with backoff (2, 4, 8 s).
    A 4xx (bad key, out of credits) is raised at once: retrying can't fix it."""
    import time
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                return (gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body), r.headers
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == tries - 1:
                raise
        except (OSError, http.client.HTTPException):  # reset, timeout, truncated body
            if i == tries - 1:
                raise
        time.sleep(2 ** (i + 1))


def get(url, headers=None):
    return fetch_url(url, headers)[0]


# ---------------------------------------------------------------- data

def fetch():
    DATA.mkdir(exist_ok=True)
    jobs = [(DATA / f"E0_{y}.csv", FD.format(code=code(y), div="E0"), None) for y in range(FIRST, CURRENT + 1)]
    jobs += [(DATA / f"E1_{y}.csv", FD.format(code=code(y), div="E1"), None) for y in range(2014, CURRENT + 1)]
    jobs += [(DATA / f"understat_{y}.json", UNDERSTAT.format(y=y), {"X-Requested-With": "XMLHttpRequest"})
             for y in range(2014, CURRENT + 1)]
    jobs += [(understat_path(lg, y), UNDERSTAT.replace("EPL", lg).format(y=y), {"X-Requested-With": "XMLHttpRequest"})
             for lg in UNDERSTAT_LEAGUES if lg != "EPL" for y in (CURRENT - 1, CURRENT)]
    jobs += [(DATA / "fixtures.csv", FIXTURES, None)]
    for path, url, headers in jobs:
        if path.exists() and not any(str(y) in path.name for y in (CURRENT, CURRENT - 1)) and path.name != "fixtures.csv":
            continue  # older seasons never change (last season is refetched in case it was cached mid-season)
        try:
            path.write_bytes(get(url, headers))
            print("fetched", path.name)
        except (OSError, http.client.HTTPException) as e:
            if not path.exists():
                raise
            print(f"could not refresh {path.name} ({e!r}); kept the cached copy")
    if key := os.environ.get("ODDS_API_KEY"):
        api = DATA / "odds_api.json"
        try:
            body, hdr = fetch_url(ODDS_API.format(key=key))
            api.write_bytes(body)
            print("fetched odds_api.json, credits left:", hdr.get("x-requests-remaining"))
        except urllib.error.HTTPError as e:
            api.unlink(missing_ok=True)  # never price bets off an old snapshot
            hint = " - check ODDS_API_KEY is your real key" if e.code == 401 else ""
            print(f"The Odds API refused the request (HTTP {e.code}{hint}); using football-data.co.uk odds instead")
        except (OSError, http.client.HTTPException) as e:
            api.unlink(missing_ok=True)
            print(f"The Odds API refused the request (unreachable: {e!r}); using football-data.co.uk odds instead")
    fetch_espn()  # Champions League model: domestic leagues + European cups
    fetch_ucl_odds()


def load_fd(div, years):
    frames = []
    for y in years:
        p = DATA / f"{div}_{y}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip", low_memory=False)
        frames.append(df.dropna(subset=["HomeTeam", "FTHG"]).assign(season=y, div=div).copy())
    df = pd.concat(frames, ignore_index=True).copy()
    df["date"] = pd.to_datetime(df.Date, dayfirst=True, format="mixed")
    df["hg"], df["ag"] = df.FTHG.astype(int), df.FTAG.astype(int)
    return df.rename(columns={"HomeTeam": "home", "AwayTeam": "away"})


def understat_path(league, y):
    return DATA / (f"understat_{y}.json" if league == "EPL" else f"understat_{league}_{y}.json")


def load_understat(y, league="EPL"):
    return json.loads(understat_path(league, y).read_text())


def understat_team_xg(league="EPL", y=None):
    """Team -> xG per game this season, from Understat's match history."""
    y = CURRENT if y is None else y
    if not understat_path(league, y).exists():
        return {}
    return {canon(t["title"]): float(np.mean([h["xG"] for h in t["history"]]))
            for t in load_understat(y, league)["teams"].values() if t["history"]}


def understat_matches(years):
    """One row per fixture (played or not) with goals and xG."""
    rows = []
    for y in years:
        p = DATA / f"understat_{y}.json"
        if not p.exists():
            continue
        for m in load_understat(y)["dates"]:
            rows.append(dict(season=y, home=canon(m["h"]["title"]), away=canon(m["a"]["title"]),
                             played=m["isResult"], kickoff=pd.Timestamp(m["datetime"]),
                             hg=m["goals"]["h"], ag=m["goals"]["a"], xg_h=m["xG"]["h"], xg_a=m["xG"]["a"]))
    df = pd.DataFrame(rows)
    for c in ("hg", "ag", "xg_h", "xg_a"):
        df[c] = pd.to_numeric(df[c])
    return df


@cache
def e0_teams(y):
    teams = set()
    if (p := DATA / f"E0_{y}.csv").exists():
        teams |= set(pd.read_csv(p, usecols=["HomeTeam"], encoding="utf-8-sig", encoding_errors="replace").HomeTeam.dropna())
    if (DATA / f"understat_{y}.json").exists():
        teams |= {canon(m["h"]["title"]) for m in load_understat(y)["dates"]}
    return frozenset(teams)


def matches(first=2014):
    """E0 + E1 results since `first`, with Understat xG joined onto EPL games (by season + pairing)."""
    df = pd.concat([load_fd("E0", range(first, CURRENT + 1)), load_fd("E1", range(first, CURRENT + 1))])
    us = understat_matches(range(max(first, 2014), CURRENT + 1)).query("played")
    unknown = set(us.home) - set(df.home)
    assert not unknown, f"add these Understat names to ALIASES: {unknown}"
    df = df.merge(us[["season", "home", "away", "xg_h", "xg_a"]], on=["season", "home", "away"], how="left")
    # Understat posts results days before football-data does; add the ones football-data lacks.
    new = us.merge(df[["season", "home", "away"]], how="left", indicator=True).query("_merge == 'left_only'")
    new = new.assign(date=new.kickoff.dt.normalize(), div="E0", hg=new.hg.astype(int), ag=new.ag.astype(int))
    df = pd.concat([df, new.drop(columns=["_merge", "kickoff", "played"])]).sort_values("date", ignore_index=True)
    # Championship games in a season that ended in promotion (see `_nll`)
    up = {s: e0_teams(s + 1) for s in df.season.unique()}
    for side in ("home", "away"):
        df[f"promo_{side}"] = [d == "E1" and t in up[s] for d, t, s in zip(df["div"], df[side], df.season)]
    return df


# ---------------------------------------------------------------- model

PARAMS = {"xi": 0.0045, "alpha": 0.3, "w": 0.1}  # backtested 2019/20-2025/26; `backtest` rewrites data/params.json
PRIOR_SD = 2.0  # weak ridge, just pins the scale; a tight one drags Championship sides toward the EPL mean
WINDOW = pd.Timedelta(days=4 * 365)
E1_WEIGHT = 0.5  # Championship games count half as much as EPL games (backtested 0.3-1.0)


def params():
    p = DATA / "params.json"
    return {**PARAMS, **(json.loads(p.read_text()) if p.exists() else {})}


def _nll(x, hi, ai, yh, ya, gh, ga, ph, pa, w, n, hf=None, lg=None, L=0, team_sd=None):
    """Weighted Dixon-Coles negative log-likelihood and gradient.
    x = [c, home, rho, promo, att[n], dfn[n], latt[L], ldfn[L]]; a team's rating is its own plus its league's (L > 0),
    log(lam) = c + home*hf + att[h] + dfn[a]; log(mu) = c + att[a] + dfn[h]. hf = 0 marks a neutral venue.
    yh/ya are the Poisson targets (goals, or goals blended with xG); gh/ga are actual goals for the rho term.
    ph/pa flag a side in the Championship season it won promotion from: it out-performs its rating by `promo`
    (partly the luck that sent it up). Without this, promoted sides are badly over-rated in the EPL."""
    c, h, rho, promo = x[0], x[1], x[2], x[3]
    att, dfn = x[4:4 + n], x[4 + n:4 + 2 * n]
    A, D = att, dfn
    if L:
        latt, ldfn = x[4 + 2 * n:4 + 2 * n + L], x[4 + 2 * n + L:]
        A, D = att + latt[lg], dfn + ldfn[lg]
    hv = h if hf is None else h * hf
    eh = c + hv + A[hi] + D[ai] + promo * (ph - pa)
    ea = c + A[ai] + D[hi] + promo * (pa - ph)
    lam, mu = np.exp(eh), np.exp(ea)
    # tau and d(log tau) w.r.t. log(lam), log(mu), rho
    tau, gl, gm, gr = np.ones_like(lam), np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)
    s = (gh == 0) & (ga == 0)
    tau[s] = 1 - lam[s] * mu[s] * rho
    gl[s] = gm[s] = -lam[s] * mu[s] * rho / tau[s]
    gr[s] = -lam[s] * mu[s] / tau[s]
    s = (gh == 0) & (ga == 1)
    tau[s] = 1 + lam[s] * rho
    gl[s], gr[s] = lam[s] * rho / tau[s], lam[s] / tau[s]
    s = (gh == 1) & (ga == 0)
    tau[s] = 1 + mu[s] * rho
    gm[s], gr[s] = mu[s] * rho / tau[s], mu[s] / tau[s]
    s = (gh == 1) & (ga == 1)
    tau[s] = 1 - rho
    gr[s] = -1 / tau[s]
    ll = w * (yh * eh - lam + ya * ea - mu + np.log(np.maximum(tau, 1e-10)))
    k = 1 / (team_sd or PRIOR_SD)**2
    kl = 1 / PRIOR_SD**2
    nll = -ll.sum() + k * (att @ att + dfn @ dfn) / 2
    dh, da = w * (yh - lam + gl), w * (ya - mu + gm)
    g = np.empty_like(x)
    g[0], g[2] = -(dh.sum() + da.sum()), -(w * gr).sum()
    g[1] = -dh.sum() if hf is None else -(dh * hf).sum()
    g[3] = -((dh - da) * (ph - pa)).sum()
    ga_ = -(np.bincount(hi, dh, n) + np.bincount(ai, da, n))
    gd_ = -(np.bincount(ai, dh, n) + np.bincount(hi, da, n))
    g[4:4 + n] = ga_ + k * att
    g[4 + n:4 + 2 * n] = gd_ + k * dfn
    if L:
        nll += kl * (latt @ latt + ldfn @ ldfn) / 2
        g[4 + 2 * n:4 + 2 * n + L] = np.bincount(lg, ga_, L) + kl * latt
        g[4 + 2 * n + L:] = np.bincount(lg, gd_, L) + kl * ldfn
    return nll, g


def fit(df, asof, xi=None, alpha=None, prev=None):
    """Fit Dixon-Coles on matches in the 4 years before `asof`. `prev` (an earlier fit) warm-starts it."""
    from scipy.optimize import minimize
    p = params()
    xi = p["xi"] if xi is None else xi
    alpha = p["alpha"] if alpha is None else alpha
    d = df[(df.date < asof) & (df.date >= asof - WINDOW)]
    teams = sorted(set(d.home) | set(d.away))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi, ai = d.home.map(idx).to_numpy(), d.away.map(idx).to_numpy()
    gh, ga = d.hg.to_numpy(float), d.ag.to_numpy(float)
    xh = d.xg_h.to_numpy(float) if "xg_h" in d else np.full(len(d), np.nan)
    xa = d.xg_a.to_numpy(float) if "xg_a" in d else np.full(len(d), np.nan)
    yh = np.where(np.isnan(xh), gh, alpha * gh + (1 - alpha) * xh)
    ya = np.where(np.isnan(xa), ga, alpha * ga + (1 - alpha) * xa)
    ph = pa = np.zeros(len(d))
    if "promo_home" in d:
        known = d.season < season_of(asof)  # promotion is only known once the season is over
        ph, pa = (d.promo_home & known).to_numpy(float), (d.promo_away & known).to_numpy(float)
    w = np.exp(-xi * (asof - d.date).dt.days.to_numpy())
    if "div" in d:
        w = w * np.where(d["div"] == "E1", E1_WEIGHT, 1)
    x0 = np.zeros(4 + 2 * n)
    x0[0] = 0.3
    if prev:
        x0[:4] = prev["x"][:4]
        for t, i in idx.items():
            if (j := prev["teams"].get(t)) is not None:
                x0[4 + i], x0[4 + n + i] = prev["x"][4 + j], prev["x"][4 + prev["n"] + j]
    bounds = [(None, None), (None, None), (-0.2, 0.2), (None, None)] + [(None, None)] * (2 * n)
    res = minimize(_nll, x0, args=(hi, ai, yh, ya, gh, ga, ph, pa, w, n), jac=True, method="L-BFGS-B", bounds=bounds)
    return {"teams": idx, "x": res.x, "n": n, "asof": asof}


def rates(m, home, away):
    """Expected goals (home, away) for a fixture."""
    x, n, i, j = m["x"], m["n"], m["teams"][home], m["teams"][away]
    att, dfn = x[4:4 + n], x[4 + n:]
    return np.exp(x[0] + x[1] + att[i] + dfn[j]), np.exp(x[0] + att[j] + dfn[i])


def score_matrix(lam, mu, rho, size=11):
    """P(home goals = i, away goals = j) with the Dixon-Coles low-score correction."""
    from scipy.stats import poisson
    m = np.outer(poisson.pmf(np.arange(size), lam), poisson.pmf(np.arange(size), mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def markets(m):
    total = np.add.outer(np.arange(len(m)), np.arange(len(m)))
    out = {"H": np.tril(m, -1).sum(), "D": np.trace(m), "A": np.triu(m, 1).sum(), "BTTS": m[1:, 1:].sum()}
    for line in (1.5, 2.5, 3.5):
        out[f"O{line}"] = m[total > line].sum()
        out[f"U{line}"] = 1 - out[f"O{line}"]
    return out


def predict_match(m, home, away):
    lam, mu = rates(m, home, away)
    sm = score_matrix(lam, mu, m["x"][2])
    top = np.argsort(sm, axis=None)[::-1][:5]
    scores = [(f"{i // len(sm)}-{i % len(sm)}", sm.flat[i]) for i in top]
    return {"xg_h": lam, "xg_a": mu, **markets(sm), "scores": scores}


def command(fn):
    COMMANDS[fn.__name__] = fn
    return fn


COMMANDS = {"fetch": fetch}
EV_MIN, KELLY_FRAC, MAX_STAKE = 0.03, 0.25, 0.02
COMMISSION = 0.05  # exchange commission on winnings
BOOKS = {"B365": "Bet365", "BFD": "Betfred", "BMGM": "BetMGM", "BV": "BetVictor", "BW": "Bet&Win", "CL": "Coral",
         "LB": "Ladbrokes", "PP": "Paddy Power", "SKB": "Sky Bet", "WH": "William Hill", "1XB": "1XBet",
         "PS": "Pinnacle", "BFE": "Betfair Exchange"}
SHARP = ["Betfair Exchange", "Betfair", "Pinnacle"]  # de-vig these first; "Betfair" is the AU exchange in The Odds API
EXCHANGES = {"Betfair Exchange", "Betfair"}
GROUPS = {"H": "1X2", "D": "1X2", "A": "1X2", "O2.5": "OU", "U2.5": "OU"}


def pct(v):
    return "" if pd.isna(v) else f"{v:.0%}"


# ---------------------------------------------------------------- tactics

def team_profiles(y=CURRENT):
    """Per-team style and luck indicators for season `y` (Understat + football-data box scores)."""
    rows = []
    for t in load_understat(y)["teams"].values():
        h = pd.DataFrame(t["history"])
        att, dfn = sum(p["att"] for p in h.ppda), sum(p["def"] for p in h.ppda)
        o_att, o_def = sum(p["att"] for p in h.ppda_allowed), sum(p["def"] for p in h.ppda_allowed)
        rows.append(dict(
            team=canon(t["title"]), mp=len(h), pts=h.pts.sum(), xpts=h.xpts.sum(),
            xg=h.xG.mean(), xga=h.xGA.mean(), npxgd=(h.npxG - h.npxGA).mean(),
            ppda=att / max(dfn, 1), opp_ppda=o_att / max(o_def, 1),
            deep=h.deep.mean(), deep_allowed=h.deep_allowed.mean(),
            g_minus_xg=h.scored.sum() - h.xG.sum(), ga_minus_xga=h.missed.sum() - h.xGA.sum(),
            form_xgd=(h.xG - h.xGA).tail(5).mean()))
    df = pd.DataFrame(rows).set_index("team")
    fd = load_fd("E0", [y])
    for c in ("HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"):
        fd[c] = pd.to_numeric(fd[c], errors="coerce")
    cols = ["shots", "shots_ag", "sot", "sot_ag", "corners", "fouls", "yellows", "reds"]
    long = pd.concat([
        fd.rename(columns=dict(home="team", HS="shots", AS="shots_ag", HST="sot", AST="sot_ag", HC="corners", HF="fouls", HY="yellows", HR="reds"))[["team", *cols]],
        fd.rename(columns=dict(away="team", AS="shots", HS="shots_ag", AST="sot", HST="sot_ag", AC="corners", AF="fouls", AY="yellows", AR="reds"))[["team", *cols]]])
    df = df.join(long.groupby("team")[cols].mean())
    q = df.quantile([1 / 3, 2 / 3])

    def tags(r):
        out = []
        if r.ppda <= q.ppda.iloc[0]: out.append("high press")
        if r.ppda >= q.ppda.iloc[1]: out.append("low block")
        if r.deep >= q.deep.iloc[1]: out.append("territorial")
        if r.opp_ppda <= q.opp_ppda.iloc[0]: out.append("gets pressed")
        if r.g_minus_xg >= 3: out.append("finishing above xG (regression risk)")
        if r.g_minus_xg <= -3: out.append("finishing below xG (due a bounce)")
        if r.ga_minus_xga <= -3: out.append("conceding below xGA")
        if r.ga_minus_xga >= 3: out.append("conceding above xGA")
        if r.form_xgd - (r.xg - r.xga) >= 0.4: out.append("form up")
        if r.form_xgd - (r.xg - r.xga) <= -0.4: out.append("form down")
        return ", ".join(out)

    df["style"] = df.apply(tags, axis=1)
    return df.sort_values("npxgd", ascending=False)


@command
def tactics():
    df = team_profiles()
    show = df.drop(columns="style").round(2)
    print(show.to_string())
    print("\nppda = opponent passes per defensive action (lower = more pressing); opp_ppda = same, faced in build-up")
    print("deep = completed passes within 20m of goal; g_minus_xg > 0 = scoring above chance quality\n")
    for t, s in df["style"].items():
        print(f"  {t:16s} {s}")
    return df


# ---------------------------------------------------------------- trends

def season_trends():
    e = load_fd("E0", range(FIRST, CURRENT + 1))
    for c in ("HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "B365H", "B365D", "B365A"):
        e[c] = pd.to_numeric(e.get(c), errors="coerce")
    goals = e.hg + e.ag
    e = e.assign(
        goals=goals, home_win=e.FTR.eq("H") * 100, draw=e.FTR.eq("D") * 100, away_win=e.FTR.eq("A") * 100,
        home_gd=e.hg - e.ag, over25=(goals > 2.5) * 100, btts=((e.hg > 0) & (e.ag > 0)) * 100, nil_nil=(goals == 0) * 100,
        shots=e.HS + e.AS, sot=e.HST + e.AST, corners=e.HC + e.AC, fouls=e.HF + e.AF,
        yellows=e.HY + e.AY, reds=e.HR + e.AR, margin=(1 / e.B365H + 1 / e.B365D + 1 / e.B365A - 1) * 100)
    fair = devig(np.c_[1 / e.B365H, 1 / e.B365D, 1 / e.B365A])
    outcome = e.FTR.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    e["odds_logloss"] = -np.log(fair[np.arange(len(e)), outcome])
    cols = ["goals", "home_win", "draw", "away_win", "home_gd", "over25", "btts", "nil_nil",
            "shots", "sot", "corners", "fouls", "yellows", "reds", "margin", "odds_logloss"]
    t = e.groupby("season")[cols].mean()
    t["conversion"] = t.goals / t.sot * 100
    us = []
    for y in range(2014, CURRENT + 1):
        h = pd.concat(pd.DataFrame(x["history"]) for x in load_understat(y)["teams"].values())
        us.append(dict(season=y, xg=h.xG.mean() * 2, ppda=sum(p["att"] for p in h.ppda) / sum(p["def"] for p in h.ppda),
                       deep=h.deep.mean()))
    t = t.join(pd.DataFrame(us).set_index("season"))
    t.index = [f"{y}/{(y + 1) % 100:02d}" for y in t.index]
    return t


@command
def trends():
    t = season_trends()
    print("Per-game averages by season (percentages for results; shots etc. from 2000/01; xG/PPDA from 2014/15)\n")
    print(t.round(2).to_string())
    dec = t.groupby([f"{s[:3]}0s" for s in t.index]).mean()
    print("\nBy decade:\n")
    print(dec.round(2).to_string())
    print("\nShots on target roughly halve in 2013/14 (football-data changed data provider): conversion isn't comparable across it.")
    return t


# ---------------------------------------------------------------- fixtures, predictions, season sim

def upcoming(days=4):
    """Next round of fixtures that haven't kicked off (from Understat's schedule), kickoff in local time.
    Games in progress are excluded: pre-match probabilities mean nothing against live prices."""
    us = understat_matches([CURRENT])
    fut = us[~us.played & (us.kickoff > NOW_UTC)].sort_values("kickoff")
    fut = fut[fut.kickoff < fut.kickoff.min() + pd.Timedelta(days=days)]
    return fut.assign(local=fut.kickoff.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ))[["home", "away", "local"]]


def predictions(m=None):
    m = m or fit(matches(), TODAY + pd.Timedelta(days=1))
    fair = fair_probs(odds())
    rows = []
    for f in upcoming().itertuples():
        p = predict_match(m, f.home, f.away)
        rows.append(dict(kickoff=f.local, home=f.home, away=f.away, **{k: v for k, v in p.items()},
                         **{f"mkt_{s}": fair.get((f.home, f.away, s)) for s in ("H", "D", "A", "O2.5")}))
    return pd.DataFrame(rows)


@command
def predict():
    df = predictions()
    prof = team_profiles()["style"]
    for r in df.to_dict("records"):
        print(f"\n{r['kickoff']:%a %d %b %H:%M}  {r['home']} v {r['away']}   expected goals {r['xg_h']:.2f}-{r['xg_a']:.2f}")
        print(f"   model   H {pct(r['H']):>4} D {pct(r['D']):>4} A {pct(r['A']):>4}   O2.5 {pct(r['O2.5']):>4}  BTTS {pct(r['BTTS'])}")
        if pd.notna(r["mkt_H"]):
            print(f"   market  H {pct(r['mkt_H']):>4} D {pct(r['mkt_D']):>4} A {pct(r['mkt_A']):>4}   O2.5 {pct(r['mkt_O2.5']):>4}")
        print("   likely  " + ", ".join(f"{s} {p:.0%}" for s, p in r["scores"]))
        for t in (r["home"], r["away"]):
            if prof.get(t):
                print(f"   {t}: {prof[t]}")
    return df


def simulate_season(m, n=10_000, seed=None):
    us = understat_matches([CURRENT])
    teams = sorted(set(us.home))
    ti = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    pts, gd, gf = np.zeros((n, k)), np.zeros((n, k)), np.zeros((n, k))
    for r in us[us.played].itertuples():
        i, j = ti[r.home], ti[r.away]
        pts[:, i] += 3 * (r.hg > r.ag) + (r.hg == r.ag)
        pts[:, j] += 3 * (r.ag > r.hg) + (r.hg == r.ag)
        gd[:, i] += r.hg - r.ag; gd[:, j] += r.ag - r.hg
        gf[:, i] += r.hg; gf[:, j] += r.ag
    now = pts[0].copy()
    rng = np.random.default_rng(seed)
    rho = m["x"][2]
    # ponytail: ratings frozen for the rest of the season; under-states spread slightly
    for r in us[~us.played].itertuples():
        sm = score_matrix(*rates(m, r.home, r.away), rho)
        s = rng.choice(sm.size, n, p=sm.ravel())
        hg, ag = s // len(sm), s % len(sm)
        i, j = ti[r.home], ti[r.away]
        pts[:, i] += 3 * (hg > ag) + (hg == ag)
        pts[:, j] += 3 * (ag > hg) + (hg == ag)
        gd[:, i] += hg - ag; gd[:, j] += ag - hg
        gf[:, i] += hg; gf[:, j] += ag
    key = pts * 1e6 + (gd + 500) * 1e3 + gf + rng.random(pts.shape)
    pos = (-key).argsort(1).argsort(1) + 1
    return pd.DataFrame(dict(team=teams, pts_now=now, exp_pts=pts.mean(0), avg_pos=pos.mean(0),
                             title=(pos == 1).mean(0), top4=(pos <= 4).mean(0), top5=(pos <= 5).mean(0),
                             relegated=(pos >= k - 2).mean(0))).sort_values("exp_pts", ascending=False, ignore_index=True)


@command
def season():
    df = simulate_season(fit(matches(), TODAY + pd.Timedelta(days=1)))
    out = df.copy()
    for c in ("title", "top4", "top5", "relegated"):
        out[c] = out[c].map(pct)
    print(out.round({"exp_pts": 1, "avg_pos": 1}).to_string(index=False))
    return df


# ---------------------------------------------------------------- odds and value

def odds():
    """Long table of current prices: home, away, book, sel (H/D/A/O2.5/U2.5), price."""
    rows = []
    api = DATA / "odds_api.json"
    if os.environ.get("ODDS_API_KEY") and api.exists():
        for ev in json.loads(api.read_text()):
            if pd.Timestamp(ev["commence_time"]).tz_convert(None) <= NOW_UTC:
                continue  # in-play prices: never compare them with a pre-match model
            h, a = canon(ev["home_team"]), canon(ev["away_team"])
            for b in ev["bookmakers"]:
                for mk in b["markets"]:
                    for o in mk["outcomes"]:
                        if mk["key"] == "h2h":
                            sel = "D" if o["name"] == "Draw" else "H" if o["name"] == ev["home_team"] else "A"
                        elif mk["key"] == "totals" and o.get("point") == 2.5:
                            sel = "O2.5" if o["name"] == "Over" else "U2.5"
                        else:
                            continue
                        rows.append(dict(home=h, away=a, book=b["title"], sel=sel, price=float(o["price"])))
    else:
        f = pd.read_csv(DATA / "fixtures.csv", encoding="utf-8-sig").query("Div == 'E0'")
        for r in f.to_dict("records"):
            for code, name in BOOKS.items():
                for sel, col in (("H", "H"), ("D", "D"), ("A", "A"), ("O2.5", ">2.5"), ("U2.5", "<2.5")):
                    if pd.notna(r.get(code + col)):
                        rows.append(dict(home=r["HomeTeam"], away=r["AwayTeam"], book=name, sel=sel, price=float(r[code + col])))
    df = pd.DataFrame(rows, columns=["home", "away", "book", "sel", "price"])
    us = understat_matches([CURRENT])
    unknown = (set(df.home) | set(df.away)) - set(us.home) - set(us.away)
    assert not unknown, f"add these odds-feed names to ALIASES: {unknown}"
    return df


def devig(imp):
    """Bookmaker margin out, row-wise: p = imp**k with k set so each row sums to 1 (power method).
    Unlike proportional scaling it handles the favourite-longshot bias: on 5,150 EPL games proportional
    put sub-10% shots at 7.3% vs 6.6% actual; power gave 6.6%."""
    imp = np.atleast_2d(np.asarray(imp, float))
    lo, hi = np.full(len(imp), 0.5), np.full(len(imp), 3.0)
    for _ in range(50):
        k = (lo + hi) / 2
        over = (imp ** k[:, None]).sum(1) > 1
        lo, hi = np.where(over, k, lo), np.where(over, hi, k)
    p = imp ** k[:, None]
    return p / p.sum(1, keepdims=True)


def fair_probs(o):
    """De-vigged market probability per (home, away, sel): sharp book if it quotes the market, else consensus."""
    out = {}
    o = o.assign(grp=o.sel.map(GROUPS), imp=1 / o.price)
    for (h, a, g), d in o.groupby(["home", "away", "grp"]):
        size = 3 if g == "1X2" else 2
        sharp = next((d[d.book == b] for b in SHARP if (d.book == b).sum() == size), None)
        imp = (sharp if sharp is not None else d).groupby("sel").imp.mean()
        if len(imp) == size:
            out |= {(h, a, s): p for s, p in zip(imp.index, devig(imp.to_numpy())[0])}
    return out


def kelly(p, odds):
    f = (p * odds - 1) / (odds - 1)
    return float(np.clip(KELLY_FRAC * f, 0, MAX_STAKE))


def pick_bets(o, probs, fair):
    """Best price per selection against the blended model/market probability. probs: (home, away) -> markets."""
    w = params()["w"]
    o = o.assign(eff=np.where(o.book.isin(EXCHANGES), 1 + (o.price - 1) * (1 - COMMISSION), o.price))
    rows = []
    for (h, a), d in o.groupby(["home", "away"]):
        if (p := probs.get((h, a))) is None:
            continue
        for sel, ds in d.groupby("sel"):
            if (h, a, sel) not in fair:
                continue
            best = ds.loc[ds.eff.idxmax()]
            pm, pf = p[sel], fair[(h, a, sel)]
            pb = w * pm + (1 - w) * pf
            label = {"H": h, "A": a, "D": "Draw", "O2.5": "Over 2.5", "U2.5": "Under 2.5"}[sel]
            rows.append(dict(fixture=f"{h} v {a}", bet=label, book=best.book, odds=best.price, model=pm, market=pf,
                             blend=pb, ev=pb * best.eff - 1, soft_line=pf * best.eff > 1, stake=kelly(pb, best.eff)))
    cols = ["fixture", "bet", "book", "odds", "model", "market", "blend", "ev", "soft_line", "stake"]
    return pd.DataFrame(rows, columns=cols).sort_values("ev", ascending=False, ignore_index=True)


def value_bets(m=None):
    m = m or fit(matches(), TODAY + pd.Timedelta(days=1))
    o = odds()
    fx = upcoming()
    return pick_bets(o.merge(fx[["home", "away"]]), {(h, a): predict_match(m, h, a) for h, a in zip(fx.home, fx.away)},
                     fair_probs(o))


@command
def value():
    df = value_bets()
    src = "The Odds API (AU)" if os.environ.get("ODDS_API_KEY") and (DATA / "odds_api.json").exists() else "football-data.co.uk"
    w = params()["w"]
    print(f"Odds: {src}.  Blend = {w:.0%} model + {1 - w:.0%} de-vigged market.  Listing EV > {EV_MIN:.0%}.\n")
    bets = df[df.ev > EV_MIN].copy()
    if bets.empty:
        print("No value bets this round. Closest:\n")
        bets = df.head(5).copy()
    for c in ("model", "market", "blend", "ev", "stake"):
        bets[c] = bets[c].map(lambda v: f"{v:.1%}")
    bets["soft_line"] = bets.soft_line.map({True: "yes", False: ""})
    print(bets.to_string(index=False))
    print("\nstake = quarter-Kelly as % of bankroll (capped at 2%). soft_line = price beats the sharp fair price on its own.")
    print("Big model-vs-market gaps usually mean team news the model can't see: check lineups before betting.")
    return df


# ---------------------------------------------------------------- player props (paper bets)

PROPS_API = ("https://api.the-odds-api.com/v4/sports/soccer_epl/events{path}?apiKey={key}"
             "&regions=au&markets=player_goal_scorer_anytime,player_assists,player_shots&oddsFormat=decimal")
PROPS_API_UCL = PROPS_API.replace("soccer_epl", "soccer_uefa_champs_league")
PROPS_DIR, LEDGER = DATA / "props", DATA / "props_ledger.csv"
PROP_MARKETS = {"player_goal_scorer_anytime": "Anytime scorer", "player_assists": "Assist", "player_shots": "Shots"}
PROPS_EV_MIN = 0.10  # untested model and big margins: demand more than the match markets
PROPS_MAX_ODDS = 15  # longshot props are mostly noise
PROPS_MAX_GAP = 1.6  # model > 1.6x the market's implied chance usually means a lineup/injury the model can't see
SHRINK_MIN = 300  # minutes of typical-for-position play mixed into each player's rate
LAST_SEASON_W = 0.5


def norm_name(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").replace(".", "").split())


def match_player(name, cands):
    """Understat id for a bookmaker's player name among `cands` (id -> name), or None if unsure.
    Books use full legal names ("Richarlison de Andrade", "Ogochukwu Onyeka Frank"), Understat short ones."""
    import difflib
    n = norm_name(name)
    by = {norm_name(v): k for k, v in cands.items()}
    if n in by:
        return by[n]
    t = set(n.split())
    hits = [k for v, k in by.items() if set(v.split()) <= t or t <= set(v.split())]
    if len(hits) == 1:
        return hits[0]
    close = difflib.get_close_matches(n, list(by), n=2, cutoff=0.85)
    return by[close[0]] if len(close) == 1 else None


@cache
def player_table(leagues=("EPL",)):
    """One row per current-season player in `leagues`: shrunk per-90 rates for xG, xA and shots, minutes per
    appearance. Last season counts from any of the five Understat leagues, so a summer signing keeps his record."""
    def load(y, lg):
        if not understat_path(lg, y).exists():
            return pd.DataFrame(columns=["id", "time", "games", "goals", "assists", "xG", "xA", "shots", "league"])
        df = pd.DataFrame(load_understat(y, lg)["players"]).assign(league=lg)
        for c in ("time", "games", "goals", "assists", "xG", "xA", "shots"):
            df[c] = pd.to_numeric(df[c])
        return df
    cur = pd.concat([load(CURRENT, lg) for lg in leagues]).drop_duplicates("id", keep="last")
    last = pd.concat([load(CURRENT - 1, lg) for lg in UNDERSTAT_LEAGUES]).groupby("id")[["time", "xG", "xA", "shots"]].sum()
    df = cur.merge(last.reset_index(), on="id", how="left", suffixes=("", "_l")).fillna(
        {"time_l": 0, "xG_l": 0, "xA_l": 0, "shots_l": 0})
    # ponytail: last season counts at half weight whatever club it was at; a role change shows up slowly
    mins = df.time + LAST_SEASON_W * df.time_l
    df["grp"] = [("GK" if "GK" in p else "F" if "F" in p else "M" if "M" in p else "D" if "D" in p else "S") for p in df.position]
    for stat, out in (("xG", "g90"), ("xA", "a90"), ("shots", "s90")):
        tot = df[stat] + LAST_SEASON_W * df[f"{stat}_l"]
        # median regular, not the minutes-weighted mean: the mean is dragged up by the few high-volume shooters
        reg = mins >= 450
        prior = (tot / mins * 90)[reg].groupby(df.grp[reg]).median().reindex(df.grp).to_numpy()
        df[out] = (tot + prior * SHRINK_MIN / 90) / ((mins + SHRINK_MIN) / 90)
    df["min_app"] = (df.time / df.games.clip(lower=1)).clip(upper=90)
    df["teams"] = df.team_title.map(lambda s: [canon(t) for t in s.split(",")])
    return df[df.games > 0].set_index("id")


def player_probs(pl, team, team_xg, team_xg_pg):
    """Per-player chances for one side of a fixture, conditional on the player appearing
    (AU books void anytime-scorer and shots bets on non-runners)."""
    from scipy.stats import poisson
    d = pl[pl.teams.map(lambda ts: team in ts)].copy()
    # this match vs the team's season average; Bayern at 3.8 xG a game at home in the league can drop to 1.3 v Arsenal
    ratio = float(np.clip(team_xg / team_xg_pg, 0.25, 2.5))
    share = ratio * d.min_app / 90
    d["p_goal"] = 1 - np.exp(-d.g90 * share)
    d["p_assist"] = 1 - np.exp(-d.a90 * share)
    d["exp_shots"] = d.s90 * share
    # ponytail: Poisson shots; real counts are over-dispersed, so high Over lines are under-priced. Fit a
    # negative binomial once per-match player shots are logged.
    d["shots_over"] = [lambda line, mu=mu: float(poisson.sf(np.floor(line), mu)) for mu in d.exp_shots]
    return d


def props_events(api):
    """Upcoming events for an Odds API sport (free), or [] with a message if unavailable."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("props needs ODDS_API_KEY; pricing only the props already fetched")
        return []
    try:
        return json.loads(get(api.format(path="", key=key).split("&regions")[0]))
    except urllib.error.HTTPError as e:
        print(f"The Odds API refused the props event list (HTTP {e.code}); pricing only the props already fetched")
    except (OSError, http.client.HTTPException) as e:
        print(f"The Odds API refused the props event list (unreachable: {e!r}); pricing only the props already fetched")
    return []


def fetch_event_props(api, ev):
    """One event's player-prop odds into data/props/, once books have posted them (1 credit per market offered)."""
    PROPS_DIR.mkdir(exist_ok=True)
    if (PROPS_DIR / f"{ev['id']}.json").exists():
        return
    try:
        body, hdr = fetch_url(api.format(path=f"/{ev['id']}/odds", key=os.environ["ODDS_API_KEY"]))
        if not json.loads(body).get("bookmakers"):
            return  # nothing posted yet (free); ask again next run instead of caching the empty answer
        (PROPS_DIR / f"{ev['id']}.json").write_bytes(body)
        print(f"fetched props for {ev['home_team']} v {ev['away_team']}, credits left:", hdr.get("x-requests-remaining"))
    except urllib.error.HTTPError as e:
        print(f"The Odds API refused the props request (HTTP {e.code}) for {ev['home_team']} v {ev['away_team']}")
    except (OSError, http.client.HTTPException) as e:
        print(f"The Odds API refused the props request (unreachable: {e!r}) for {ev['home_team']} v {ev['away_team']}")


def load_event_props(ev_id):
    p = PROPS_DIR / f"{ev_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def price_event(ev, sides, comp, espn_id=None):
    """Rows for every priced prop in one event. sides: (home label, away label) -> player_probs frames."""
    (home, away), frames = list(sides), list(sides.values())
    both = pd.concat([f.assign(side=lbl) for lbl, f in zip((home, away), frames)])
    both = both[~both.index.duplicated()]
    cands = both.player_name.to_dict()
    prices = [(b["title"], mk["key"], o["description"], o.get("point", 0.5), float(o["price"]))
              for b in ev["bookmakers"] if b["title"] not in EXCHANGES  # exchange props are empty 1.01 shells
              for mk in b["markets"] if mk["key"] in PROP_MARKETS
              for o in mk["outcomes"] if o["name"] in ("Yes", "Over") and o.get("description") != "No Scorer"]
    o = pd.DataFrame(prices, columns=["book", "market", "name", "line", "price"])
    o["pid"] = o.name.map(lambda n: match_player(n, cands))
    rows = []
    for (pid, mk, line), d in o.dropna(subset=["pid"]).groupby(["pid", "market", "line"]):
        r = both.loc[pid]
        p = r.p_goal if mk == "player_goal_scorer_anytime" else r.p_assist if mk == "player_assists" else r.shots_over(line)
        best = d.loc[d.price.idxmax()]
        rows.append(dict(comp=comp, espn_id=espn_id, event_id=ev["id"], kickoff=pd.Timestamp(ev["commence_time"]).tz_convert(None),
                         home=home, away=away, team=r.side, pid=pid, player=r.player_name, market=mk, line=line,
                         book=best.book, odds=best.price, books=len(d), model=p,
                         market_imp=(1 / d.price).median() if len(d) > 1 else np.nan, ev=p * best.price - 1,
                         base_games=r.games, base_goals=r.goals, base_assists=r.assists, base_shots=r.shots))
    fair = both.assign(comp=comp, match=f"{home} v {away}", team=both.side)
    return rows, fair


ESPN_TO_UNDERSTAT = {"Internazionale": "Inter", "Paris Saint-Germain": "Paris Saint Germain", "RB Leipzig": "RasenBallsport Leipzig"}


def understat_team(name, slug):
    """Understat team (canon) for an ESPN club in a top-five league, or None."""
    import difflib
    lg = next((k for k, v in UNDERSTAT_LEAGUES.items() if v == slug), None)
    if lg is None or not understat_path(lg, CURRENT).exists():
        return None
    titles = list(understat_team_xg(lg))
    name = canon(ESPN_TO_UNDERSTAT.get(name, name))  # EPL titles are stored in football-data form ("Man City")
    keys = {_team_key(t): t for t in titles}
    k = _team_key(name)
    if k in keys:
        return keys[k]
    hits = [t for kk, t in keys.items() if set(k.split()) <= set(kk.split()) or set(kk.split()) <= set(k.split())]
    if len(hits) == 1:  # "Marseille" / "Olympique Marseille"
        return hits[0]
    close = difflib.get_close_matches(k, list(keys), n=2, cutoff=0.75)
    return keys[close[0]] if len(close) == 1 else None


def epl_prop_rows(m):
    fx = upcoming()
    want = set(zip(fx.home, fx.away))
    for ev in props_events(PROPS_API):
        if (canon(ev["home_team"]), canon(ev["away_team"])) in want:
            fetch_event_props(PROPS_API, ev)
    pl, txg = player_table(), understat_team_xg()
    rows, fair = [], []
    evs = [json.loads(p.read_text()) for p in PROPS_DIR.glob("*.json")]
    for f in fx.itertuples():
        ev = next((e for e in evs if (canon(e["home_team"]), canon(e["away_team"])) == (f.home, f.away)
                   and pd.Timestamp(e["commence_time"]).tz_convert(None) > NOW_UTC), None)
        if ev is None:
            continue
        lam, mu = rates(m, f.home, f.away)
        r, fp = price_event(ev, {f.home: player_probs(pl, f.home, lam, txg[f.home]),
                                 f.away: player_probs(pl, f.away, mu, txg[f.away])}, "EPL")
        rows += r; fair.append(fp)
    return rows, fair


def ucl_prop_rows(em):
    """Props for the next UCL matchday, sides from the five Understat leagues only (their domestic per-90 rates,
    scaled to this match's expected goals from the cross-league model). Other clubs' players aren't priced."""
    fx = ucl_upcoming()
    if fx.empty:
        return [], []
    tl = team_leagues(euro_matches())
    us = {t: understat_team(n, tl.get(t)) for t, n in pd.concat([fx[["home_id", "home"]].set_axis(["id", "n"], axis=1),
                                                                  fx[["away_id", "away"]].set_axis(["id", "n"], axis=1)]).values}
    events = props_events(PROPS_API_UCL) if fx.date.min() <= NOW_UTC + pd.Timedelta(days=4) else []
    by_fx = {}
    for ev in events:
        if (f := match_fixture(ev, fx)) is not None and (us.get(f.home_id) or us.get(f.away_id)):
            by_fx[f.id] = ev
            fetch_event_props(PROPS_API_UCL, ev)
    pl = player_table(tuple(UNDERSTAT_LEAGUES))
    txg = {t: x for lg in UNDERSTAT_LEAGUES for t, x in understat_team_xg(lg).items()}
    rows, fair = [], []
    for f in fx.itertuples():
        ev = load_event_props(by_fx[f.id]["id"]) if f.id in by_fx else None
        if ev is None:
            continue
        lam, mu = euro_rates(em, f.home_id, f.away_id, f.neutral)
        empty = pl.iloc[:0].assign(p_goal=[], p_assist=[], exp_shots=[], shots_over=[])
        sides = {}
        for lbl, tid, x in ((f.home, f.home_id, lam), (f.away, f.away_id, mu)):
            t = us.get(tid)
            sides[lbl] = player_probs(pl, t, x, txg[t]) if t else empty
        r, fp = price_event(ev, sides, "UCL", espn_id=f.id)
        rows += r; fair.append(fp)
    return rows, fair


def prop_rows(m=None, em=None):
    """Every priced prop for the next EPL round and UCL matchday: model chance, best AU price, market median."""
    m = m or fit(matches(), TODAY + pd.Timedelta(days=1))
    em = em or fit_euro(euro_matches(), TODAY + pd.Timedelta(days=1))
    rows, fair = epl_prop_rows(m)
    r2, f2 = ucl_prop_rows(em)
    rows, fair = rows + r2, fair + f2
    if fair:  # model prices for everyone, incl. assists, which no AU book offers
        fp = pd.concat(fair).sort_values("p_goal", ascending=False).groupby(["match", "team"], sort=False).head(4)
        fp[["comp", "match", "team", "player_name", "min_app", "p_goal", "p_assist", "exp_shots"]].to_csv(
            DATA / "props_model.csv", index=False)
    return pd.DataFrame(rows)


def prop_label(r):
    return PROP_MARKETS[r["market"]] + (f" over {r['line']:g}" if r["market"] == "player_shots" else "")


def picks(df):
    gap = (df.model / df.market_imp).fillna(1)
    # two or more books, so the gap check has a market to compare with; one-book lines were mostly model error
    return df[(df.ev > PROPS_EV_MIN) & (df.odds <= PROPS_MAX_ODDS) & (gap <= PROPS_MAX_GAP) & (df.books >= 2)].copy()


def espn_box(event_id, slug="uefa.champions"):
    """Player name -> (appeared, goals, assists, shots) from ESPN's match summary; None until it's final."""
    d = json.loads(get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/summary?event={event_id}"))
    if not d.get("header", {}).get("competitions", [{}])[0].get("status", {}).get("type", {}).get("completed"):
        return None
    out = {}
    for team in d.get("rosters", []):
        for p in team.get("roster", []):
            st = {x["name"]: x.get("value") or 0 for x in p.get("stats", [])}
            out[p["athlete"]["displayName"]] = (bool(p.get("starter") or p.get("subbedIn")),
                                                int(st.get("totalGoals", 0)), int(st.get("goalAssists", 0)),
                                                int(st.get("totalShots", 0)))
    return out


def grade(r, got):
    won = got > (0.5 if r.market != "player_shots" else r.line)
    return ("win", r.stake * (r.odds - 1)) if won else ("loss", -r.stake)


def settle_ucl(ledger):
    """UCL props: grade from ESPN's box score (domestic Understat totals don't move in Europe)."""
    todo = ledger[ledger.result.isna() & (ledger.comp == "UCL")
                  & (pd.to_datetime(ledger.kickoff) < NOW_UTC - pd.Timedelta(hours=3))]
    for eid, grp in todo.groupby("espn_id"):
        try:
            box = espn_box(str(int(float(eid))))
        except (OSError, http.client.HTTPException, json.JSONDecodeError):
            continue  # try again next run
        if box is None:
            continue
        names = dict(enumerate(box))
        for i, r in grp.iterrows():
            k = match_player(r.player, names)
            appeared, goals, assists, shots = box[names[k]] if k is not None else (False, 0, 0, 0)
            if not appeared:
                ledger.loc[i, ["result", "pnl"]] = "void", 0.0  # not in the squad or unused sub
                continue
            got = {"player_goal_scorer_anytime": goals, "player_assists": assists, "player_shots": shots}[r.market]
            ledger.loc[i, ["result", "pnl"]] = grade(r, got)
    return ledger


def settle(ledger):
    """Grade EPL paper bets from the change in each player's Understat season totals across the match,
    UCL ones from ESPN's box score."""
    pl = pd.DataFrame(load_understat(CURRENT)["players"]).set_index("id")
    us = understat_matches([CURRENT])
    ledger["result"] = ledger.result.astype(object)
    ledger = settle_ucl(ledger)
    for i, r in ledger[ledger.result.isna() & (ledger.comp == "EPL")].iterrows():
        logged, ko = pd.Timestamp(r.logged), pd.Timestamp(r.kickoff)
        fx = us[(us.home == r.home) & (us.away == r.away) & (us.kickoff.dt.date == ko.date())]
        if fx.empty or not fx.played.iat[0]:
            continue
        team = us[us.played & ((us.home == r.team) | (us.away == r.team)) & (us.kickoff > logged)]
        if len(team) != 1 or str(r.pid) not in pl.index:
            ledger.loc[i, "result"] = "unsettled"  # totals now span more than this match
            continue
        p = pl.loc[str(r.pid)]
        if int(p.games) == r.base_games:
            ledger.loc[i, ["result", "pnl"]] = "void", 0.0
            continue
        got = {"player_goal_scorer_anytime": int(p.goals) - r.base_goals, "player_assists": int(p.assists) - r.base_assists,
               "player_shots": int(p.shots) - r.base_shots}[r.market]
        ledger.loc[i, ["result", "pnl"]] = grade(r, got)
    return ledger


def update_ledger(df):
    cols = ["logged", "comp", "espn_id", "event_id", "kickoff", "home", "away", "team", "pid", "player", "market", "line", "book", "odds",
            "model", "market_imp", "ev", "stake", "base_games", "base_goals", "base_assists", "base_shots", "result", "pnl"]
    led = pd.read_csv(LEDGER, dtype={"pid": str}) if LEDGER.exists() else pd.DataFrame(columns=cols)
    led = led.reindex(columns=cols).fillna({"comp": "EPL"})  # ledgers from before UCL props have no comp column
    new = picks(df).assign(logged=NOW_UTC, stake=lambda d: [kelly(p, o) for p, o in zip(d.model, d.odds)],
                           result=np.nan, pnl=np.nan)
    seen = set(zip(led.event_id, led.pid.astype(str), led.market, led.line))
    new = new[[k not in seen for k in zip(new.event_id, new.pid.astype(str), new.market, new.line)]]
    led = settle(pd.concat([led, new[cols]], ignore_index=True) if len(new) else led)
    led.to_csv(LEDGER, index=False)
    return led, len(new)


def ledger_summary(led, comp=None):
    done = led[led.result.isin(["win", "loss"]) & ((led.comp == comp) if comp else True)]
    if done.empty:
        return "no paper bets settled yet"
    exp = (done.model * done.stake * done.odds - done.stake).sum()
    return (f"{len(done)} settled: {(done.result == 'win').sum()} won ({(done.result == 'win').mean():.0%} vs "
            f"{done.model.mean():.0%} expected), P&L {done.pnl.sum():+.2%} of bankroll on {done.stake.sum():.1%} staked "
            f"(model expected {exp:+.2%})")


@command
def props():
    df = prop_rows()
    if df.empty:
        print("No player-prop prices for the next round yet.")
        return df
    df.to_csv(DATA / "props_latest.csv", index=False)
    led, n_new = update_ledger(df)
    print(f"Player props: {len(df)} priced lines, AU books.  Paper picks need EV > {PROPS_EV_MIN:.0%}, odds <= "
          f"{PROPS_MAX_ODDS}, model <= {PROPS_MAX_GAP}x the market's implied chance.\n")
    pk = picks(df).sort_values("ev", ascending=False)
    if pk.empty:
        print("No paper picks this round.")
    else:
        out = pd.DataFrame({"comp": pk.comp, "match": pk.home + " v " + pk.away, "player": pk.player,
                            "bet": pk.apply(prop_label, axis=1),
                            "book": pk.book, "odds": pk.odds, "model": pk.model.map("{:.1%}".format),
                            "market": pk.market_imp.map(lambda v: "" if pd.isna(v) else f"{v:.1%}"),
                            "ev": pk.ev.map("{:+.1%}".format)})
        print(out.to_string(index=False))
    print(f"\nlogged {n_new} new paper bets; EPL {ledger_summary(led, 'EPL')}; UCL {ledger_summary(led, 'UCL')}")
    print("Chances assume the player appears (non-runners are void). Check lineups: a benched starter halves these.")
    return df


# ---------------------------------------------------------------- Europe: cross-league model for the Champions League

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={ym}&limit=500"
EURO_COMPS = {"uefa.champions": "UCL", "uefa.champions_qual": "UCL qual", "uefa.europa": "UEL", "uefa.europa_qual": "UEL qual",
              "uefa.europa.conf": "UECL", "uefa.europa.conf_qual": "UECL qual"}
# Every domestic league ESPN carries that sends clubs to Europe. Clubs from the rest (Czechia, Ukraine, Croatia,
# Azerbaijan, ...) are rated from their European games alone, as one "other" league.
DOMESTIC = ["eng.1", "eng.2", "esp.1", "ger.1", "ita.1", "fra.1", "ned.1", "por.1", "bel.1", "tur.1", "sco.1", "gre.1",
            "aut.1", "sui.1", "den.1", "nor.1", "swe.1", "rus.1", "rou.1", "isr.1", "irl.1", "wal.1", "nir.1", "mlt.1"]
ESPN_FROM = "2019-07"
ESPN_DIR = DATA / "espn"


def espn_events(slug, ym):
    """Compact rows for one league-month: finished games (90-minute score) and scheduled ones."""
    rows = []
    for e in json.loads(get(ESPN.format(slug=slug, ym=ym.replace("-", "")))).get("events", []):
        c = e["competitions"][0]
        st = c["status"]["type"]
        side = {t["homeAway"]: t for t in c["competitors"]}
        if set(side) != {"home", "away"}:
            continue
        r = dict(id=e["id"], date=e["date"], slug=slug, done=st["completed"], neutral=bool(c.get("neutralSite")),
                 country=c.get("venue", {}).get("address", {}).get("country"),
                 home_id=side["home"]["team"]["id"], home=side["home"]["team"]["displayName"],
                 away_id=side["away"]["team"]["id"], away=side["away"]["team"]["displayName"], hg=None, ag=None)
        if st["completed"]:
            if st["name"] in ("STATUS_FINAL_AET", "STATUS_FINAL_PEN"):
                # extra time: count only goals in the 90 minutes (stoppage shows as 90'+n)
                g = {"home": 0, "away": 0}
                ids = {side[s]["team"]["id"]: s for s in side}
                for d in c.get("details", []):
                    clk = d.get("clock", {})
                    if d.get("scoringPlay") and not d.get("shootout") and (
                            clk.get("value", 1e9) <= 5400 or str(clk.get("displayValue", "")).startswith("90'+")):
                        g[ids.get(d["team"]["id"], "home")] += 1
                r["hg"], r["ag"] = g["home"], g["away"]
            else:
                r["hg"], r["ag"] = int(side["home"]["score"]), int(side["away"]["score"])
        rows.append(r)
    return rows


def fetch_espn():
    """Domestic leagues + European cups since 2019/20 from ESPN, one cached file per league-month.
    Months before last month never change, so a normal run refetches only ~60 small pages."""
    from concurrent.futures import ThreadPoolExecutor
    ESPN_DIR.mkdir(exist_ok=True)
    months = [str(p) for p in pd.period_range(ESPN_FROM, TODAY.to_period("M"), freq="M")]
    fresh = set(months[-2:])
    jobs = [(s, ym) for s in [*EURO_COMPS, *DOMESTIC] for ym in months
            if ym in fresh or not (ESPN_DIR / f"{s}_{ym}.json").exists()]

    def one(job):
        s, ym = job
        try:
            (ESPN_DIR / f"{s}_{ym}.json").write_text(json.dumps(espn_events(s, ym)))
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:  # URLError is an OSError
            return f"{s} {ym}: {e!r}"
    with ThreadPoolExecutor(6) as ex:
        errs = [e for e in ex.map(one, jobs) if e]
    print(f"fetched {len(jobs) - len(errs)} ESPN league-months" + (f"; {len(errs)} failed, e.g. {errs[0]}" if errs else ""))


@cache
def euro_matches():
    """Every ESPN game (domestic and European), one row each; `euro` marks the European cups."""
    rows = [r for p in sorted(ESPN_DIR.glob("*.json")) for r in json.loads(p.read_text())]
    df = pd.DataFrame(rows).drop_duplicates("id", keep="last")
    df["date"] = pd.to_datetime(df.date, utc=True).dt.tz_convert(None)
    df["euro"] = df.slug.isin(list(EURO_COMPS))
    return df.sort_values("date", ignore_index=True)


EURO_PARAMS = {"xi": 0.003, "euro_w": 1.0, "team_sd": 1.0, "by_country": False}  # backtested 2023/24-2026/27  # `ucl_backtest` rewrites data/euro_params.json


def euro_params():
    p = DATA / "euro_params.json"
    return {**EURO_PARAMS, **(json.loads(p.read_text()) if p.exists() else {})}


def team_leagues(d, by_country=False):
    """Each team's most recent domestic league in `d`. Teams seen only in Europe are 'other', or with
    `by_country` 'other:<country of their home European games>' so Czech clubs aren't averaged with Andorran ones."""
    dom = d[~d.euro]
    long = pd.concat([dom[["date", "home_id", "slug"]].rename(columns={"home_id": "team"}),
                      dom[["date", "away_id", "slug"]].rename(columns={"away_id": "team"})]).sort_values("date")
    out = long.groupby("team").slug.last().to_dict()
    if by_country:
        home = d[d.euro & ~d.neutral & d.country.notna()]
        for t, c in home.groupby("home_id").country.agg(lambda s: s.mode().iat[0]).items():
            out.setdefault(t, f"other:{c}")
    return out


def fit_euro(df, asof, xi=None, euro_w=None, team_sd=None, by_country=None, leagues=True, prev=None):
    """Dixon-Coles over ~24 domestic leagues and the European cups, goals only, 4-year window.
    A team's strength = its league's + its own deviation (shrunk with sd `team_sd`), so a club with few games
    is pulled toward its league, not toward the average of all clubs. European games are what put the
    leagues on one scale."""
    from scipy.optimize import minimize
    p = euro_params()
    xi, euro_w, team_sd = (p["xi"] if xi is None else xi), (p["euro_w"] if euro_w is None else euro_w), \
        (p["team_sd"] if team_sd is None else team_sd)
    by_country = p["by_country"] if by_country is None else by_country
    d = df[df.done & (df.date < asof) & (df.date >= asof - WINDOW)]
    teams = sorted(set(d.home_id) | set(d.away_id))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    tl = team_leagues(d, by_country)
    lgs = [*DOMESTIC, *sorted({v for v in tl.values() if v.startswith("other:")}), "other"] if leagues else []
    L = len(lgs)
    lg = np.array([lgs.index(tl.get(t, "other")) for t in teams]) if leagues else None
    hi, ai = d.home_id.map(idx).to_numpy(), d.away_id.map(idx).to_numpy()
    gh, ga = d.hg.to_numpy(float), d.ag.to_numpy(float)
    w = np.exp(-xi * (asof - d.date).dt.days.to_numpy()) * np.where(d.euro, euro_w, 1)
    hf = 1 - d.neutral.to_numpy(float)
    z = np.zeros(len(d))
    x0 = np.zeros(4 + 2 * n + 2 * L)
    x0[0] = 0.3
    if prev:
        x0[:4] = prev["x"][:4]
        for t, i in idx.items():
            if (j := prev["teams"].get(t)) is not None:
                x0[4 + i], x0[4 + n + i] = prev["x"][4 + j], prev["x"][4 + prev["n"] + j]
        for k, name in enumerate(lgs):
            if name in prev["leagues"]:
                j = prev["leagues"].index(name)
                x0[4 + 2 * n + k], x0[4 + 2 * n + L + k] = prev["x"][4 + 2 * prev["n"] + j], prev["x"][4 + 2 * prev["n"] + prev["L"] + j]
    bounds = [(None, None), (None, None), (-0.2, 0.2), (0, 0)] + [(None, None)] * (2 * n + 2 * L)
    res = minimize(_nll, x0, args=(hi, ai, gh, ga, gh, ga, z, z, w, n, hf, lg, L, team_sd), jac=True,
                   method="L-BFGS-B", bounds=bounds)
    names = pd.concat([d[["home_id", "home"]].set_axis(["id", "name"], axis=1),
                       d[["away_id", "away"]].set_axis(["id", "name"], axis=1)]).groupby("id").name.last().to_dict()
    return {"teams": idx, "x": res.x, "n": n, "L": L, "lg": lg, "leagues": lgs, "asof": asof, "names": names}


def euro_strength(m, t):
    """(attack, defence) on the common scale; a club the fit hasn't seen gets the 'other' league average."""
    x, n, L = m["x"], m["n"], m["L"]
    la, ld = (x[4 + 2 * n:4 + 2 * n + L], x[4 + 2 * n + L:]) if L else (np.zeros(1), np.zeros(1))
    if (i := m["teams"].get(t)) is None:
        return la[-1], ld[-1]
    k = m["lg"][i] if L else 0
    return x[4 + i] + la[k], x[4 + n + i] + ld[k]


def euro_rates(m, home, away, neutral=False):
    (ah, dh), (aa, da) = euro_strength(m, home), euro_strength(m, away)
    c, h = m["x"][0], 0 if neutral else m["x"][1]
    return np.exp(c + h + ah + da), np.exp(c + aa + dh)


def euro_predict(m, home, away, neutral=False):
    lam, mu = euro_rates(m, home, away, neutral)
    sm = score_matrix(lam, mu, m["x"][2])
    top = np.argsort(sm, axis=None)[::-1][:5]
    return {"xg_h": lam, "xg_a": mu, **markets(sm), "scores": [(f"{i // len(sm)}-{i % len(sm)}", sm.flat[i]) for i in top]}


def ucl_upcoming(days=4):
    """Next Champions League matchday from ESPN's schedule (team ids, not names)."""
    df = euro_matches()
    fut = df[(df.slug == "uefa.champions") & ~df.done & (df.date > NOW_UTC)].sort_values("date")
    fut = fut[fut.date < fut.date.min() + pd.Timedelta(days=days)]
    return fut.assign(local=fut.date.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ))


TEAM_NOISE = {"fc", "cf", "ac", "as", "sc", "afc", "sk", "fk", "ssc", "rc", "sv", "vfb", "vfl", "bc", "club", "de", "cp",
              "kv", "sl", "if", "bk", "ff", "nk", "pae", "the", "and", "calcio", "1", "04", "05"}


def _team_key(s):
    return " ".join(t for t in norm_name(s).replace("/", " ").split() if t not in TEAM_NOISE)


def match_fixture(ev, fx):
    """The ESPN fixture an Odds API event is, by kickoff (±1 day) and both names; None if unsure."""
    import difflib
    ko = pd.Timestamp(ev["commence_time"]).tz_convert(None)
    near = fx[(fx.date - ko).abs() <= pd.Timedelta(days=1)]
    sim = lambda a, b: difflib.SequenceMatcher(None, _team_key(a), _team_key(b)).ratio()
    score = [sim(ev["home_team"], r.home) + sim(ev["away_team"], r.away) for r in near.itertuples()]
    if not score or max(score) < 1.2:
        return None
    best = near.iloc[int(np.argmax(score))]
    return best if sorted(score)[-2:-1] != [max(score)] else None


ODDS_API_UCL = ODDS_API.replace("soccer_epl", "soccer_uefa_champs_league")


def fetch_ucl_odds():
    """Odds for the next UCL matchday, only when one is within 4 days (2 credits a call)."""
    key = os.environ.get("ODDS_API_KEY")
    api = DATA / "odds_api_ucl.json"
    fx = ucl_upcoming()
    if not key or fx.empty or fx.date.min() > NOW_UTC + pd.Timedelta(days=4):
        api.unlink(missing_ok=True)
        return
    try:
        body, hdr = fetch_url(ODDS_API_UCL.format(key=key))
        api.write_bytes(body)
        print("fetched odds_api_ucl.json, credits left:", hdr.get("x-requests-remaining"))
    except urllib.error.HTTPError as e:
        api.unlink(missing_ok=True)
        print(f"The Odds API refused the UCL request (HTTP {e.code}); no UCL prices this run")
    except (OSError, http.client.HTTPException) as e:
        api.unlink(missing_ok=True)
        print(f"The Odds API refused the UCL request (unreachable: {e!r}); no UCL prices this run")


def ucl_odds(fx):
    """Long price table for UCL fixtures, keyed by ESPN names."""
    rows, api = [], DATA / "odds_api_ucl.json"
    if api.exists():
        for ev in json.loads(api.read_text()):
            if pd.Timestamp(ev["commence_time"]).tz_convert(None) <= NOW_UTC or (f := match_fixture(ev, fx)) is None:
                continue
            for b in ev["bookmakers"]:
                for mk in b["markets"]:
                    for o in mk["outcomes"]:
                        if mk["key"] == "h2h":
                            sel = "D" if o["name"] == "Draw" else "H" if o["name"] == ev["home_team"] else "A"
                        elif mk["key"] == "totals" and o.get("point") == 2.5:
                            sel = "O2.5" if o["name"] == "Over" else "U2.5"
                        else:
                            continue
                        rows.append(dict(home=f.home, away=f.away, book=b["title"], sel=sel, price=float(o["price"])))
    return pd.DataFrame(rows, columns=["home", "away", "book", "sel", "price"])


def ucl_predictions(m=None):
    m = m or fit_euro(euro_matches(), TODAY + pd.Timedelta(days=1))
    fx = ucl_upcoming()
    o = ucl_odds(fx)
    fair = fair_probs(o) if len(o) else {}
    rows = [dict(kickoff=f.local, home=f.home, away=f.away, **euro_predict(m, f.home_id, f.away_id, f.neutral),
                 **{f"mkt_{s}": fair.get((f.home, f.away, s)) for s in ("H", "D", "A", "O2.5")})
            for f in fx.itertuples()]
    probs = {(r["home"], r["away"]): r for r in rows}
    return pd.DataFrame(rows), (pick_bets(o, probs, fair) if len(o) else pd.DataFrame())


EURO_TEST_FROM = pd.Timestamp("2023-07-01")
EURO_CONFIGS = {  # name -> overrides of EURO_PARAMS; the first is the default
    "leagues, xi=0.003, team sd 1.0": {},
    "no league effects": {"leagues": False},
    "other leagues by country": {"by_country": True},
    "European games x2": {"euro_w": 2.0},
    "xi=0.0045": {"xi": 0.0045},
    "xi=0.002": {"xi": 0.002},
    "team sd 0.5": {"team_sd": 0.5},
    "team sd 0.7": {"team_sd": 0.7},
    "xi=0.002, team sd 0.7": {"xi": 0.002, "team_sd": 0.7},
    "xi=0.003, team sd 0.7, by country": {"team_sd": 0.7, "by_country": True},
}


@command
def ucl_backtest(step=14):
    """Walk-forward on every European game since July 2023, refitting every `step` days on data known then.
    There is no free archive of European odds, so the yardsticks are base rates and the model's own variants."""
    import time
    df = euro_matches()
    test = df[df.euro & df.done & (df.date >= EURO_TEST_FROM)]
    y = np.select([test.hg > test.ag, test.hg == test.ag], [0, 1], 2)
    past = df[df.euro & df.done & (df.date < EURO_TEST_FROM)]
    base = np.bincount(np.select([past.hg > past.ag, past.hg == past.ag], [0, 1], 2), minlength=3) / len(past)
    starts = pd.date_range(EURO_TEST_FROM, test.date.max() + pd.Timedelta(days=1), freq=f"{step}D")
    out = {"base rates": np.tile(base, (len(test), 1))}
    for name, kw in EURO_CONFIGS.items():
        t0, prev, P = time.time(), None, np.zeros((len(test), 3))
        for s in starts:
            sel = np.flatnonzero((test.date >= s).to_numpy() & (test.date < s + pd.Timedelta(days=step)).to_numpy())
            if not len(sel):
                continue
            prev = fit_euro(df, s, prev=prev, **{**EURO_PARAMS, **kw})
            for k in sel:
                r = test.iloc[k]
                mk = markets(score_matrix(*euro_rates(prev, r.home_id, r.away_id, r.neutral), prev["x"][2]))
                P[k] = mk["H"], mk["D"], mk["A"]
        out[name] = P
        print(f"  {name}: {time.time() - t0:.0f}s")
    # Cross-league scale check: for each league, its clubs' actual minus predicted points per European game
    tl = team_leagues(df[df.date < test.date.max()])
    lh, la = test.home_id.map(tl).fillna("other").to_numpy(), test.away_id.map(tl).fillna("other").to_numpy()
    act_h = np.choose(y, [3, 1, 0]); act_a = np.choose(y, [0, 1, 3])
    bias = {}
    for k, P in out.items():
        exp_h, exp_a = 3 * P[:, 0] + P[:, 1], 3 * P[:, 2] + P[:, 1]
        cross = lh != la
        r = pd.concat([pd.DataFrame({"lg": lh[cross], "d": (act_h - exp_h)[cross]}),
                       pd.DataFrame({"lg": la[cross], "d": (act_a - exp_a)[cross]})])
        bias[k] = r.groupby("lg").d.agg(["mean", "count"])
    b0 = pd.DataFrame({k: v["mean"] for k, v in bias.items()}).assign(games=bias[next(iter(EURO_CONFIGS))]["count"])
    b0 = b0[b0.games >= 40].sort_values("games", ascending=False).round(3)
    print("\nactual minus predicted points per European game, by league (0 = on scale):\n")
    print(b0.drop(columns="base rates").to_string())
    ucl_only = (test.slug == "uefa.champions").to_numpy()
    rows = [[k, logloss(P, y), brier(P, y), logloss(P[ucl_only], y[ucl_only])] for k, P in out.items()]
    res = pd.DataFrame(rows, columns=["model", "logloss_all", "brier_all", "logloss_ucl"]).round(4)
    print(f"\n{len(test):,} European games ({ucl_only.sum():,} in the UCL proper) since {EURO_TEST_FROM:%b %Y}\n")
    print(res.to_string(index=False))
    best = min(EURO_CONFIGS, key=lambda k: res.set_index("model").logloss_all[k])
    kw = {**EURO_PARAMS, **{k: v for k, v in EURO_CONFIGS[best].items() if k != "leagues"}}
    if EURO_CONFIGS[best].get("leagues", True):
        (DATA / "euro_params.json").write_text(json.dumps(kw))
        print(f"\nsaved {best} to data/euro_params.json")
    (DATA / "euro_backtest.json").write_text(json.dumps({"games": int(len(test)), "ucl_games": int(ucl_only.sum()),
                                                         "from": f"{EURO_TEST_FROM:%b %Y}", "rows": rows, "best": best,
                                                         "bias": b0[[best, "games"]].reset_index(names="league").values.tolist()}))
    return res


@command
def ucl():
    pr, vb = ucl_predictions()
    if pr.empty:
        print("No Champions League fixtures scheduled.")
        return pr
    w = params()["w"]
    for r in pr.to_dict("records"):
        print(f"\n{r['kickoff']:%a %d %b %H:%M}  {r['home']} v {r['away']}   expected goals {r['xg_h']:.2f}-{r['xg_a']:.2f}")
        print(f"   model   H {pct(r['H']):>4} D {pct(r['D']):>4} A {pct(r['A']):>4}   O2.5 {pct(r['O2.5']):>4}  BTTS {pct(r['BTTS'])}")
        if pd.notna(r["mkt_H"]):
            print(f"   market  H {pct(r['mkt_H']):>4} D {pct(r['mkt_D']):>4} A {pct(r['mkt_A']):>4}   O2.5 {pct(r['mkt_O2.5']):>4}")
        print("   likely  " + ", ".join(f"{s} {p:.0%}" for s, p in r["scores"]))
    print(f"\nUCL value bets (blend {w:.0%} model + {1 - w:.0%} market, EV > {EV_MIN:.0%}):\n")
    bets = vb[vb.ev > EV_MIN].copy() if len(vb) else vb
    if bets.empty:
        print("none" if len(vb) else "no UCL prices (they are fetched only within 4 days of a matchday)")
    else:
        for c in ("model", "market", "blend", "ev", "stake"):
            bets[c] = bets[c].map(lambda v: f"{v:.1%}")
        bets["soft_line"] = bets.soft_line.map({True: "yes", False: ""})
        print(bets.to_string(index=False))
    return pr


# ---------------------------------------------------------------- backtest

def logloss(P, y):
    return -np.log(P[np.arange(len(y)), y]).mean()


def brier(P, y):
    return ((P - np.eye(3)[y]) ** 2).sum(1).mean()


def fair3(e, cols):
    return devig(1 / e[cols].apply(pd.to_numeric, errors="coerce").to_numpy())


def walk_forward(df, e0, xi, alpha):
    """Refit every Monday on data before it; predict that week's EPL games. Returns (n, 3) H/D/A probabilities."""
    P, prev = np.zeros((len(e0), 3)), None
    for wk, idx in e0.groupby("week").indices.items():
        prev = fit(df, wk, xi, alpha, prev)
        for k in idx:
            mk = markets(score_matrix(*rates(prev, e0.home.iat[k], e0.away.iat[k]), prev["x"][2]))
            P[k] = mk["H"], mk["D"], mk["A"]
    return P


@command
def backtest(seasons=range(2019, CURRENT)):
    df = matches(2014)
    e0 = df[(df["div"] == "E0") & df.season.isin(seasons)].reset_index(drop=True)
    e0["week"] = e0.date - pd.to_timedelta(e0.date.dt.weekday, unit="D")
    open_mkt = fair3(e0, ["PSH", "PSD", "PSA"])
    close_mkt = fair3(e0, ["PSCH", "PSCD", "PSCA"])
    e0 = e0[np.isfinite(open_mkt).all(1) & np.isfinite(close_mkt).all(1)].reset_index(drop=True)  # few lack Pinnacle odds
    open_mkt, close_mkt = fair3(e0, ["PSH", "PSD", "PSA"]), fair3(e0, ["PSCH", "PSCD", "PSCA"])
    y = e0.FTR.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    print(f"{len(e0)} EPL matches, seasons {seasons[0]}/{seasons[0] % 100 + 1:02d} to {seasons[-1]}/{seasons[-1] % 100 + 1:02d}\n")

    grid = {}
    for xi in (0.003, 0.0045, 0.006):
        for alpha in (0.6, 0.3, 0.0):
            grid[(xi, alpha)] = P = walk_forward(df, e0, xi, alpha)
            print(f"xi={xi:<6} alpha={alpha:<4} model log loss {logloss(P, y):.4f}  brier {brier(P, y):.4f}")
    xi, alpha = min(grid, key=lambda k: logloss(grid[k], y))
    P = grid[(xi, alpha)]
    ws = np.round(np.arange(0, 1.01, 0.1), 1)
    w = min(ws, key=lambda w: logloss(w * P + (1 - w) * open_mkt, y))
    print(f"\nbest: xi={xi} alpha={alpha}  blend weight w={w}\n")
    span = f"{seasons[0]}/{seasons[0] % 100 + 1:02d} to {seasons[-1]}/{seasons[-1] % 100 + 1:02d}"
    summary = {"matches": len(e0), "seasons": span, "params": {"xi": xi, "alpha": alpha, "w": float(w)},
               "accuracy": [], "bets": []}
    print(f"{'':28s}log loss  brier")
    for name, Q in [("model", P), ("opening market (Pinnacle)", open_mkt), (f"blend w={w}", w * P + (1 - w) * open_mkt),
                    ("closing market (Pinnacle)", close_mkt)]:
        print(f"{name:28s}{logloss(Q, y):.4f}   {brier(Q, y):.4f}")
        summary["accuracy"].append([name, logloss(Q, y), brier(Q, y)])

    # promoted sides: are they systematically mis-rated?
    prev_e0 = {s: set(df[(df["div"] == "E0") & (df.season == s - 1)].home) for s in seasons}
    for side, col in (("home", 0), ("away", 2)):
        mask = np.array([t not in prev_e0[s] for t, s in zip(e0[side], e0.season)])
        print(f"promoted {side} sides ({mask.sum()} games): win prob model {P[mask, col].mean():.3f}, "
              f"market {open_mkt[mask, col].mean():.3f}, actual {(y[mask] == col).mean():.3f}")

    print("\nBetting simulation, 1X2, flat 1-unit stakes, bet when EV > {:.0%}:".format(EV_MIN))
    print(f"{'strategy':22s}{'prices':8s}{'bets':>6s}{'ROI':>8s}{'CLV':>8s}")
    for name, Q in [("model only", P), (f"blend w={w}", w * P + (1 - w) * open_mkt), ("soft lines (w=0)", open_mkt)]:
        for book in ("B365", "Max"):
            price = e0[[f"{book}H", f"{book}D", f"{book}A"]].apply(pd.to_numeric, errors="coerce").to_numpy()
            bet = (Q * price - 1 > EV_MIN) & ~np.isnan(price)
            won = np.eye(3, dtype=bool)[y]
            profit = np.where(won, price - 1, -1)[bet]
            clv = (price * close_mkt - 1)[bet]  # expected return if Pinnacle's closing price is the truth
            print(f"{name:22s}{book:8s}{bet.sum():6d}{profit.mean():8.1%}{clv.mean():8.1%}")
            summary["bets"].append([name, book, int(bet.sum()), profit.mean(), clv.mean()])
    print("\nROI on a few hundred bets is noisy (±5-10%, far more on longshots). CLV = expected return valued at")
    print("Pinnacle's de-vigged closing price: the sturdier test of whether an edge is real.")
    (DATA / "params.json").write_text(json.dumps(summary["params"]))
    (DATA / "backtest.json").write_text(json.dumps(summary))
    print("saved data/params.json and data/backtest.json")


# ---------------------------------------------------------------- report

TREND_CHARTS = [  # (title, unit, [(series label, column)])
    ("Results: share of matches", "%", [("Home win", "home_win"), ("Draw", "draw"), ("Away win", "away_win")]),
    ("Goals per game", "", [("Goals", "goals"), ("Expected goals (xG)", "xg")]),
    ("Home advantage: home goal difference per game", "", [("Home goal difference", "home_gd")]),
    ("Over 2.5 goals: share of matches", "%", [("Over 2.5 goals", "over25")]),
    ("Fouls per game", "", [("Fouls", "fouls")]),
    ("Pressing: league PPDA (lower = more pressing)", "", [("PPDA", "ppda")]),
]

REPORT_CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#006300}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#0ca30c}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#0ca30c}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding-block:24px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1,h2{font-family:"Barlow Condensed","Arial Narrow",system-ui,sans-serif;font-weight:600;text-wrap:balance;letter-spacing:.01em}
h1{font-size:34px;line-height:1.1;margin:0 0 6px}h2{font-size:23px;margin:40px 0 4px}
.sub,.note{color:var(--ink2);margin:0 0 12px}.note{font-size:13px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin-top:12px}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th{text-align:left;color:var(--ink2);font-weight:600;border-bottom:1px solid var(--axis);padding:6px 10px;white-space:nowrap}
td{border-bottom:1px solid var(--grid);padding:6px 10px;white-space:nowrap}
td.wrapcell{white-space:normal;min-width:320px}
tr:last-child td{border-bottom:0}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:12px}
.chart h3{font-size:14px;margin:0 0 8px;font-weight:600}.chart .box{position:relative;height:240px}
details summary{cursor:pointer;color:var(--ink2);margin-top:12px}
.empty{color:var(--ink2)}
.tabs{display:flex;gap:4px;margin:20px 0 4px;border-bottom:1px solid var(--axis)}
.tabs button{font:600 15px/1.2 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink2);background:none;border:0;
border-bottom:2px solid transparent;margin-bottom:-1px;padding:10px 14px;cursor:pointer;border-radius:8px 8px 0 0}
.tabs button:hover{color:var(--ink);background:var(--surface)}
.tabs button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--s1)}
.tabs button:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
section[role="tabpanel"] > h2:first-of-type{margin-top:24px}
"""

REPORT_JS = """
const tabs = [...document.querySelectorAll('[role="tab"]')];
function show(id, focus) {
  for (const t of tabs) {
    const on = t.id === id;
    t.setAttribute('aria-selected', on); t.tabIndex = on ? 0 : -1;
    document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
    if (on && focus) t.focus();
  }
  try { localStorage.setItem('tab', id); } catch (e) {}
}
tabs.forEach((t, i) => {
  t.addEventListener('click', () => { show(t.id); history.replaceState(null, '', '#' + t.id.slice(2)); });
  t.addEventListener('keydown', e => {
    const d = {ArrowRight: 1, ArrowLeft: -1}[e.key];
    if (d) { e.preventDefault(); show(tabs[(i + d + tabs.length) % tabs.length].id, true); }
  });
});
let start = 't-' + location.hash.slice(1);
if (!document.getElementById(start)) { try { start = localStorage.getItem('tab'); } catch (e) { start = null; } }
if (start && document.getElementById(start)) show(start);
const T = JSON.parse(document.getElementById('trend-data').textContent);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const crosshair = {id: 'crosshair', afterDatasetsDraw(c) {
  const a = c.tooltip && c.tooltip.getActiveElements();
  if (!a || !a.length) return;
  const x = a[0].element.x, {top, bottom} = c.chartArea, g = c.ctx;
  g.save(); g.strokeStyle = css('--axis'); g.lineWidth = 1;
  g.beginPath(); g.moveTo(x, top); g.lineTo(x, bottom); g.stroke(); g.restore();
}};
const charts = T.charts.map(([title, unit, series], i) => new Chart(document.getElementById('c' + i), {
  type: 'line', plugins: [crosshair],
  data: {labels: T.seasons, datasets: series.map(([label, col]) => ({label, data: T.data[col],
    borderWidth: 2, pointRadius: 0, pointHoverRadius: 5, pointHoverBorderWidth: 2, tension: 0}))},
  options: {responsive: true, maintainAspectRatio: false, animation: false,
    interaction: {mode: 'index', intersect: false},
    plugins: {legend: {display: series.length > 1, align: 'start',
                       labels: {usePointStyle: true, pointStyle: 'line', boxWidth: 18}},
              tooltip: {usePointStyle: true, callbacks: {
                labelPointStyle: () => ({pointStyle: 'line', rotation: 0}),
                label: x => ` ${x.parsed.y == null ? '-' : x.parsed.y.toFixed(unit ? 1 : 2) + unit}  ${x.dataset.label}`}}},
    scales: {x: {grid: {display: false}, ticks: {maxTicksLimit: 8, maxRotation: 0}},
             y: {border: {display: false}, ticks: {maxTicksLimit: 6}}}}}));
function paint() {
  const s = [css('--s1'), css('--s2'), css('--s3')], ink2 = css('--ink2'), grid = css('--grid'), surf = css('--surface');
  for (const c of charts) {
    c.data.datasets.forEach((d, i) => Object.assign(d, {borderColor: s[i], backgroundColor: s[i],
      pointHoverBackgroundColor: s[i], pointHoverBorderColor: surf}));
    c.options.color = ink2;
    c.options.plugins.legend.labels.color = ink2;
    Object.assign(c.options.plugins.tooltip, {backgroundColor: surf, titleColor: css('--ink'), bodyColor: css('--ink'),
      borderColor: css('--axis'), borderWidth: 1});
    for (const ax of ['x', 'y']) c.options.scales[ax].ticks.color = css('--muted');
    c.options.scales.y.grid = {color: grid};
    c.update('none');
  }
}
paint();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', paint);
new MutationObserver(paint).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
"""


def html_table(df, wrap=()):
    """Escaped HTML table; `wrap` columns may wrap (long text)."""
    import html
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f'<td{" class=wrapcell" if c in wrap else ""}>{html.escape(str(v))}</td>'
                                    for c, v in zip(df.columns, row)) + "</tr>" for row in df.itertuples(index=False))
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def props_section(comp):
    """Paper picks, record and model prices for one competition, as HTML."""
    import html
    if not (DATA / "props_latest.csv").exists():
        return '<p class="empty">Run <code>uv run epl.py props</code> to fill this section.</p>'
    pp = pd.read_csv(DATA / "props_latest.csv", parse_dates=["kickoff"])
    pp = pp[(pp.comp if "comp" in pp else pd.Series("EPL", index=pp.index)) == comp]
    pk = picks(pp[pp.kickoff > NOW_UTC]).sort_values("ev", ascending=False)
    led = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=["result"])
    led = led.reindex(columns=[*led.columns, *(["comp"] if "comp" not in led else [])]).fillna({"comp": "EPL"})
    pk = pd.DataFrame({"Match": pk.home + " v " + pk.away, "Player": pk.player,
                       "Bet": pk.apply(prop_label, axis=1) if len(pk) else [], "Book": pk.book,
                       "Odds": pk.odds.map("{:.2f}".format), "Model": pk.model.map("{:.1%}".format),
                       "Market": pk.market_imp.map(lambda v: "–" if pd.isna(v) else f"{v:.1%}"),
                       "EV": pk.ev.map("{:+.1%}".format)})
    none = ("No paper picks this round." if len(pp) else
            "No AU player prices yet. Books usually post them a day or two before kickoff.")
    out = (f'<div class="card">{html_table(pk) if len(pk) else f"<p class=empty>{none}</p>"}</div>'
           f'<p class="note">Paper record: {html.escape(ledger_summary(led, comp))}.</p>')
    if (DATA / "props_model.csv").exists():
        fp = pd.read_csv(DATA / "props_model.csv")
        fp = fp[(fp.comp if "comp" in fp else pd.Series("EPL", index=fp.index)) == comp]
        if len(fp):
            fp = pd.DataFrame({"Match": fp.match, "Player": fp.player_name + " (" + fp.team + ")",
                               "Mins/app": fp.min_app.round().astype(int),
                               "Score": [f"{p:.0%} ({1 / p:.2f})" for p in fp.p_goal],
                               "Assist": [f"{p:.0%} ({1 / p:.2f})" for p in fp.p_assist],
                               "Exp. shots": fp.exp_shots.round(1)})
            out += ('<details><summary>Model prices, top four scorers per side (fair odds in brackets; assists have no '
                    f'AU market, compare these with any book)</summary><div class="card">{html_table(fp)}</div></details>')
    return out


@command
def report():
    import html
    m = fit(matches(), TODAY + pd.Timedelta(days=1))
    p = params()
    src = "The Odds API (AU bookmakers)" if os.environ.get("ODDS_API_KEY") and (DATA / "odds_api.json").exists() \
        else "football-data.co.uk"

    pr = predictions(m)
    prof = team_profiles()
    fx = pd.DataFrame({
        "Kickoff": pr.kickoff.dt.strftime("%a %d %b %H:%M"), "Match": pr.home + " v " + pr.away,
        "xG": pr.xg_h.map("{:.2f}".format) + "–" + pr.xg_a.map("{:.2f}".format),
        "Model H/D/A": [f"{pct(a)} / {pct(b)} / {pct(c)}" for a, b, c in zip(pr.H, pr.D, pr.A)],
        "Market H/D/A": [f"{pct(a)} / {pct(b)} / {pct(c)}" if pd.notna(a) else "–" for a, b, c in zip(pr.mkt_H, pr.mkt_D, pr.mkt_A)],
        "Over 2.5 model / market": [f"{pct(a)} / {pct(b) or '–'}" for a, b in zip(pr["O2.5"], pr["mkt_O2.5"])],
        "BTTS": pr.BTTS.map(pct),
        "Likely scores": [", ".join(f"{s} {q:.0%}" for s, q in sc[:3]) for sc in pr.scores],
        "Styles": [" | ".join(f"{t}: {prof['style'].get(t) or '–'}" for t in (h, a)) for h, a in zip(pr.home, pr.away)]})

    vb = value_bets(m)
    bets = vb[vb.ev > EV_MIN]
    vb_note = "" if len(bets) else '<p class="empty">No bets clear the threshold this round. The five closest:</p>'
    bets = (bets if len(bets) else vb.head(5)).copy()
    bets = pd.DataFrame({"Match": bets.fixture, "Bet": bets.bet, "Book": bets.book, "Odds": bets.odds.map("{:.2f}".format),
                         "Model": bets.model.map(pct), "Market": bets.market.map(pct), "Blend": bets.blend.map("{:.1%}".format),
                         "EV": bets.ev.map("{:+.1%}".format), "Soft line": bets.soft_line.map({True: "yes", False: ""}),
                         "Stake": bets.stake.map("{:.1%}".format)})

    props_html = {c: props_section(c) for c in ("EPL", "UCL")}
    upr, uvb = ucl_predictions()
    if upr.empty:
        ucl_html = '<p class="empty">No Champions League fixtures scheduled.</p>'
    else:
        ufx = pd.DataFrame({
            "Kickoff": upr.kickoff.dt.strftime("%a %d %b %H:%M"), "Match": upr.home + " v " + upr.away,
            "xG": upr.xg_h.map("{:.2f}".format) + "–" + upr.xg_a.map("{:.2f}".format),
            "Model H/D/A": [f"{pct(a)} / {pct(b)} / {pct(c)}" for a, b, c in zip(upr.H, upr.D, upr.A)],
            "Market H/D/A": [f"{pct(a)} / {pct(b)} / {pct(c)}" if pd.notna(a) else "–" for a, b, c in zip(upr.mkt_H, upr.mkt_D, upr.mkt_A)],
            "Over 2.5 model / market": [f"{pct(a)} / {pct(b) or '–'}" for a, b in zip(upr["O2.5"], upr["mkt_O2.5"])],
            "Likely scores": [", ".join(f"{s} {q:.0%}" for s, q in sc[:3]) for sc in upr.scores]})
        ucl_html = f'<div class="card">{html_table(ufx)}</div>'
        ub = uvb[uvb.ev > EV_MIN] if len(uvb) else uvb
        if len(uvb) == 0:
            ucl_html += '<p class="note">No UCL prices yet: they are fetched only within 4 days of a matchday.</p>'
        elif len(ub) == 0:
            ucl_html += f'<p class="note">No UCL bets clear EV {EV_MIN:.0%} this matchday.</p>'
        else:
            ub = pd.DataFrame({"Match": ub.fixture, "Bet": ub.bet, "Book": ub.book, "Odds": ub.odds.map("{:.2f}".format),
                               "Model": ub.model.map(pct), "Market": ub.market.map(pct), "Blend": ub.blend.map("{:.1%}".format),
                               "EV": ub.ev.map("{:+.1%}".format), "Soft line": ub.soft_line.map({True: "yes", False: ""}),
                               "Stake": ub.stake.map("{:.1%}".format)})
            ucl_html += f'<h3>Value bets</h3><div class="card">{html_table(ub)}</div>'
    ucl_bt = '<p class="empty">Run <code>uv run epl.py ucl_backtest</code> to fill this section.</p>'
    if (DATA / "euro_backtest.json").exists():
        eb = json.loads((DATA / "euro_backtest.json").read_text())
        et = pd.DataFrame(eb["rows"], columns=["Model", "Log loss, all European", "Brier, all European", "Log loss, UCL"]).round(4)
        ucl_bt = (f'<p class="sub">Walk-forward over {eb["games"]:,} European games ({eb["ucl_games"]:,} in the UCL proper) '
                     f'since {html.escape(eb["from"])}, refitting every two weeks. Lower is better. In use: {html.escape(eb["best"])}.</p>'
                     f'<div class="card">{html_table(et)}</div>')
    ss = simulate_season(m)
    ss = pd.DataFrame({"Team": ss.team, "Points now": ss.pts_now.astype(int), "Expected points": ss.exp_pts.round(1),
                       "Avg finish": ss.avg_pos.round(1), "Title": ss.title.map(pct), "Top 4": ss.top4.map(pct),
                       "Top 5": ss.top5.map(pct), "Relegated": ss.relegated.map(pct)})

    tp = prof.reset_index()
    tp = pd.DataFrame({"Team": tp.team, "MP": tp.mp, "Pts": tp.pts, "xPts": tp.xpts.round(1),
                       "xG/g": tp.xg.round(2), "xGA/g": tp.xga.round(2), "PPDA": tp.ppda.round(1),
                       "Opp PPDA": tp.opp_ppda.round(1), "Deep/g": tp.deep.round(1),
                       "G − xG": tp.g_minus_xg.round(1), "GA − xGA": tp.ga_minus_xga.round(1),
                       "Shots/g": tp.shots.round(1), "Shots ag/g": tp.shots_ag.round(1), "Style": tp["style"]})

    tr = season_trends()
    data = {c: [None if pd.isna(v) else round(float(v), 3) for v in tr[c]] for _, _, ser in TREND_CHARTS for _, c in ser}
    trend_json = json.dumps({"seasons": list(tr.index), "charts": TREND_CHARTS, "data": data}).replace("</", "<\\/")
    dec = tr.groupby([f"{s[:3]}0s" for s in tr.index]).mean()
    keep = {"goals": "Goals/g", "home_win": "Home win %", "draw": "Draw %", "away_win": "Away win %", "home_gd": "Home GD/g",
            "over25": "Over 2.5 %", "btts": "BTTS %", "shots": "Shots/g", "fouls": "Fouls/g", "yellows": "Yellows/g",
            "margin": "B365 margin %", "xg": "xG/g", "ppda": "PPDA"}
    dec = dec[list(keep)].rename(columns=keep).round(2).reset_index(names="Decade").fillna("–")
    full = tr.round(2).reset_index(names="Season").fillna("–")

    bt = json.loads((DATA / "backtest.json").read_text()) if (DATA / "backtest.json").exists() else None
    if bt:
        acc = pd.DataFrame(bt["accuracy"], columns=["Predictor", "Log loss", "Brier"]).round(4)
        bs = pd.DataFrame(bt["bets"], columns=["Strategy", "Prices", "Bets", "ROI", "CLV"])
        bs["ROI"], bs["CLV"] = bs.ROI.map("{:+.1%}".format), bs.CLV.map("{:+.1%}".format)
        bt_html = (f'<p class="sub">Walk-forward over {bt["matches"]:,} EPL matches ({html.escape(bt["seasons"])}), refitting weekly '
                   f'on data available at the time. Lower log loss and Brier are better.</p>'
                   f'<div class="card">{html_table(acc)}</div><div class="card">{html_table(bs)}</div>'
                   '<p class="note">CLV = expected return of each bet valued at Pinnacle\'s de-vigged closing price, the sturdier '
                   'test of a real edge; ROI on a few hundred bets swings ±5–10% on luck alone. "Max" is the best price across '
                   'all bookmakers football-data tracks, so it flatters what one account can get.</p>')
    else:
        bt_html = '<p class="empty">Run <code>uv run epl.py backtest</code> to fill this section.</p>'

    charts = "".join(f'<div class="card chart"><h3>{html.escape(t)}</h3><div class="box"><canvas id="c{i}" role="img" '
                     f'aria-label="{html.escape(t)}, by season. Values are in the season table below."></canvas></div></div>'
                     for i, (t, _, _) in enumerate(TREND_CHARTS))
    head = f"""<title>Football Predictor</title><meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600&display=swap">
<style>{REPORT_CSS}</style>"""
    body = f"""<main>
<h1>Football Predictor</h1>
<p class="sub">Generated {datetime.now():%a %d %b %Y %H:%M}. Odds from {html.escape(src)}.</p>

<div class="tabs" role="tablist" aria-label="Competition">
<button role="tab" id="t-epl" aria-controls="p-epl" aria-selected="true">Premier League</button>
<button role="tab" id="t-ucl" aria-controls="p-ucl" aria-selected="false" tabindex="-1">Champions League</button>
</div>

<section role="tabpanel" id="p-epl" aria-labelledby="t-epl">
<p class="sub">Dixon–Coles, time decay ξ={p['xi']}, xG weight {1 - p['alpha']:.0%}, blended {p['w']:.0%} model /
{1 - p['w']:.0%} market for value bets.</p>

<h2>Next round</h2>
<p class="sub">Model probabilities against the de-vigged market. Kickoffs in local time.</p>
<div class="card">{html_table(fx, wrap=("Styles",))}</div>

<h2>Value bets</h2>
<p class="sub">Best available price where the blended probability gives EV above {EV_MIN:.0%}. Stake = quarter-Kelly, % of bankroll,
capped at {MAX_STAKE:.0%}. "Soft line" = the price beats the sharp market's fair price on its own.</p>
<div class="card">{vb_note}{html_table(bets)}</div>
<p class="note">A big model-vs-market gap usually means team news the model can't see. Check lineups first.</p>

<h2>Player props (paper bets)</h2>
<p class="sub">Anytime scorer, assists and shots from each player's shrunk xG, xA and shots per 90, scaled to this match's
expected goals. Picks need EV above {PROPS_EV_MIN:.0%} at the best AU price, odds up to {PROPS_MAX_ODDS}, and the model no more
than {PROPS_MAX_GAP}× the market's chance. Logged, not staked, until the record says otherwise. Market = median implied chance
across AU books, margin included.</p>
{props_html["EPL"]}
<p class="note">Chances assume the player appears (books void non-runners). A benched starter roughly halves them: check lineups.</p>

<h2>Season simulation</h2>
<p class="sub">10,000 simulations of the remaining fixtures from current ratings.</p>
<div class="card">{html_table(ss)}</div>

<h2>Team profiles, {CURRENT}/{(CURRENT + 1) % 100:02d}</h2>
<p class="sub">PPDA = opponent passes per defensive action (lower = more pressing). Opp PPDA = the same, faced in build-up.
Deep = completed passes within 20 m of goal. G − xG above 0 = scoring above chance quality (tends to regress).</p>
<div class="card">{html_table(tp, wrap=("Style",))}</div>

<h2>How the Premier League has changed</h2>
<p class="sub">Season averages since 1993/94. Shots and fouls from 2000/01, xG and PPDA from 2014/15 (Understat).</p>
<div class="charts">{charts}</div>
<div class="card">{html_table(dec)}</div>
<details><summary>Every season, every measure</summary><div class="card">{html_table(full)}</div></details>
<p class="note">Shots on target roughly halve in 2013/14 when football-data changed data provider, so conversion rates before
and after are not comparable. The current season is partial.</p>

<h2>Backtest</h2>
{bt_html}
</section>

<section role="tabpanel" id="p-ucl" aria-labelledby="t-ucl" hidden>
<p class="sub">Cross-league Dixon–Coles on goals from {len(DOMESTIC)} domestic leagues plus every European cup and qualifier since
2019/20 (ESPN). A club's strength is its league's plus its own; European games put the leagues on one scale. Clubs from
leagues ESPN doesn't carry (Czechia, Ukraine, Croatia, Azerbaijan...) are rated from European games alone. Value bets use the
same {p['w']:.0%} model / {1 - p['w']:.0%} market blend as the EPL. That blend was tuned on EPL games, not European ones.</p>

<h2>Next matchday</h2>
{ucl_html}

<h2>Player props (paper bets)</h2>
<p class="sub">Anytime scorer, assists and shots from each player's shrunk xG, xA and shots per 90 in their domestic league (Understat's five leagues; other clubs' players aren't priced), scaled to this match's
expected goals. Picks need EV above {PROPS_EV_MIN:.0%} at the best AU price, odds up to {PROPS_MAX_ODDS}, and the model no more
than {PROPS_MAX_GAP}× the market's chance. Logged, not staked, until the record says otherwise. Market = median implied chance
across AU books, margin included.</p>
{props_html["UCL"]}
<p class="note">Chances assume the player appears (books void non-runners). A benched starter roughly halves them: check lineups.</p>

<h2>Backtest</h2>
{ucl_bt}
</section>
</main>
<script type="application/json" id="trend-data">{trend_json}</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>{REPORT_JS}</script>"""
    out = Path(__file__).parent / "report.html"
    out.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
                   f'content="width=device-width,initial-scale=1,viewport-fit=cover">{head}</head><body>{body}</body></html>')
    (DATA / "report_page.html").write_text(head + body)  # same page without the document wrapper, for claude.ai
    print("wrote", out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if cmd not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[cmd]()
