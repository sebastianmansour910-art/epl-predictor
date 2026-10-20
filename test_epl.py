# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "pandas"]
# ///
"""Self-check: uv run test_epl.py"""
import numpy as np
import pandas as pd
from scipy.optimize import check_grad

import epl

rng = np.random.default_rng(0)

# gradient of the likelihood matches finite differences (incl. the rho terms)
n, k = 6, 400
hi, ai = rng.integers(0, n, k), rng.integers(0, n, k)
gh, ga = rng.poisson(1.3, k).astype(float), rng.poisson(1.1, k).astype(float)
ph = (rng.random(k) < 0.2).astype(float)
args = (hi, ai, gh + rng.random(k) * 0.3, ga, gh, ga, ph, (rng.random(k) < 0.2) * (1 - ph), rng.random(k), n)
x = np.r_[0.2, 0.3, -0.1, 0.15, rng.normal(0, 0.3, 2 * n)]
assert check_grad(lambda v: epl._nll(v, *args)[0], lambda v: epl._nll(v, *args)[1], x) < 1e-3
# ... and with league effects, neutral venues and a tighter team prior (the cross-league model)
L, lg = 3, rng.integers(0, 3, n)
kw = dict(hf=(rng.random(k) < 0.9).astype(float), lg=lg, L=L, team_sd=0.5)
x = np.r_[x, rng.normal(0, 0.3, 2 * L)]
assert check_grad(lambda v: epl._nll(v, *args, **kw)[0], lambda v: epl._nll(v, *args, **kw)[1], x) < 1e-3

# fit recovers known ratings from synthetic seasons
n = 20
att, dfn, c, h = rng.normal(0, 0.3, n), rng.normal(0, 0.3, n), 0.1, 0.25
asof = pd.Timestamp("2024-01-01")
rows = [dict(date=asof - pd.Timedelta(days=int(rng.integers(1, 1400))), home=f"T{i}", away=f"T{j}",
             hg=rng.poisson(np.exp(c + h + att[i] + dfn[j])), ag=rng.poisson(np.exp(c + att[j] + dfn[i])))
        for _ in range(6) for i in range(n) for j in range(n) if i != j]
m = epl.fit(pd.DataFrame(rows), asof, xi=0, alpha=1)
order = [m["teams"][f"T{i}"] for i in range(n)]
fa, fd = m["x"][4:4 + n][order], m["x"][4 + n:][order]
assert abs(m["x"][1] - h) < 0.06, m["x"][1]
assert abs(m["x"][2]) < 0.1, m["x"][2]
assert np.abs((fa - fa.mean()) - (att - att.mean())).mean() < 0.08
assert np.abs((fd - fd.mean()) - (dfn - dfn.mean())).mean() < 0.08

# score matrix and derived markets are proper probabilities
for lam, mu, rho in [(1.5, 1.1, -0.1), (3.2, 0.4, 0.05), (0.6, 0.6, -0.2)]:
    sm = epl.score_matrix(lam, mu, rho)
    mk = epl.markets(sm)
    assert (sm >= 0).all() and abs(sm.sum() - 1) < 1e-9
    assert abs(mk["H"] + mk["D"] + mk["A"] - 1) < 1e-9
    assert abs(sum(i * p for i, p in enumerate(sm.sum(1))) - lam) < 0.05  # mean home goals ~ lam

# de-vig: rows sum to 1, symmetric markets split evenly, longshots shaded below proportional
imp = np.array([1 / 1.4, 1 / 4.8, 1 / 9.0])
p = epl.devig(imp)[0]
assert abs(p.sum() - 1) < 1e-9 and p[2] < imp[2] / imp.sum()
assert np.allclose(epl.devig([1 / 1.95, 1 / 1.95]), 0.5)

# Kelly: nothing staked without an edge, capped with one
assert epl.kelly(0.5, 1.9) == 0 and epl.kelly(0.5, 2.0) == 0
assert 0 < epl.kelly(0.55, 2.0) <= epl.MAX_STAKE and epl.kelly(0.9, 3.0) == epl.MAX_STAKE

# odds from The Odds API: games already kicked off (live prices) are dropped
import json, os, tempfile
from pathlib import Path
d = Path(tempfile.mkdtemp())
game = {"h": {"title": "Arsenal"}, "a": {"title": "Chelsea"}, "isResult": False, "datetime": "2099-01-01 15:00:00",
        "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None}}
(d / f"understat_{epl.CURRENT}.json").write_text(json.dumps({"dates": [game]}))
book = lambda price: [{"title": "TAB", "markets": [{"key": "h2h", "outcomes": [
    {"name": "Arsenal", "price": price}, {"name": "Draw", "price": 4.0}, {"name": "Chelsea", "price": 3.0}]}]}]
(d / "odds_api.json").write_text(json.dumps([
    {"commence_time": "2000-01-01T15:00:00Z", "home_team": "Arsenal", "away_team": "Chelsea", "bookmakers": book(351.0)},
    {"commence_time": "2099-01-01T15:00:00Z", "home_team": "Arsenal", "away_team": "Chelsea", "bookmakers": book(2.1)}]))
epl.DATA, os.environ["ODDS_API_KEY"] = d, "test"
o = epl.odds()
assert len(o) == 3 and o.query("sel == 'H'").price.tolist() == [2.1], o


# bookmaker names map onto Understat's short ones, and ambiguous names are refused
c = {"1": "Richarlison", "2": "Sávio", "3": "Frank Onyeka", "4": "Micky van de Ven", "5": "Ben Davies", "6": "Ben White"}
assert epl.match_player("Richarlison de Andrade", c) == "1"
assert epl.match_player("Savio Moreira de Oliveira", c) == "2"
assert epl.match_player("Ogochukwu Onyeka Frank", c) == "3"
assert epl.match_player("Mickey van de Ven", c) == "4"
assert epl.match_player("Ben", c) is None
assert epl.match_player("Erling Haaland", c) is None

# Odds API fixtures find their ESPN twin by kickoff and both names
fx = pd.DataFrame(dict(date=pd.to_datetime(["2026-10-20 19:00"] * 3), home=["Paris Saint-Germain", "Porto", "Bodo/Glimt"],
                       away=["Barcelona", "PSV Eindhoven", "Napoli"]))
ev = lambda h, a, t="2026-10-20T19:00:00Z": dict(commence_time=t, home_team=h, away_team=a)
assert epl.match_fixture(ev("Paris Saint Germain", "Barcelona"), fx).home == "Paris Saint-Germain"
assert epl.match_fixture(ev("FC Porto", "PSV"), fx).home == "Porto"
assert epl.match_fixture(ev("Bodø/Glimt", "SSC Napoli"), fx).home == "Bodo/Glimt"
assert epl.match_fixture(ev("Paris Saint Germain", "Barcelona", "2026-10-25T19:00:00Z"), fx) is None
assert epl.match_fixture(ev("Real Madrid", "RB Leipzig"), fx) is None

# downloads retry dropped connections, but not a 4xx
import io, time, urllib.error, urllib.request
calls = []
class Resp(io.BytesIO):
    headers = {"x-requests-remaining": "7"}
    def __enter__(self): return self
    def __exit__(self, *a): pass
def flaky(req, timeout):
    calls.append(1)
    if len(calls) < 3:
        raise ConnectionResetError(54, "Connection reset by peer")
    return Resp(b"ok")
real_open, real_sleep = urllib.request.urlopen, time.sleep
urllib.request.urlopen, time.sleep = flaky, lambda s: None
try:
    assert epl.fetch_url("http://x")[0] == b"ok" and len(calls) == 3
    calls.clear()
    def denied(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError("http://x", 401, "no", {}, None)
    urllib.request.urlopen = denied
    try:
        epl.fetch_url("http://x"); raise AssertionError("401 should raise")
    except urllib.error.HTTPError:
        assert len(calls) == 1
finally:
    urllib.request.urlopen, time.sleep = real_open, real_sleep
print("ok")
