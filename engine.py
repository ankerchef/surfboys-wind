#!/usr/bin/env python3
"""SURFBOYS wind engine.

Runs hourly (GitHub Actions). Pulls six deterministic models + two ensembles
from Open-Meteo, scores every daylight hour per spot for kiting, blends the
models with weights learned from verification, and every day checks
yesterday's forecasts against measured wind from the nearest DMI station.

Outputs (in data/):
  latest.json   forecast for today + 5 days, per spot, per hour, per model
  verify.json   model skill (MAE per lead day), hit rate, daily log
  brief.md      Danish bullet summary for the 9:00 update
  history/fc-YYYY-MM-DD.json   the morning forecast snapshot (for verification)

Stdlib only. Set MOCK_DIR to read API responses from files instead of the web.
"""
import json, math, os, sys, time, urllib.request, urllib.parse, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
HIST = DATA / "history"
TZ = "Europe/Copenhagen"
DAYS = 6  # today + 5

# Deterministic models (Open-Meteo ids). Seamless = high-res first, then global.
MODELS = {
    "dmi_seamless":   "DMI HARMONIE 2 km",
    "icon_seamless":  "DWD ICON-D2 2 km",
    "metno_seamless": "MET Nordic 1 km",
    "ukmo_seamless":  "UK Met Office",
    "ecmwf_ifs025":   "ECMWF IFS",
    "gfs_seamless":   "NOAA GFS",
}
ENSEMBLES = {"ecmwf_ifs025": "ECMWF ENS (51)", "icon_seamless": "ICON-EPS (40)"}
HIRES = {"dmi_seamless", "icon_seamless", "metno_seamless"}

DMI_OBS = "https://opendataapi.dmi.dk/v2/metObs/collections"
FALLBACK_STATIONS = {  # stationId, name — used only if station lookup fails
    "lynaes": ("06159", "Røsnæs Fyr"), "sydvest": ("06180", "Københavns Lufthavn"),
    "sylten": ("06183", "Drogden Fyr"), "hornbaek": ("06168", "Nakkehoved Fyr"),
    "kabel": ("06169", "Gniben"), "gudmindrup": ("06169", "Gniben"),
}

# ---------------------------------------------------------------- helpers
def http_json(url, tries=3):
    mock = os.environ.get("MOCK_DIR")
    if mock:
        key = "".join(c if c.isalnum() else "_" for c in url)[:180]
        p = Path(mock) / (key + ".json")
        return json.loads(p.read_text()) if p.exists() else None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "surfboys-wind/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"  ! {e} ({url[:90]}...)", file=sys.stderr)
            time.sleep(2 + 3 * i)
    return None


def in_sector(d, sec):
    a, b = sec
    return (a <= d <= b) if a <= b else (d >= a or d <= b)

def ang_diff(a, b):
    d = abs(a - b) % 360
    return 360 - d if d > 180 else d

def vec_mean(dirs, weights):
    x = sum(w * math.sin(math.radians(d)) for d, w in zip(dirs, weights))
    y = sum(w * math.cos(math.radians(d)) for d, w in zip(dirs, weights))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def wmean(vals, weights):
    s = sum(weights)
    return sum(v * w for v, w in zip(vals, weights)) / s if s else None

def wstd(vals, weights):
    m = wmean(vals, weights)
    s = sum(weights)
    return math.sqrt(sum(w * (v - m) ** 2 for v, w in zip(vals, weights)) / s) if s else 0

# ---------------------------------------------------------------- scoring
def speed_score(kn, T):
    if kn is None: return 0
    if kn < T["foil"]: return 0
    if kn < T["min"]: return 0.25
    if kn < T["good_lo"]: return 0.25 + 0.75 * (kn - T["min"]) / (T["good_lo"] - T["min"])
    if kn <= T["good_hi"]: return 1.0
    if kn <= T["overpowered"]: return 1.0 - 0.2 * (kn - T["good_hi"]) / (T["overpowered"] - T["good_hi"])
    if kn <= T["max"]: return 0.8 - 0.4 * (kn - T["overpowered"]) / (T["max"] - T["overpowered"])
    return 0.1

def dir_score(d, spot):
    if d is None: return 0
    if in_sector(d, spot["good"]): return 1.0
    if in_sector(d, spot["ok"]): return 0.6
    return 0.0

def hour_score(kn, gust, d, spot, T):
    s = speed_score(kn, T) * dir_score(d, spot)
    if kn and gust:
        gf = (gust - kn) / max(kn, 1)
        if gf > 0.65: s *= 0.6
        elif gf > 0.45: s *= 0.8
    return round(100 * s)

def has_session(scores, need=2, thr=50):
    run = best = 0
    for s in scores:
        run = run + 1 if s >= thr else 0
        best = max(best, run)
    return best >= need

def best_window(hours, thr=50):
    """Longest run of kiteable hours; fallback: best 3h block."""
    best, cur = None, None
    for h in hours:
        if h["score"] >= thr:
            cur = [h] if cur is None else cur + [h]
            if best is None or len(cur) > len(best) or (len(cur) == len(best) and
                    sum(x["score"] for x in cur) > sum(x["score"] for x in best)):
                best = list(cur)
        else:
            cur = None
    if not best and hours:
        i = max(range(len(hours)), key=lambda i: sum(h["score"] for h in hours[i:i + 3]))
        best = hours[i:i + 3]
    return best or []

# ---------------------------------------------------------------- fetch
def fetch_models(spots):
    lat = ",".join(str(s["lat"]) for s in spots)
    lon = ",".join(str(s["lon"]) for s in spots)
    out = {}  # model -> list per spot of {time:[], kn:[], gust:[], dir:[]}
    sun = None
    for m in MODELS:
        q = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon, "models": m, "timezone": TZ,
            "hourly": "wind_speed_10m,wind_gusts_10m,wind_direction_10m",
            "daily": "sunrise,sunset", "wind_speed_unit": "ms", "forecast_days": DAYS + 1,
        })
        j = http_json("https://api.open-meteo.com/v1/forecast?" + q)
        if not j: print(f"  - {m}: no data", file=sys.stderr); continue
        j = j if isinstance(j, list) else [j]
        per = []
        for loc in j:
            h = loc["hourly"]
            per.append({"time": h["time"], "ws": h["wind_speed_10m"],
                        "gust": h.get("wind_gusts_10m"), "dir": h["wind_direction_10m"]})
        if sun is None and all("daily" in loc for loc in j):
            sun = [dict(zip(loc["daily"]["time"], zip(loc["daily"]["sunrise"], loc["daily"]["sunset"])))
                   for loc in j]
        out[m] = per
        print(f"  + {m}")
    return out, sun

def fetch_ensembles(spots):
    lat = ",".join(str(s["lat"]) for s in spots)
    lon = ",".join(str(s["lon"]) for s in spots)
    out = {}  # model -> per spot -> {time, members:[{kn:[],dir:[]}]}
    for m in ENSEMBLES:
        q = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon, "models": m, "timezone": TZ,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms", "forecast_days": DAYS + 1,
        })
        j = http_json("https://ensemble-api.open-meteo.com/v1/ensemble?" + q)
        if not j: continue
        j = j if isinstance(j, list) else [j]
        per = []
        for loc in j:
            h = loc["hourly"]
            mem = []
            for k in h:
                if k.startswith("wind_speed_10m"):
                    suf = k[len("wind_speed_10m"):]
                    dk = "wind_direction_10m" + suf
                    if dk in h: mem.append({"ws": h[k], "dir": h[dk]})
            per.append({"time": h["time"], "members": mem})
        out[m] = per
        print(f"  + ens {m}: {len(per[0]['members']) if per else 0} members")
    return out

# ---------------------------------------------------------------- weights
def load_weights(v):
    """Inverse-MAE² weights per model per lead day from verification; equal until 5+ samples."""
    sk = v.get("skill", {})
    w = {}
    for lead in range(DAYS):
        raw = {}
        for m in MODELS:
            e = sk.get(m, {}).get(str(lead))
            raw[m] = 1 / max(e["mae"], 0.8) ** 2 if e and e["n"] >= 5 else None
        known = [x for x in raw.values() if x]
        base = sum(known) / len(known) if known else 1.0
        tot = sum((x or base) for x in raw.values())
        # shrink 40% toward equal weights so one lucky week cannot take over
        w[lead] = {m: round(0.6 * (raw[m] or base) / tot + 0.4 / len(MODELS), 3) for m in MODELS}
    return w

# ---------------------------------------------------------------- build
def build(spots, T, det, sun, ens, weights, now):
    today = now.date()
    res = []
    for si, spot in enumerate(spots):
        days = []
        for d in range(DAYS):
            date = (today + dt.timedelta(days=d)).isoformat()
            sr, ss = (sun[si].get(date) if sun and si < len(sun) else None) or (date + "T08:00", date + "T16:00")
            h0, h1 = int(sr[11:13]) + (1 if int(sr[14:16]) > 30 else 0), int(ss[11:13])
            hours, member_sessions, member_w = [], [], []
            per_model_scores = {m: [] for m in det}
            for hr in range(h0, h1 + (1 if int(ss[14:16]) >= 30 else 0)):
                t = f"{date}T{hr:02d}:00"
                kns, gus, drs, ws, ms = [], [], [], [], {}
                for m, per in det.items():
                    p = per[si]
                    try: i = p["time"].index(t)
                    except ValueError: continue
                    kn, d_ = p["ws"][i], p["dir"][i]
                    if kn is None or d_ is None: continue
                    g = p["gust"][i] if p["gust"] else None
                    kns.append(kn); drs.append(d_); gus.append(g if g is not None else kn * 1.3)
                    ws.append(weights[d][m]); ms[m] = [round(kn, 1), round(d_)]
                    per_model_scores[m].append(hour_score(kn, g, d_, spot, T))
                if not kns: continue
                kn = wmean(kns, ws); g = wmean(gus, ws); dd = vec_mean(drs, ws)
                hours.append({"h": hr, "ws": round(kn, 1), "g": round(g, 1), "d": round(dd),
                              "s": hour_score(kn, g, dd, spot, T), "sp": round(wstd(kns, ws), 1), "m": ms})
            # deterministic vote (weighted share of models with a session)
            dvote_w = sum(weights[d][m] for m, sc in per_model_scores.items() if sc and has_session(sc))
            dtot = sum(weights[d][m] for m, sc in per_model_scores.items() if sc) or 1
            det_share = dvote_w / dtot
            # ensemble vote
            ens_yes = ens_n = 0
            for m, per in ens.items():
                p = per[si]
                idx = [i for i, t in enumerate(p["time"]) if t.startswith(date) and h0 <= int(t[11:13]) <= h1]
                for mem in p["members"]:
                    sc = [hour_score(mem["ws"][i], None, mem["dir"][i], spot, T) for i in idx
                          if mem["ws"][i] is not None and mem["dir"][i] is not None]
                    if sc: ens_n += 1; ens_yes += has_session(sc)
            ens_share = ens_yes / ens_n if ens_n else None
            wdet = [0.8, 0.65, 0.5, 0.35, 0.3, 0.25][d]
            chance = det_share if ens_share is None else wdet * det_share + (1 - wdet) * ens_share
            hh = [{"score": h["s"], **h} for h in hours]
            win = best_window(hh)
            peak = max(win, key=lambda h: h["ws"]) if win else None
            spread = (sum(h["sp"] for h in win) / len(win)) if win else 0
            score = round(sum(h["s"] for h in win) / len(win)) if win else 0
            conf = "høj" if spread < 1.3 and d <= 1 else "mellem" if spread < 2.3 and d <= 3 else "lav"
            verdict = "GO" if chance >= 0.6 and score >= 60 else "MÅSKE" if chance >= 0.3 and score >= 40 else "NEJ"
            days.append({
                "date": date, "sun": [sr[11:16], ss[11:16]], "chance": round(100 * chance),
                "det": round(100 * det_share), "ens": None if ens_share is None else round(100 * ens_share),
                "score": score, "verdict": verdict, "conf": conf,
                "win": [win[0]["h"], win[-1]["h"] + 1] if win else None,
                "ws": peak["ws"] if peak else None, "g": peak["g"] if peak else None,
                "d": round(vec_mean([h["d"] for h in win], [1] * len(win))) if win else None,
                "hours": hours,
            })
        res.append({k: spot[k] for k in ("id", "name", "area", "good", "ok", "water", "lat", "lon")} | {"days": days})
    return res

# ---------------------------------------------------------------- verification
def nearest_station(spot):
    j = http_json(f"{DMI_OBS}/station/items?status=Active&limit=500&bbox="
                  f"{spot['lon']-0.6},{spot['lat']-0.4},{spot['lon']+0.6},{spot['lat']+0.4}")
    best = None
    for f in (j or {}).get("features", []):
        p = f["properties"]
        if "wind_speed" not in (p.get("parameterId") or []): continue
        lo, la = f["geometry"]["coordinates"][:2]
        dist = 111 * math.hypot(la - spot["lat"], (lo - spot["lon"]) * math.cos(math.radians(la)))
        if best is None or dist < best[2]: best = (p["stationId"], p.get("name", ""), dist)
    if best: return {"id": best[0], "name": best[1], "km": round(best[2], 1)}
    sid, nm = FALLBACK_STATIONS[spot["id"]]
    return {"id": sid, "name": nm, "km": None}

def obs_hourly(station, date):
    """Hourly mean m/s, vector-mean dir, max gust m/s for a local date."""
    d0 = dt.datetime.fromisoformat(date + "T00:00:00").replace(tzinfo=local_tz())
    a = d0.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    b = (d0 + dt.timedelta(days=1)).astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    series = {}
    for par in ("wind_speed", "wind_dir", "wind_max"):
        j = http_json(f"{DMI_OBS}/observation/items?stationId={station}&parameterId={par}"
                      f"&datetime={a}/{b}&limit=10000")
        for f in (j or {}).get("features", []):
            p = f["properties"]
            t = dt.datetime.fromisoformat(p["observed"].replace("Z", "+00:00")).astimezone(local_tz())
            series.setdefault(t.hour, {}).setdefault(par, []).append(p["value"])
    out = {}
    for h, v in series.items():
        if "wind_speed" not in v: continue
        o = {"ws": round(sum(v["wind_speed"]) / len(v["wind_speed"]), 1)}
        if "wind_dir" in v: o["d"] = round(vec_mean(v["wind_dir"], [1] * len(v["wind_dir"])))
        if "wind_max" in v: o["g"] = round(max(v["wind_max"]), 1)
        out[h] = o
    return out

def verify(spots, T, now, stations):
    v = read_json(DATA / "verify.json", {"log": [], "skill": {}, "stations": {}})
    yday = (now.date() - dt.timedelta(days=1)).isoformat()
    if any(r["date"] == yday for r in v["log"]):
        return v
    print("verifying", yday)
    rows = []
    for spot in spots:
        st = stations[spot["id"]]
        obs = obs_hourly(st["id"], yday)
        if not obs: continue
        for lead in range(DAYS):
            issued = (now.date() - dt.timedelta(days=1 + lead)).isoformat()
            snap = read_json(HIST / f"fc-{issued}.json", None)
            if not snap or spot["id"] not in snap: continue
            day = snap[spot["id"]].get(yday)
            if not day: continue
            row = {"date": yday, "spot": spot["id"], "lead": lead, "models": {}}
            obs_sc, fc_sc = [], []
            for h in sorted(day["hours"], key=lambda x: x["h"]):
                o = obs.get(h["h"])
                if not o: continue
                obs_sc.append(hour_score(o["ws"], o.get("g"), o.get("d"), spot, T))
                fc_sc.append(h["s"])
                for m, (kn, d_) in h["m"].items():
                    r = row["models"].setdefault(m, {"e": 0, "de": 0, "n": 0})
                    r["e"] += abs(kn - o["ws"]); r["n"] += 1
                    if "d" in o: r["de"] += ang_diff(d_, o["d"])
            if not obs_sc: continue
            for m, r in row["models"].items():
                r["mae"] = round(r.pop("e") / r["n"], 2); r["dmae"] = round(r.pop("de") / r["n"])
            row["obs_session"] = has_session(obs_sc)
            row["fc_session"] = has_session(fc_sc)
            row["obs_peak"] = max(o["ws"] for o in obs.values())
            rows.append(row)
    v["log"] = (v["log"] + rows)[-6 * DAYS * 60:]  # ~60 days
    if not rows:  # mark the day as done anyway so we don't refetch every hour
        v["log"].append({"date": yday, "spot": None, "lead": None, "models": {}})
    # rolling 30-day skill
    cut = (now.date() - dt.timedelta(days=30)).isoformat()
    sk = {}
    for r in v["log"]:
        if r["spot"] is None or r["date"] < cut: continue
        for m, x in r["models"].items():
            s = sk.setdefault(m, {}).setdefault(str(r["lead"]), {"sum": 0, "dsum": 0, "n": 0})
            s["sum"] += x["mae"]; s["dsum"] += x["dmae"]; s["n"] += 1
    v["skill"] = {m: {l: {"mae": round(s["sum"] / s["n"], 2), "dmae": round(s["dsum"] / s["n"]), "n": s["n"]}
                      for l, s in ls.items()} for m, ls in sk.items()}
    hits = [r for r in v["log"] if r["spot"] and r["date"] >= cut and r["lead"] in (0, 1)]
    v["hit"] = {"n": len(hits),
                "correct": sum(r["obs_session"] == r["fc_session"] for r in hits),
                "missed": sum(r["obs_session"] and not r["fc_session"] for r in hits),
                "false": sum(r["fc_session"] and not r["obs_session"] for r in hits)}
    v["updated"] = now.isoformat(timespec="minutes")
    return v

# ---------------------------------------------------------------- output
def brief(res, v, now):
    DK = ["man", "tir", "ons", "tor", "fre", "lør", "søn"]
    L = [f"# Surfboys vind · {DK[now.weekday()]} {now.day}/{now.month}", ""]
    today = sorted(res, key=lambda s: -s["days"][0]["chance"] * (s["days"][0]["score"] + 1))
    go = [s for s in today if s["days"][0]["verdict"] != "NEJ"]
    L.append("## I dag")
    if not go:
        top = today[0]["days"][0]
        L.append(f"- Ingen kite i dag. Max ~{top['ws'] or 0:.0f} m/s ved {today[0]['name']}.")
    for s in go:
        d = s["days"][0]
        L.append(f"- **{s['name']}** {d['verdict']} · {d['chance']}% · kl. {d['win'][0]}–{d['win'][1]} · "
                 f"{d['ws']:.0f} m/s (stød {d['g']:.0f}) fra {compass(d['d'])} · sikkerhed {d['conf']}")
    L += ["", "## Næste 5 dage"]
    for di in range(1, DAYS):
        best = max(res, key=lambda s: s["days"][di]["chance"] * (s["days"][di]["score"] + 1))
        d = best["days"][di]
        wd = DK[dt.date.fromisoformat(d["date"]).weekday()]
        if d["verdict"] == "NEJ":
            L.append(f"- {wd}: ingen kite (bedst {best['name']} {d['chance']}%)")
        else:
            L.append(f"- {wd}: **{best['name']}** {d['chance']}% · {d['ws']:.0f} m/s {compass(d['d'])} · kl. {d['win'][0]}–{d['win'][1]}")
    if v.get("hit", {}).get("n"):
        h = v["hit"]
        L += ["", f"_Træfsikkerhed 30 dage: {round(100*h['correct']/h['n'])}% ({h['n']} spot-dage)._"]
    return "\n".join(L) + "\n"

def compass(d):
    if d is None: return "-"
    return ["N", "NNØ", "NØ", "ØNØ", "Ø", "ØSØ", "SØ", "SSØ", "S", "SSV", "SV", "VSV", "V", "VNV", "NV", "NNV"][round(d / 22.5) % 16]

def read_json(p, default):
    try: return json.loads(Path(p).read_text())
    except Exception: return default

def local_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(TZ)
    except Exception:
        return dt.timezone(dt.timedelta(hours=1))

def main():
    cfg = json.loads((ROOT / "spots.json").read_text())
    spots, T = cfg["spots"], cfg["thresholds_ms"]
    now = dt.datetime.now(local_tz())
    if os.environ.get("NOW"): now = dt.datetime.fromisoformat(os.environ["NOW"]).replace(tzinfo=local_tz())
    DATA.mkdir(exist_ok=True); HIST.mkdir(exist_ok=True)

    prev = read_json(DATA / "verify.json", {})
    stations = prev.get("stations") or {}
    for s in spots:
        if s["id"] not in stations: stations[s["id"]] = nearest_station(s)

    v = verify(spots, T, now, stations)
    v["stations"] = stations
    weights = load_weights(v)

    print("fetching models")
    det, sun = fetch_models(spots)
    if len(det) < 2:
        print("too few models, keeping previous output", file=sys.stderr); sys.exit(1)
    ens = fetch_ensembles(spots)
    res = build(spots, T, det, sun, ens, weights, now)

    latest = {"generated": now.isoformat(timespec="minutes"), "models": MODELS, "ensembles": ENSEMBLES,
              "models_ok": list(det), "ens_ok": list(ens), "weights": weights, "thresholds": T,
              "stations": stations, "spots": res}
    (DATA / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, separators=(",", ":")))
    (DATA / "verify.json").write_text(json.dumps(v, ensure_ascii=False, separators=(",", ":")))
    (DATA / "brief.md").write_text(brief(res, v, now))

    # morning snapshot used for verification later
    snapf = HIST / f"fc-{now.date().isoformat()}.json"
    if now.hour >= 8 and not snapf.exists():
        snap = {s["id"]: {d["date"]: {"hours": [{"h": h["h"], "s": h["s"], "m": h["m"]} for h in d["hours"]]}
                          for d in s["days"]} for s in res}
        snapf.write_text(json.dumps(snap, separators=(",", ":")))
    for f in HIST.glob("fc-*.json"):  # keep 90 days
        if f.stem[3:] < (now.date() - dt.timedelta(days=90)).isoformat(): f.unlink()
    print((DATA / "brief.md").read_text())

if __name__ == "__main__":
    main()
