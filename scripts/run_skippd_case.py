"""Reproduce the SKIPP'D forecast-driven dispatch and frequency case; NumPy only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DAY = np.datetime64("2017-09-20")
DT, NT, F0 = 1.0, 240, 50.0
WIN0_H, CLIFF_H = 12.0, 13.9667
T_CLIFF = int(round((CLIFF_H - WIN0_H) * 60))
PMAX = np.array([80.0, 80.0, 50.0, 55.0, 30.0, 40.0])
PMIN = 0.30 * PMAX
C2 = np.array([0.02000, 0.01750, 0.06250, 0.00834, 0.02500, 0.02500])
C1 = np.array([2.00, 1.75, 1.00, 3.25, 3.00, 3.00])
NG = len(PMAX)
RAMP = 0.02 * PMAX
RES10 = float((10 * RAMP).sum())
PV_CAP, KW2MW = 200.0, 200.0 / 30.1
BESS_P, BESS_E = 30.0, 120.0
SOC0, SOC_LO, SOC_HI = 0.55, 0.10, 0.90
ETA_C = ETA_D = 0.95
VOLL, C_CURT, C_DEG = 1000.0, 50.0, 40.0
LOAD_PEAK = 283.4
HOURLY_LOAD = np.array([
    .72, .70, .68, .67, .68, .71, .76, .82, .86, .88, .87, .85,
    .84, .85, .87, .90, .94, .98, 1.00, .99, .96, .92, .85, .78,
])
HOURS = WIN0_H + np.arange(NT) * DT / 60
TRIG_KW = 0.30 * 30.1
S3FLAGS = dict(charge=True, preramp=True, fastctrl=True, reserve=True)


def load_at(hour):
    return np.interp(hour, np.arange(24), HOURLY_LOAD) * LOAD_PEAK


class CaseData:
    """Read aligned existing assets without training, downloading, or writing."""

    def __init__(self, root):
        root = Path(root)
        with np.load(root / "data/proc/test_meta.npz", allow_pickle=False) as meta:
            self.minutes = meta["minutes"].astype("datetime64[s]")
            self.power = meta["pv"].astype(float)
        with np.load(root / "data/proc/test_samples.npz", allow_pickle=False) as samples:
            anchors = samples["anchor_time"].astype("datetime64[s]")
            win, tgt = samples["win"], samples["tgt"]
        if not np.all(np.diff(self.minutes[win], axis=1) == np.timedelta64(60, "s")):
            raise ValueError("Non-contiguous input windows")
        if not np.array_equal(self.minutes[win[:, -1]], anchors):
            raise ValueError("Issue-time alignment mismatch")
        if not np.array_equal(self.minutes[tgt], anchors + np.timedelta64(15, "m")):
            raise ValueError("Forecast horizon must be 15 minutes")
        select = anchors.astype("datetime64[D]") == DAY
        self.issue_hours = (anchors[select] - DAY) / np.timedelta64(1, "h")
        order = np.argsort(self.issue_hours)
        self.issue_hours = self.issue_hours[order]
        issue_seconds = (anchors[select][order] - DAY) / np.timedelta64(1, "s")
        if len(issue_seconds) < 2 or not np.all(np.diff(issue_seconds) == 60):
            raise ValueError("Missing event-day prediction periods")
        self.target_hours = self.issue_hours + .25
        self.predictions = {}
        for name, stem in [("PGF-TPA", "pgf_nocsi_tpa"),
                           ("Sky-BiMamba", "skybimamba"), ("SUNSET", "sunset")]:
            path = root / f"results/preds_TABLE3_{stem}_15.npy"
            values = np.load(path, allow_pickle=False)
            if values.shape != anchors.shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid predictions: {path.name}")
            self.predictions[name] = values[select][order].astype(float)
        event = self.minutes.astype("datetime64[D]") == DAY
        self.day_hours = (self.minutes[event] - DAY) / np.timedelta64(1, "h")
        self.day_power = self.power[event]
        if (len(self.day_hours) < 2 or not np.isfinite(self.day_power).all()
                or np.any(np.diff(self.day_hours) <= 0)):
            raise ValueError("Invalid event-day measurements")
        if self.day_hours[0] > WIN0_H - 3.25 or self.day_hours[-1] < 16:
            raise ValueError("Event-day measurements do not cover the required interval")
        self.pv = np.array([self.obs(h) for h in HOURS]) * KW2MW
        self.load = load_at(HOURS)
        self.frequency_seconds = np.arange(144000) * .05
        self.frequency_hours = WIN0_H + T_CLIFF / 60 + self.frequency_seconds / 3600
        self.frequency_pv = np.array([self.obs(h) for h in self.frequency_hours]) * KW2MW
        self.frequency_load = load_at(self.frequency_hours)
        # Fixed calibration from the original numerical case, shared by all forecasts.
        self.margin = 175.0 - (load_at(14.0167 + .0167)
                              - self.forecast("PGF-TPA", 14.0167) * KW2MW)

    def obs(self, hour):
        # Preserve whole-second quantization of the original observation lookup.
        t = DAY + np.timedelta64(int(round(hour * 3600)), "s")
        j = min(max(int(np.searchsorted(self.minutes, t)), 1), len(self.minutes) - 1)
        gap = (self.minutes[j] - self.minutes[j - 1]) / np.timedelta64(1, "s")
        w = 0.0 if gap == 0 else (t - self.minutes[j - 1]) / np.timedelta64(1, "s") / gap
        return float(self.power[j - 1] * (1 - w) + self.power[j] * w)

    def forecast(self, method, target_hour):
        if method == "Persistence":
            return self.obs(target_hour - .25)
        if method == "3h-MA":
            issue = target_hour - .25
            mask = (self.day_hours >= issue - 3) & (self.day_hours <= issue)
            if not mask.any():
                raise ValueError("No past measurements for moving-average baseline")
            return float(self.day_power[mask].mean())
        if method == "PerfectForesight":
            return self.obs(target_hour)
        if not self.target_hours[0] <= target_hour <= self.target_hours[-1]:
            raise ValueError("Requested forecast is outside the saved target interval")
        return float(np.interp(target_hour, self.target_hours, self.predictions[method]))

    def alarm_chain(self, method):
        rows = []
        # Preserve the original scan: issues 12:00..15:45, targets 12:15..16:00.
        hour = WIN0_H
        while hour + 1e-9 <= 16.0 - .25:
            pred = self.forecast(method, hour + .25)
            obs = self.obs(hour)
            rows.append(dict(method=method, issue_hour=hour, target_hour=hour + .25,
                             observed_kw=obs, forecast_kw=pred,
                             triggered=bool(obs - pred >= TRIG_KW)))
            hour += 1 / 60
        return rows

    def first_alarm(self, method):
        return next((r["issue_hour"] for r in self.alarm_chain(method) if r["triggered"]), None)

    def pre_level(self, method):
        return float(load_at(14.0167 + .0167)
                     - self.forecast(method, 14.0167) * KW2MW + self.margin)


def lowpass(values, tau):
    """Preserve the archived filter alpha=1/(60*tau); see the experiment README."""
    out = values.copy()
    alpha = DT / 60 / tau
    for k in range(1, len(out)):
        out[k] = out[k - 1] + alpha * (values[k] - out[k - 1])
    return out


def ed_dispatch(target, p_prev):
    lo = np.maximum(PMIN, p_prev - RAMP*DT)
    hi = np.minimum(PMAX, p_prev + RAMP*DT)
    p = np.clip(p_prev, lo, hi)
    gap = target - p.sum()
    for _ in range(400):
        if abs(gap) < 1e-6:
            break
        if gap > 0:
            cand = np.where(p < hi - 1e-9)[0]
            if cand.size == 0: break
            i = cand[np.argmin(C1[cand] + 2*C2[cand]*p[cand])]
            s = min(gap, hi[i] - p[i]); p[i] += s; gap -= s
        else:
            cand = np.where(p > lo + 1e-9)[0]
            if cand.size == 0: break
            i = cand[np.argmax(C1[cand] + 2*C2[cand]*p[cand])]
            s = max(gap, lo[i] - p[i]); p[i] += s; gap -= s
    return p


def ed_alloc(target):
    p = PMIN.copy(); rem = max(target - p.sum(), 0.0)
    for i in np.argsort(C1 + 2*C2*p):
        s = min(rem, PMAX[i] - p[i]); p[i] += s; rem -= s
    return p


def simulate(data, has_bess, flags, t_alarm, p_pre, PV):
    """Minute-resolution production balance with the archived intervention flags."""
    LOAD = data.load
    p_th = ed_alloc(max(LOAD[0] - PV[0], PMIN.sum()))
    soc = SOC0
    T_ALARM = int(round((t_alarm - WIN0_H) * 60)) if t_alarm is not None else -1
    Pth_u = np.zeros((NT, NG)); Pb = np.zeros(NT)
    Pv_used = np.zeros(NT); Pshed = np.zeros(NT); Pcurt = np.zeros(NT); SOC = np.zeros(NT)
    for t in range(NT):
        load, pv = LOAD[t], PV[t]
        pb = 0.0
        if flags.get("charge") and T_ALARM <= t <= T_CLIFF and soc < SOC_HI:
            pb = -BESS_P
        if has_bess and pb < 0:
            pb = -min(-pb, BESS_P, (SOC_HI - soc)*BESS_E/(ETA_C*DT/60))
        elif has_bess and pb > 0:
            pb = min(pb, BESS_P, (soc - SOC_LO)*BESS_E*ETA_D/(DT/60))
        pre_extra = 0.0
        if flags.get("preramp") and T_CLIFF - 15 <= t <= T_CLIFF + 10:
            pre_extra = max(0.0, p_pre - (load - pv - pb))
        p_new = ed_dispatch(max(load - pv - pb + pre_extra, PMIN.sum()), p_th)
        surplus = p_new.sum() + pv + pb - load
        if surplus > 1e-6 and has_bess:
            room_p = (BESS_P + pb) if pb < 0 else BESS_P
            room_e = (SOC_HI - soc)*BESS_E/(ETA_C*DT/60)
            add = max(0.0, min(surplus, room_p, room_e))
            if add > 0:
                pb -= add
                p_new = ed_dispatch(max(load - pv - pb + pre_extra, PMIN.sum()), p_th)
                surplus = p_new.sum() + pv + pb - load
        curt = max(0.0, surplus)
        pv_used = pv - curt
        shed = 0.0
        if surplus <= 1e-6:
            deficit = -(p_new.sum() + pv + pb - load)
            if deficit > 1e-6 and has_bess:
                room_p = BESS_P - pb
                room_e = (soc - SOC_LO)*BESS_E*ETA_D/(DT/60)
                dis = max(0.0, min(deficit, room_p, room_e))
                pb += dis; deficit -= dis
            shed = deficit
        if has_bess and pb < 0:
            soc += -pb*ETA_C*DT/60/BESS_E
        elif has_bess and pb > 0:
            soc -= pb/ETA_D*DT/60/BESS_E
        soc = min(max(soc, SOC_LO), SOC_HI)
        p_th = p_new
        Pth_u[t] = p_new; Pb[t] = pb; Pv_used[t] = pv_used
        Pshed[t] = shed; Pcurt[t] = curt; SOC[t] = soc
    fuel = float(sum((C2[i]*Pth_u[:, i]**2 + C1[i]*Pth_u[:, i]).sum()
                     for i in range(NG))*DT/60)
    e_shed, e_curt = float(Pshed.sum()*DT/60), float(Pcurt.sum()*DT/60)
    e_avail, e_used = float(PV.sum()*DT/60), float(Pv_used.sum()*DT/60)
    thr = float(np.abs(Pb).sum()*DT/60)
    return dict(Pth=Pth_u.sum(axis=1), Pth_u=Pth_u, Pb=Pb, Pv=PV,
                Pv_used=Pv_used, Pshed=Pshed, Pcurt=Pcurt, SOC=SOC,
                fuel=fuel, e_shed=e_shed, e_curt=e_curt, thr=thr,
                cost=fuel + e_shed*VOLL + e_curt*C_CURT + thr*C_DEG,
                util=e_used/e_avail*100, shed_max=float(Pshed.max()))


def freq_sim(data, res, has_bess, flags, PV_kw_smooth=None):
    dt = 0.05
    FS0 = T_CLIFF
    FS1 = min(T_CLIFF + 120, NT)
    n = int((FS1 - FS0)*DT*60.0/dt)
    tg = np.arange(n)*dt
    pv_forcing = data.frequency_pv if PV_kw_smooth is None else (
        np.interp(data.frequency_hours, HOURS, PV_kw_smooth) * KW2MW)
    load_forcing = data.frequency_load
    p0 = float(res["Pth"][FS0])
    pb0 = float(res["Pb"][FS0])
    pv_c0 = float(res["Pv"][FS0] - res["Pv_used"][FS0])
    S_eq, H_sys = 350.0, 4.0
    KM = 2*H_sys*S_eq/F0
    D_MW = LOAD_PEAK/F0
    G_gov = PMAX.sum()/(0.05*F0)
    res_allo = min(PMAX.sum() - p0, 100.0 if flags.get("reserve") else RES10)
    Fhp, Tg, Tr = 0.28, 0.25, 8.0
    fast = flags.get("fastctrl")
    agc_dly = 5.0 if fast else 30.0
    bess_dly = 0.0 if fast else 30.0
    ufls = [(49.0, 0.05, 0.0, False), (48.75, 0.05, 0.0, False),
            (48.5, 0.10, 0.0, False)]
    PICKUP = 0.2
    shed_mw, shed_e = 0.0, 0.0
    f = np.full(n, F0)
    rampsum = float(RAMP.sum())
    xg = xt = agc = 0.0
    pb = pb0
    pv_curt = pv_c0
    rocof = 0.0
    for k in range(1, n):
        sec = float(tg[k])
        load, pv = float(load_forcing[k]), float(pv_forcing[k])
        if has_bess:
            gap = load - pv - p0 - pb - shed_mw + pv_curt
            if sec >= bess_dly and gap > 0:
                pb_t = min(max(gap, 0.0), BESS_P)
            else:
                pb_t = pb0
            r10 = BESS_P/10.0
            pb += min(max(pb_t - pb, -r10*dt), r10*dt)
            pb = min(max(pb, -BESS_P), BESS_P)
        else:
            pb = 0.0
        df = float(f[k-1]) - F0
        xg += dt*((-df/(0.05*F0))*G_gov - xg)/Tg
        p_pri = Fhp*xg + (1-Fhp)*xt
        xt += dt*(xg - xt)/Tr
        p_pri = min(max(p_pri, -0.3*res_allo), res_allo)
        if sec >= agc_dly:
            gap = load - pv - (p0 + p_pri + agc) - pb - shed_mw + pv_curt
            agc += min(max(gap, -rampsum*dt/60), rampsum*dt/60)
            if abs(df) < 0.05 and agc > 0.0:
                agc = max(0.0, agc - 2.0*dt/60)
            agc = min(max(agc, -res_allo), res_allo)
        p_mech = p0 + p_pri + agc
        excess = p_mech + pb - (load - pv - shed_mw) + D_MW*df
        tgt_c = max(excess, 0.0) + max(0.0, df - 0.05)*80.0
        rate_c = 100.0 if tgt_c > pv_curt else 30.0
        pv_curt += min(max(tgt_c - pv_curt, -rate_c*dt/60), rate_c*dt/60)
        pv_curt = float(min(max(pv_curt, 0.0), pv))
        pnet = p_mech + pb - (load - pv + pv_curt - shed_mw) - D_MW*df
        dfdt = pnet/KM
        f[k] = f[k-1] + dfdt*dt
        rocof = min(rocof, dfdt)
        for i, (fth, share, timer, done) in enumerate(ufls):
            if done: continue
            timer = timer + dt if f[k] < fth else 0.0
            if timer >= PICKUP:
                ufls[i] = (fth, share, timer, True)
                shed_mw += share*LOAD_PEAK
            else:
                ufls[i] = (fth, share, timer, done)
        shed_e += shed_mw*dt/3600.0
    return dict(fnadir=float(f.min()), fmax=float(f.max()), rocof=rocof,
                fired=[u[3] for u in ufls], shed_mw=shed_mw, shed_e=shed_e)


def metrics(name, res, fr):
    return dict(name=name, curt=round(res["e_curt"], 2), shed=round(res["e_shed"], 2),
                cost=round(res["cost"]), util=round(res["util"], 2),
                fmin=round(fr["fnadir"], 3),
                ufls=[i+1 for i, v in enumerate(fr["fired"]) if v])


def validate_dispatch(data, res):
    if not all(np.isfinite(res[k]).all() for k in
               ("Pth_u", "Pb", "Pv", "Pv_used", "Pshed", "Pcurt", "SOC")):
        raise ValueError("Non-finite dispatch results")
    balance = res["Pth"] + res["Pb"] + res["Pv_used"] + res["Pshed"]
    if not np.allclose(balance, data.load, atol=2e-6, rtol=0):
        raise ValueError("Dispatch does not balance supply and served load")
    if np.any(res["SOC"] < SOC_LO - 1e-9) or np.any(res["SOC"] > SOC_HI + 1e-9):
        raise ValueError("SOC outside case limits")
    if np.any(np.abs(res["Pb"]) > BESS_P + 1e-6):
        raise ValueError("Battery power outside case limits")
    if np.any(res["Pth_u"] < PMIN - 1e-6) or np.any(res["Pth_u"] > PMAX + 1e-6):
        raise ValueError("Thermal power outside case limits")


def run_experiment(data, suite="full", progress=None):
    out = dict(alarm_pgf=data.first_alarm("PGF-TPA"),
               p_pre_pgf=data.pre_level("PGF-TPA"), margin=data.margin,
               runs={}, smooth={})
    traces = []

    def run(name, has_bess, flags, alarm, pre, pv=None, smooth=False):
        if progress:
            progress(name)
        pv = data.pv if pv is None else pv
        res = simulate(data, has_bess, flags, alarm, pre, pv)
        validate_dispatch(data, res)
        fr = freq_sim(data, res, has_bess, flags, pv / KW2MW if smooth else None)
        row = metrics(name, res, fr)
        row.update(frequency_max_hz=float(fr["fmax"]), rocof_hz_per_s=float(fr["rocof"]),
                   ufls_shed_mw=float(fr["shed_mw"]), ufls_shed_mwh=float(fr["shed_e"]))
        if not smooth:
            row.update(t_alarm=round(alarm, 4) if alarm is not None else None,
                       p_pre=round(pre, 1) if pre else None)
        out["smooth" if smooth else "runs"][name] = row
        for i, hour in enumerate(HOURS):
            traces.append(dict(scenario=name, local_hour=float(hour),
                               load_mw=float(data.load[i]), pv_available_mw=float(pv[i]),
                               thermal_mw=float(res["Pth"][i]), battery_mw=float(res["Pb"][i]),
                               pv_used_mw=float(res["Pv_used"][i]), shed_mw=float(res["Pshed"][i]),
                               curtailed_mw=float(res["Pcurt"][i]), soc=float(res["SOC"][i])))

    alarm, pre = out["alarm_pgf"], out["p_pre_pgf"]
    run("S1", False, {}, None, 0.0)
    run("S2", True, {}, None, 0.0)
    run("S3", True, S3FLAGS, alarm, pre)
    if suite == "full":
        for flag in S3FLAGS:
            run(f"S3-{flag}", True, dict(S3FLAGS, **{flag: False}), alarm, pre)
        run("S2+charge", True, dict(charge=True), alarm, 0.0)
        run("S2+preramp", True, dict(preramp=True), None, pre)
        run("S2+ctrl+res", True, dict(fastctrl=True, reserve=True), None, 0.0)
        for method in ("Sky-BiMamba", "SUNSET", "Persistence", "3h-MA"):
            ta = data.first_alarm(method)
            if ta is None:
                out["runs"][f"S3@{method}"] = dict(
                    out["runs"]["S2"], name=f"S3@{method}",
                    note="never triggers under the rule; reduces to S2")
            else:
                run(f"S3@{method}", True, S3FLAGS, ta, data.pre_level(method))
        run("S3orig-validate", True, S3FLAGS, 13 + 2 / 60, 175.0)
        out["pf_trigger"] = data.first_alarm("PerfectForesight")
        run("PerfectForesight", True, S3FLAGS, out["pf_trigger"], data.pre_level("PerfectForesight"))
        for tau in (1, 3, 5, 10):
            pv = lowpass(data.pv, tau)
            for name, hb, flags, ta, pp in [("S1", False, {}, None, 0.0),
                                           ("S2", True, {}, None, 0.0),
                                           ("S3", True, S3FLAGS, alarm, pre)]:
                run(f"tau{tau}_{name}", hb, flags, ta, pp, pv=pv, smooth=True)
    return out, traces


def check_reference(actual, expected, suite="full"):
    """Check every reproduced field against the independent frozen numeric record."""
    for key in ("alarm_pgf", "p_pre_pgf", "margin", "pf_trigger"):
        if key in actual and not np.isclose(actual[key], expected[key], rtol=0, atol=1e-8):
            raise AssertionError(f"Reference mismatch: {key}")
    for group in ("runs", "smooth"):
        names = set(expected[group]) if suite == "full" else (
            {"S1", "S2", "S3"} if group == "runs" else set())
        if set(actual[group]) != names:
            raise AssertionError(f"Scenario coverage mismatch: {group}")
        for name, row in actual[group].items():
            ref = expected[group][name]
            for key, value in ref.items():
                observed = row.get(key)
                if isinstance(value, (int, float)):
                    equal = observed is not None and np.isclose(observed, value, rtol=0, atol=1e-8)
                else:
                    equal = observed == value
                if not equal:
                    raise AssertionError(f"Reference mismatch: {group}/{name}/{key}: {observed} != {value}")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository containing the existing assets")
    parser.add_argument("--suite", choices=("main", "full"), default="full")
    parser.add_argument("--output", type=Path, help="Default: <root>/outputs/skippd_case")
    parser.add_argument("--reference", type=Path, help="Default: <root>/experiments/skippd_case/reference.json")
    parser.add_argument("--check", action="store_true", help="Compare results with the frozen numeric reference")
    args = parser.parse_args()
    reference_path = args.reference or args.root / "experiments/skippd_case/reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    for relative, digest in reference["inputs"].items():
        path = args.root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input mismatch: {relative}. Run git lfs pull and verify release assets.")
    data = CaseData(args.root)
    actual, traces = run_experiment(data, args.suite, lambda name: print(f"Running {name}", flush=True))
    if args.check:
        check_reference(actual, reference["expected"], args.suite)
        print("PASS: all selected scenarios match the frozen numerical reference", flush=True)
    actual["metadata"] = dict(event_date=str(DAY), suite=args.suite, horizon_minutes=15,
                              scan_issue_hours=[12, 15.75], inputs=reference["inputs"],
                              filter_alpha="1/(60*tau)", frequency_step_seconds=.05)
    output = args.output or args.root / "outputs/skippd_case"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(actual, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = []
    for row in list(actual["runs"].values()) + list(actual["smooth"].values()):
        rows.append(dict(scenario=row["name"], curtailment_mwh=row["curt"],
                         unserved_energy_mwh=row["shed"], total_cost=row["cost"],
                         utilization_percent=row["util"], frequency_min_hz=row["fmin"],
                         alarm_hour=row.get("t_alarm"), thermal_target_mw=row.get("p_pre"),
                         ufls_stages=";".join(map(str, row["ufls"]))))
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "dispatch.csv", traces)
    chain = [row for method in ("PGF-TPA", "Sky-BiMamba", "SUNSET", "Persistence", "3h-MA")
             for row in data.alarm_chain(method)]
    write_csv(output / "alarm_chain.csv", chain)
    print(f"Wrote {len(rows)} scenarios to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
