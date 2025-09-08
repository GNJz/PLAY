#!/usr/bin/env python3
import glob, os
import numpy as np
import pandas as pd

def drift_metrics(diag_csv: str):
    df = pd.read_csv(diag_csv, usecols=["t","drift"])
    t = df["t"].to_numpy(); d = np.abs(df["drift"].to_numpy())
    if len(t) < 2:
        return dict(rms=0.0, mx=float(d.max() if len(d) else 0.0))
    w = np.diff(t)
    rms = float(np.sqrt(((d[:-1]**2) * w).sum() / max(w.sum(), 1e-12)))
    return dict(rms=rms, mx=float(d.max()))

def theta_rate(main_csv: str) -> float:
    df = pd.read_csv(main_csv, usecols=["t","theta"])
    dt = np.diff(df["t"].to_numpy()); dth = np.abs(np.diff(df["theta"].to_numpy()))
    return float(np.mean(dth) / max(np.mean(dt), 1e-12)) if len(dt) else 0.0

def first_collision(diag_csv: str, th: float = 0.05):
    df = pd.read_csv(diag_csv, usecols=["t","rmin"])
    u = df[df["rmin"] <= th]
    return float(u["t"].iloc[0]) if len(u) else None

def collect(root: str, nu_str: str):
    """return (run, nu, mb, md, tr, tc) or None if any file missing"""
    b_diag = glob.glob(f"{root}/baseline_nu{nu_str}/data/diagnostics_*.csv")
    d_diag = glob.glob(f"{root}/dtg_nu{nu_str}/data/diagnostics_*.csv")
    d_main = glob.glob(f"{root}/dtg_nu{nu_str}/data/threebody3d_*.csv")
    if not (b_diag and d_diag and d_main):
        return None
    mb, md = drift_metrics(b_diag[0]), drift_metrics(d_diag[0])
    tr = theta_rate(d_main[0])
    tc = first_collision(d_diag[0])
    return os.path.basename(root), float(nu_str), mb, md, tr, tc

def main():
    roots = [
        "runs/validate_nominal_nu0.0035",
        "runs/validate_turb_nu0.0035",
        "runs/endurance_nominal_nu0.0035",
    ]
    rows = []
    for root in roots:
        for nu in ["0.003","0.0035"]:
            got = collect(root, nu)
            if got is None:
                continue
            run, nu_f, mb, md, tr, tc = got
            imp = 100.0 * (mb["rms"] - md["rms"]) / max(mb["rms"], 1e-12)
            rows.append(dict(
                run=run, nu=nu_f,
                drift_rms_base=mb["rms"], drift_rms_dtg=md["rms"],
                improve_pct=imp, theta_rate=tr, t_collision=tc
            ))
    df = pd.DataFrame(rows).sort_values(["run","nu"])
    if df.empty:
        print("No rows found. Check paths under runs/*")
        return
    print(df.to_string(index=False, formatters={"improve_pct": lambda v: f"{v:6.2f}%"}))
    os.makedirs("runs/summary", exist_ok=True)
    out_csv = "runs/summary/validate_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()