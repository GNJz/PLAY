#!/usr/bin/env python3
"""
next_steps_pipeline.py

1) Aggregate all baseline vs DTG pairs under given run roots
2) Compute drift metrics (max, time-weighted RMS, AUC), theta_rate, first collision time
3) Save:
   - runs/summary/all_pairs_summary.csv
   - runs/summary/best_by_nu_nocollide.csv
   - runs/summary/summary.md (human-readable)
4) Recommend ν per context (nominal vs. turb) from existing results
5) Print out ready-to-run CLI for 300s validation runs using the previously used BEST/ADAPT flags

Usage
-----
python3 next_steps_pipeline.py \
  --roots runs/micro_* runs/best_nu_* runs/turb_test runs/reprod_best_* \
  --cand_nu 0.0025 0.003 0.0035 0.0036

You can safely add/remove roots; the script will skip missing ones.
"""
import argparse, glob, os, json
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

# ---------- metrics ----------

def _safe_read(path: str, usecols: List[str]):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, usecols=[c for c in usecols if c != "*" and c != "?"])
    except Exception:
        return None

def drift_metrics(csv_path: str) -> Dict[str, float]:
    df = _safe_read(csv_path, ["t", "drift"])  # diagnostics
    if df is None or df.empty:
        return dict(max=0.0, rms=0.0, auc=0.0)
    t = df["t"].to_numpy()
    d = np.abs(df["drift"].to_numpy())
    if len(t) < 2:
        return dict(max=float(d.max() if len(d) else 0.0), rms=0.0, auc=0.0)
    w = np.diff(t)
    rms = float(np.sqrt(((d[:-1]**2) * w).sum() / max(w.sum(), 1e-12)))
    auc = float(((d[:-1]) * w).sum() / max(t[-1] - t[0], 1e-12))
    return dict(max=float(d.max()), rms=rms, auc=auc)


def theta_rate(csv_path: str) -> float:
    df = _safe_read(csv_path, ["t", "theta"])  # main CSV also has theta
    if df is None or len(df) < 2:
        return 0.0
    dt = np.diff(df["t"].to_numpy())
    dth = np.abs(np.diff(df["theta"].to_numpy()))
    mdt = float(np.mean(dt)) if len(dt) else 0.0
    return float(np.mean(dth) / max(mdt, 1e-12)) if mdt > 0 else 0.0


def first_collision_time(diag_csv: str, thresh: float = 0.05) -> Optional[float]:
    df = _safe_read(diag_csv, ["t", "rmin"])  # diagnostics has rmin
    if df is None or df.empty:
        return None
    u = df[df["rmin"] <= thresh]
    if len(u):
        return float(u["t"].iloc[0])
    return None

# ---------- scan & pair ----------

def find_pairs(roots: List[str]):
    rows = []
    for root_glob in roots:
        for base_dir in glob.glob(root_glob):
            # Expect subdirs baseline_nu*/dtg_nu*/
            for kind in ["baseline", "dtg"]:
                for d in glob.glob(os.path.join(base_dir, f"{kind}_nu*")):
                    tag = os.path.basename(base_dir)
                    kind_tag = os.path.basename(d)
                    # parse nu
                    try:
                        nu = float(kind_tag.split("nu")[-1])
                    except Exception:
                        continue
                    data_dir = os.path.join(d, "data")
                    diag = None
                    csv_candidates = []
                    # diagnostics file
                    for f in glob.glob(os.path.join(data_dir, "diagnostics_*.csv")):
                        diag = f
                        break
                    # main csv (for theta)
                    for f in glob.glob(os.path.join(data_dir, "threebody3d_*.csv")):
                        csv_candidates.append(f)
                    csv_main = csv_candidates[0] if csv_candidates else None

                    rows.append(dict(
                        group_tag=tag,
                        nu=nu,
                        kind=kind,
                        diag=diag,
                        csv_main=csv_main,
                    ))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    # pair baseline vs dtg inside (group_tag, nu)
    out = []
    for (g, nu), sub in df.groupby(["group_tag", "nu" ]):
        sub = sub.set_index("kind")
        if "baseline" not in sub.index or "dtg" not in sub.index:
            continue
        b = sub.loc["baseline"]
        d = sub.loc["dtg"]
        if not (isinstance(b, pd.Series) and isinstance(d, pd.Series)):
            continue
        # metrics
        mb = drift_metrics(b["diag"]) if pd.notna(b["diag"]) else dict(max=0.0, rms=0.0, auc=0.0)
        md = drift_metrics(d["diag"]) if pd.notna(d["diag"]) else dict(max=0.0, rms=0.0, auc=0.0)
        thr = theta_rate(d["csv_main"]) if pd.notna(d["csv_main"]) else 0.0
        tc = first_collision_time(d["diag"]) if pd.notna(d["diag"]) else None
        out.append(dict(
            group_tag=g,
            nu=nu,
            drift_rms_base=mb["rms"],
            drift_rms_dtg=md["rms"],
            drift_max_base=mb["max"],
            drift_max_dtg=md["max"],
            improve_pct=100.0 * (mb["rms"] - md["rms"]) / max(mb["rms"], 1e-12),
            theta_rate_dtg=thr,
            collided_dtg = bool(tc is not None),
            t_collision_dtg = tc,
            diag_base=b["diag"], diag_dtg=d["diag"],
        ))
    return pd.DataFrame(out)

# ---------- recommendation ----------

def recommend_nu(df: pd.DataFrame, cand: List[float]) -> Optional[float]:
    if df.empty:
        return None
    ok = df[(~df["collided_dtg"]) & (df["nu"].isin(cand))].copy()
    if ok.empty:
        return None
    ok = ok.sort_values(["improve_pct", "drift_rms_dtg"], ascending=[False, True])
    return float(ok.iloc[0]["nu"])

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "runs/micro_*", "runs/best_nu_*", "runs/turb_test", "runs/reprod_*", "runs/grid_*", "runs/ck0.03_cmp_*"
    ])
    ap.add_argument("--cand_nu", nargs="+", type=float, default=[0.0025, 0.003, 0.0035, 0.0036])
    ap.add_argument("--summary_dir", default="runs/summary")
    args = ap.parse_args()

    os.makedirs(args.summary_dir, exist_ok=True)

    df = find_pairs(args.roots)
    all_csv = os.path.join(args.summary_dir, "all_pairs_summary.csv")
    df.to_csv(all_csv, index=False)

    # best per nu without collision
    ok = df[~df["collided_dtg"]].copy()
    best = pd.DataFrame()
    if not ok.empty:
        best = (ok.sort_values(["nu", "improve_pct"], ascending=[True, False])
                   .groupby("nu", as_index=False).head(1))
        best_csv = os.path.join(args.summary_dir, "best_by_nu_nocollide.csv")
        best.to_csv(best_csv, index=False)
    else:
        best_csv = None

    # two contexts: nominal vs turb (heuristic: look for group_tag containing 'turb')
    nominal_df = df[~df["group_tag"].str.contains("turb", na=False)]
    turb_df    = df[df["group_tag"].str.contains("turb", na=False)]
    rec_nominal = recommend_nu(nominal_df, args.cand_nu)
    rec_turb    = recommend_nu(turb_df,    args.cand_nu)

    # write markdown summary
    md = ["# Auto Summary\n"]
    md.append(f"- Saved: `{all_csv}`" )
    if best_csv:
        md.append(f"- Saved: `{best_csv}`")
    md.append("")
    if rec_nominal is not None:
        md.append(f"**Recommended ν (nominal)**: **{rec_nominal:.4f}**")
    if rec_turb is not None:
        md.append(f"**Recommended ν (turb)**: **{rec_turb:.4f}**")

    if not best.empty:
        md.append("\n## Best per ν (no collision)\n")
        md.append(best[["nu", "group_tag", "improve_pct", "drift_rms_base", "drift_rms_dtg", "theta_rate_dtg"]]
                    .to_markdown(index=False))
    with open(os.path.join(args.summary_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # print ready-to-run 300s commands
    print("\n=== Ready-to-run 300s validation (copy/paste) ===\n")
    print("# Assumes your BEST/ADAPT arrays are set in the shell, as in previous runs.\n")
    for label, rec in [("nominal", rec_nominal), ("turb", rec_turb)]:
        if rec is None:
            continue
        out = f"runs/validate_{label}_nu{rec:.4f}"
        turb_flag = " --turb 1e-3" if label == "turb" else ""
        cmd = (
            "python3 three_body_3d_ns_dtg_v3_5_full_FIXED.py \\\n  --compare --nu_scan {nu:.4f} \\\n  --tmax 300 --dt 0.0005 --out {out} --no_plots{turb} \\\n  --adaptive_step --adapt_r_trigger 0.03 --adapt_factor 0.5 --min_max_step 1e-4 \\\n  --seed 42 \"${{BEST[@]}}\"".format(nu=rec, out=out, turb=turb_flag)
        )
        print(f"# {label}\n{cmd}\n")

    print("Done.")

if __name__ == "__main__":
    main()
