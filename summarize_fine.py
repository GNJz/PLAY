import os, glob, re, math
import numpy as np
import pandas as pd

os.makedirs("runs/summary", exist_ok=True)

def _first_or_none(lst):
    return lst[0] if lst else None

def _read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def drift_metrics(diag_csv):
    """
    진단 CSV에서 드리프트 RMS를 최대한 robust하게 추출/계산.
    우선순위:
    1) drift_rms / rms_drift / driftRMS 같은 단일 컬럼
    2) 'drift' 시계열 → sqrt(mean(drift^2))
    3) drift_x, drift_y, (drift_z) 합성 → sqrt(mean(dx^2+dy^2(+dz^2)))
    못 찾으면 None 반환
    """
    df = _read_csv(diag_csv)
    if df is None or df.empty:
        return None

    # 1) 다양한 네이밍 후보
    for c in df.columns:
        cl = c.lower()
        if cl in ("drift_rms", "rms_drift", "drift_rms_total", "drms"):
            try:
                v = float(df[c].dropna().iloc[-1])
                return {"rms": v}
            except Exception:
                pass

    # 2) 'drift' 시계열의 RMS
    for c in df.columns:
        if c.lower() == "drift":
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                return {"rms": float(np.sqrt((s**2).mean()))}

    # 3) 벡터 합성 RMS
    dx = dy = dz = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("drift_x","dx","err_x","ex"): dx = pd.to_numeric(df[c], errors="coerce")
        if cl in ("drift_y","dy","err_y","ey"): dy = pd.to_numeric(df[c], errors="coerce")
        if cl in ("drift_z","dz","err_z","ez"): dz = pd.to_numeric(df[c], errors="coerce")
    if dx is not None and dy is not None:
        n = min(len(dx.dropna()), len(dy.dropna()), len(dz.dropna()) if dz is not None else 10**9)
        if n > 0:
            dx2 = dx.dropna().iloc[:n]**2
            dy2 = dy.dropna().iloc[:n]**2
            if dz is not None:
                dz2 = dz.dropna().iloc[:n]**2
                rms = float(np.sqrt((dx2+dy2+dz2).mean()))
            else:
                rms = float(np.sqrt((dx2+dy2).mean()))
            return {"rms": rms}

    return None

def theta_rate(main_csv):
    """
    main CSV에서 평균 |d theta/dt| 추정. (없으면 None)
    """
    df = _read_csv(main_csv)
    if df is None or df.empty: return None
    # 후보 컬럼 이름
    tcol = _first_or_none([c for c in df.columns if c.lower() in ("t","time")])
    thcol = _first_or_none([c for c in df.columns if c.lower() in ("theta","th","theta_gate","gate_theta")])
    if not tcol or not thcol: return None
    t = pd.to_numeric(df[tcol], errors="coerce")
    th = pd.to_numeric(df[thcol], errors="coerce")
    mask = t.notna() & th.notna()
    t = t[mask].to_numpy()
    th = th[mask].to_numpy()
    if len(t) < 2: return None
    dt = np.diff(t)
    dth = np.diff(th)
    dt = dt[np.isfinite(dt) & (dt!=0)]
    dth = dth[:len(dt)]
    if len(dt)==0: return None
    return float(np.mean(np.abs(dth/dt)))

def first_collision(diag_csv):
    """
    event 컬럼에서 'collision' 최초 시간, 없으면 None
    """
    df = _read_csv(diag_csv)
    if df is None or df.empty: return None
    ecol = _first_or_none([c for c in df.columns if c.lower() in ("event","evt","status")])
    tcol = _first_or_none([c for c in df.columns if c.lower() in ("t","time")])
    if not ecol or not tcol: return None
    e = df[ecol].astype(str).str.lower()
    hit = df.loc[e.str.contains("collision", na=False)]
    if hit.empty: return None
    try:
        return float(pd.to_numeric(hit[tcol], errors="coerce").dropna().iloc[0])
    except Exception:
        return None

def parse_params(run_dirname):
    """
    폴더명 a1.1_tau1.3_kc0.05_gf0.30 → dict 파싱
    """
    d = {}
    m = re.search(r"a(?P<a>[0-9.]+)", run_dirname)
    if m: d["alpha_dtg"] = float(m.group("a"))
    m = re.search(r"tau(?P<tau>[0-9.]+)", run_dirname)
    if m: d["tau_theta"] = float(m.group("tau"))
    m = re.search(r"kc(?P<kc>[0-9.]+)", run_dirname)
    if m: d["k_ctrl"] = float(m.group("kc"))
    m = re.search(r"gf(?P<gf>[0-9.]+)", run_dirname)
    if m: d["gate_floor"] = float(m.group("gf"))
    return d

rows = []
roots = sorted(glob.glob("runs/fine_*"))
nus = ["0.003","0.0035"]

for root in roots:
    # 하위 조합 폴더들
    for cfg_dir in sorted(glob.glob(os.path.join(root, "a*_tau*_kc*_gf*"))):
        params = parse_params(os.path.basename(cfg_dir))
        if not params:
            continue
        for nu in nus:
            b_diag = glob.glob(f"{cfg_dir}/baseline_nu{nu}/data/diagnostics_*.csv")
            d_diag = glob.glob(f"{cfg_dir}/dtg_nu{nu}/data/diagnostics_*.csv")
            d_main = glob.glob(f"{cfg_dir}/dtg_nu{nu}/data/threebody3d_*.csv")
            if not (b_diag and d_diag):
                print(f"[skip] missing diagnostics for {cfg_dir} nu={nu}")
                continue
            mb = drift_metrics(b_diag[0])
            md = drift_metrics(d_diag[0])
            if mb is None or md is None:
                print(f"[skip] drift metrics not found → {cfg_dir} nu={nu}")
                continue
            base = mb["rms"]; dtg = md["rms"]
            if not (math.isfinite(base) and math.isfinite(dtg)):
                print(f"[skip] non-finite RMS → {cfg_dir} nu={nu}")
                continue
            imp = 100.0 * (base - dtg) / max(base, 1e-12)
            th_rate = theta_rate(d_main[0]) if d_main else None
            tcol = first_collision(d_diag[0])

            rows.append(dict(
                run=os.path.relpath(cfg_dir),
                nu=float(nu),
                drift_rms_base=base,
                drift_rms_dtg=dtg,
                improve_pct=imp,
                theta_rate=th_rate,
                t_collision=tcol,
                **params
            ))

# 데이터프레임 & 저장
if not rows:
    print("No valid rows found. (확인: runs/fine_*/ 하위에 diagnostics / threebody3d CSV 존재 여부)")
    exit(0)

df = pd.DataFrame(rows).sort_values(["improve_pct"], ascending=False)
csv_path = "runs/summary/fine_summary.csv"
md_path  = "runs/summary/fine_summary.md"
df.to_csv(csv_path, index=False)

md = ["# Fine sweep summary\n", df.to_markdown(index=False), ""]
open(md_path, "w").write("\n".join(md))

# 추천 (충돌 없는 케이스 중 최고 개선)
ok = df[df["t_collision"].isna()].copy()
print()
if not ok.empty:
    best = ok.iloc[0]
    print("Recommended config (no collision):")
    print(f"- alpha_dtg={best.alpha_dtg}, tau_theta={best.tau_theta}, k_ctrl={best.k_ctrl}, gate_floor={best.gate_floor}")
    print(f"- improve≈{best.improve_pct:.2f}%  (RMS {best.drift_rms_base:.6f} → {best.drift_rms_dtg:.6f})  theta_rate≈{best.theta_rate if pd.notna(best.theta_rate) else 'NA'}")
else:
    print("No collision-free rows found in sweep (check t_collision).")

print(f"\nSaved: {csv_path}\nSaved: {md_path}")
