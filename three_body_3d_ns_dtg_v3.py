#!/usr/bin/env python3
# three_body_3d_ns_dtg_v3_1.py
# 3체(3D) + 항력(ν) + (옵션) 난류 + DTG(연속시간)
# 개선점:
# - Lyapunov: 경량(top-1) 모드(--lyap_light) + 스펙트럼(--lyap_spectrum) 선택
# - 비교 모드(--compare): baseline(dtg=0) vs dtg=1 자동 연쇄 실행 후 상대 개선율 판정
# - 이벤트: escape / collision(min distance)
# - 바이너리 검출 가속(누적합)
# - 플롯 토글(--no_plots)
# - eps 기본 1e-8로 상향(+경고), provenance 강화

import argparse, os, json, time, sys, platform
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_EPS = 1e-8

# ========================== Utilities ==========================
def unpack_state_bodies(s, N=3):
    s = np.asarray(s, float).reshape(-1)
    if s.size != 6*N:
        raise ValueError(f"state length must be 6N (= {6*N}), got {s.size}")
    pos = s[:3*N].reshape(3, N, order="F")
    vel = s[3*N:6*N].reshape(3, N, order="F")
    return pos, vel

def pack_state_bodies(pos, vel):
    if pos.shape != vel.shape or pos.shape[0] != 3:
        raise ValueError("pos/vel must be (3,N)")
    N = pos.shape[1]
    return np.concatenate([pos.reshape(3*N, order="F"),
                           vel.reshape(3*N, order="F")])

def split_state_with_theta(s, N=3):
    s = np.asarray(s, float).reshape(-1)
    body, theta = s[:-1], float(s[-1])
    pos, vel = unpack_state_bodies(body, N=N)
    return pos, vel, theta

def pairwise_dr(pos):
    dr = pos[:, None, :] - pos[:, :, None]    # (3,N,N)
    r2 = np.sum(dr*dr, axis=0)                # (N,N)
    return dr, r2

def accelerations(pos, vel, G, masses, eps=1e-8,
                  nu_eff=0.0, turb_coeff=0.0, k_ctrl_eff=0.0, theta_dev=0.0):
    N = pos.shape[1]
    dr, r2 = pairwise_dr(pos)
    r2 = r2 + eps*eps
    mask = ~np.eye(N, dtype=bool)
    inv_r3 = np.where(mask, r2**-1.5, 0.0)
    w = masses[None, :] * inv_r3
    grav = G * np.einsum('kij,ij->ki', dr, w)     # (3,N)
    visc = -nu_eff * vel
    turb = turb_coeff * np.sin(np.sum(pos*pos, axis=0))[None, :] * vel if turb_coeff else 0.0
    ctrl = -k_ctrl_eff * theta_dev * vel
    return grav + visc + turb + ctrl

def total_energy_body(s_body, G=1.0, masses=(1,1,1), eps=1e-8):
    pos, vel = unpack_state_bodies(s_body, N=3)
    K = 0.5 * np.sum(masses * np.sum(vel*vel, axis=0))
    dr, r2 = pairwise_dr(pos)
    r = np.sqrt(r2 + eps*eps)
    iu = np.triu_indices(len(masses), 1)
    U = -G * np.sum(masses[iu[0]] * masses[iu[1]] / r[iu])
    return K + U

def make_ic(mode="exp2", alpha=1.0):
    if mode == "exp1":
        raw = [0,0,0, 0,0,0, 1,0,0, 0,0.6,0.1, -1,0,0, 0,-0.6,-0.1]
    elif mode == "exp2":
        raw = [0,0,0, 0,0,0, 1,0,0, 0,0.8,0.2, -1,0,0, 0,-0.5,-0.05]
    elif mode == "exp3":
        raw = [0,0,0, 0,0,0, 1,0,0, 0,1.0,0.4, -1,0,0, 0,-0.2,0.0]
    elif mode == "figure8":
        raw = [
            -0.97000436, 0.24308753, 0, 0.466203685, 0.43236573, 0,
             0.97000436,-0.24308753, 0, 0.466203685, 0.43236573, 0,
             0.0,        0.0,       0,-0.93240737,-0.86473146, 0
        ]
    else:
        raise ValueError("mode must be exp1|exp2|exp3|figure8")
    s = np.array(raw, float)
    pos = np.vstack([s[[0,6,12]], s[[1,7,13]], s[[2,8,14]]])
    vel = np.vstack([s[[3,9,15]], s[[4,10,16]], s[[5,11,17]]])
    vel *= alpha
    return pos, vel

def zero_com_and_momentum(pos, vel, masses):
    m = masses.reshape(1, -1)
    com = (pos*m).sum(axis=1)/m.sum()
    pos = pos - com.reshape(3,1)
    p = (vel*m).sum(axis=1, keepdims=True)
    vel = vel - p/m.sum()
    return pos, vel

def positions_from_sol_bodyY(Y, N=3):
    rows = np.arange(N)
    x_rows = 3*rows + 0; y_rows = 3*rows + 1; z_rows = 3*rows + 2
    return Y[x_rows, :], Y[y_rows, :], Y[z_rows, :]

def min_pair_distance(pos):
    _, r2 = pairwise_dr(pos)
    N = r2.shape[0]
    m = r2 + np.eye(N)*1e9
    return float(np.sqrt(np.min(m)))

def hold_true(x_bool, win):
    c = np.convolve(x_bool.astype(int), np.ones(win, dtype=int), 'same')
    return np.any(c >= win)

# ================= Conservative RHS / Lyapunov =================
def rhs_body_conservative(t, s_body, args_rhs):
    G, masses, eps = args_rhs["G"], args_rhs["masses"], args_rhs["eps"]
    pos, vel = unpack_state_bodies(s_body, N=3)
    acc = accelerations(pos, vel, G, masses, eps=eps,
                        nu_eff=0.0, turb_coeff=0.0, k_ctrl_eff=0.0, theta_dev=0.0)
    return pack_state_bodies(vel, acc)

def lyap_spectrum_benettin(s0_body, rhs_body, tmax=12.0, dt=0.002, k=3, args_rhs=None, seed=0):
    rng = np.random.default_rng(seed)
    n = s0_body.size
    Q = rng.normal(size=(n, k))
    Q, _ = np.linalg.qr(Q)
    lyap = np.zeros(k)
    t = 0.0
    s = s0_body.copy()
    while t < tmax - 1e-12:
        t_next = min(t + dt, tmax)
        sol = solve_ivp(lambda tt, ss: rhs_body(tt, ss, args_rhs),
                        (t, t_next), s, method="DOP853", rtol=1e-9, atol=1e-12, max_step=dt)
        s = sol.y[:, -1]
        eps_fd = 1e-8
        Z = np.zeros_like(Q)
        for j in range(k):
            sol2 = solve_ivp(lambda tt, ss: rhs_body(tt, ss, args_rhs),
                             (t, t_next), s + eps_fd*Q[:, j],
                             method="DOP853", rtol=1e-9, atol=1e-12, max_step=dt)
            Z[:, j] = (sol2.y[:, -1] - s)/eps_fd
        Q, R = np.linalg.qr(Z)
        lyap += np.log(np.abs(np.diag(R)) + 1e-300)
        t = t_next
    return lyap / (tmax)

def benettin_light_max(s0_body, rhs_body, dt=0.01, renorm_every=50, tmax=12.0, args_rhs=None, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=s0_body.size); v /= np.linalg.norm(v)
    s = s0_body.copy()
    lam_sum, t, d0 = 0.0, 0.0, 1e-8
    s_pert = s + d0 * v
    while t < tmax - 1e-12:
        t_next = min(t + renorm_every * dt, tmax)
        sol1 = solve_ivp(lambda tt, ss: rhs_body(tt, ss, args_rhs),
                         (t, t_next), s, method="DOP853", rtol=1e-9, atol=1e-12, max_step=dt)
        sol2 = solve_ivp(lambda tt, ss: rhs_body(tt, ss, args_rhs),
                         (t, t_next), s_pert, method="DOP853", rtol=1e-9, atol=1e-12, max_step=dt)
        s = sol1.y[:, -1]; s_pert = sol2.y[:, -1]
        d = np.linalg.norm(s_pert - s);  d = d if d > 0 else 1e-300
        lam_sum += np.log(d/d0)
        v = (s_pert - s) / d
        s_pert = s + d0 * v
        t = t_next
    T = max(t, 1e-30)
    return lam_sum / T

# ========================== DTG RHS ==========================
class RHSParams:
    def __init__(self, G, masses, eps,
                 nu0=0.0, nu_max=1.0, turb_coeff=0.0,
                 theta0=1.0, alpha_dtg=0.7, beta_dtg=0.5, b_dtg=0.0, tau_theta=1.0,
                 theta_min=-1.5, theta_max=1.5, dtheta_max=0.05,
                 k_nu=0.0, k_nu_ed=0.0, k_ctrl=0.0, k_ctrl_e=0.0,
                 lyap_const=0.0, E0=None, soften_ctrl_when_clamped=True):
        self.G, self.masses, self.eps = G, np.asarray(masses,float), float(eps)
        self.nu0, self.nu_max, self.turb_coeff = float(nu0), float(nu_max), float(turb_coeff)
        self.theta0 = float(theta0)
        self.alpha_dtg, self.beta_dtg, self.b_dtg = float(alpha_dtg), float(beta_dtg), float(b_dtg)
        self.tau_theta = float(tau_theta)
        self.theta_min, self.theta_max = float(theta_min), float(theta_max)
        self.dtheta_max = float(dtheta_max)
        self.k_nu, self.k_nu_ed = float(k_nu), float(k_nu_ed)
        self.k_ctrl, self.k_ctrl_e = float(k_ctrl), float(k_ctrl_e)
        self.lyap_const = float(lyap_const)
        self.E0 = None if E0 is None else float(E0)
        self.soften_ctrl_when_clamped = bool(soften_ctrl_when_clamped)

def rhs_with_dtg(t, s_all, prm: RHSParams):
    pos, vel, theta = split_state_with_theta(s_all, N=3)
    if prm.E0 is not None:
        E = total_energy_body(pack_state_bodies(pos, vel), prm.G, prm.masses, eps=prm.eps)
        ed = (E - prm.E0) / (abs(prm.E0) + 1e-15)
    else:
        ed = 0.0
    target = prm.b_dtg + prm.alpha_dtg*ed - prm.beta_dtg*prm.lyap_const
    dtheta_raw = (target - theta) / max(prm.tau_theta, 1e-9)
    dtheta = np.clip(dtheta_raw, -prm.dtheta_max, prm.dtheta_max)

    theta_clamped = np.clip(theta, prm.theta_min, prm.theta_max)
    theta_dev = theta_clamped - prm.theta0
    nu_eff = np.clip(prm.nu0 + prm.k_nu*theta_dev + prm.k_nu_ed*ed, 0.0, prm.nu_max)
    k_ctrl_eff = prm.k_ctrl * (1.0 + prm.k_ctrl_e*abs(ed))
    if prm.soften_ctrl_when_clamped and (theta != theta_clamped):
        k_ctrl_eff *= 0.5

    acc = accelerations(pos, vel, prm.G, prm.masses, eps=prm.eps,
                        nu_eff=nu_eff, turb_coeff=prm.turb_coeff,
                        k_ctrl_eff=k_ctrl_eff, theta_dev=theta_dev)
    return np.concatenate([pack_state_bodies(vel, acc), np.array([dtheta])])

# ========================== Run core ==========================
def run_simulation(args):
    if args.eps < 1e-10:
        print(f"[warn] eps={args.eps:g} is very small; risk of blow-up near close encounters.")

    os.makedirs(args.out, exist_ok=True)
    for sub in ["figures", "data", "meta"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    masses = np.array([1.0,1.0,1.0], float)
    pos0, vel0 = make_ic(args.ic, args.alpha)
    if args.zero_init:
        pos0, vel0 = zero_com_and_momentum(pos0, vel0, masses)

    s0_body = pack_state_bodies(pos0, vel0)
    E0 = total_energy_body(s0_body, args.G, masses, eps=args.eps)

    # Lyapunov choice
    lyap_top = None
    lyap_spec = None
    if args.lyap_light:
        lyap_top = benettin_light_max(
            s0_body, rhs_body_conservative,
            dt=args.lyap_light_dt, renorm_every=args.lyap_light_renorm,
            tmax=max(6.0, args.tmax),
            args_rhs=dict(G=args.G, masses=masses, eps=args.eps),
            seed=args.seed
        )
        print(f"[Lyapunov(top, light) ≈] {lyap_top:.6f}")
    if args.lyap_spectrum:
        lyap_spec = lyap_spectrum_benettin(
            s0_body, rhs_body_conservative,
            tmax=max(6.0, args.tmax), dt=args.dt, k=args.lyap_k,
            args_rhs=dict(G=args.G, masses=masses, eps=args.eps),
            seed=args.seed
        )
        print("[Lyapunov spectrum (top-k)]", " ".join(f"{v:.6f}" for v in lyap_spec))

    lyap_const = float(lyap_top if (lyap_top is not None) else (lyap_spec[0] if lyap_spec is not None else args.lyap_const))

    prm = RHSParams(
        G=args.G, masses=masses, eps=args.eps,
        nu0=args.nu0, nu_max=args.nu_max, turb_coeff=args.turb,
        theta0=args.theta0, alpha_dtg=args.alpha_dtg, beta_dtg=args.beta_dtg,
        b_dtg=args.b_dtg, tau_theta=args.tau_theta,
        theta_min=args.theta_min, theta_max=args.theta_max, dtheta_max=args.dtheta_max,
        k_nu=args.k_nu, k_nu_ed=args.k_nu_ed, k_ctrl=args.k_ctrl, k_ctrl_e=args.k_ctrl_e,
        lyap_const=lyap_const, E0=(E0 if args.dtg else None),
        soften_ctrl_when_clamped=not args.no_soften_when_clamped
    )

    t_eval = np.arange(0.0, args.tmax + 1e-12, args.dt)
    s_all = np.concatenate([s0_body, np.array([args.theta0])])

    # events
    def event_escape(t, s):
        pos, _, _ = split_state_with_theta(s, N=3)
        r = np.sqrt(np.sum(pos*pos, axis=0))
        return float(args.R_esc - np.max(r))
    event_escape.terminal = True
    event_escape.direction = -1.0

    def event_collision(t, s):
        pos, _, _ = split_state_with_theta(s, N=3)
        return min_pair_distance(pos) - args.r_collide
    event_collision.terminal = True
    event_collision.direction = -1.0

    sol = solve_ivp(lambda tt, ss: rhs_with_dtg(tt, ss, prm),
                    (0.0, args.tmax), s_all, t_eval=t_eval,
                    events=(event_escape, event_collision),
                    method="DOP853", rtol=1e-9, atol=1e-12, max_step=args.dt)

    Y, T = sol.y, sol.t
    bodyY, theta_hist = Y[:-1, :], Y[-1, :]
    x, y, z = positions_from_sol_bodyY(bodyY, N=3)

    # diagnostics
    E_series = np.array([total_energy_body(bodyY[:, i], args.G, masses, eps=args.eps)
                         for i in range(bodyY.shape[1])])
    drift = (E_series - E0) / (abs(E0) + 1e-15)

    rmin_series = np.array([min_pair_distance(unpack_state_bodies(bodyY[:, i], N=3)[0])
                            for i in range(bodyY.shape[1])])

    r12 = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2 + (z[0]-z[1])**2)
    bin_window = max(1, int(args.bin_hold/args.dt))
    binary_flag = hold_true((r12 < args.r_bin), bin_window)

    # CSVs
    df_main = pd.DataFrame({
        "t": T,
        "x1": x[0], "y1": y[0], "z1": z[0],
        "x2": x[1], "y2": y[1], "z2": z[1],
        "x3": x[2], "y3": y[2], "z3": z[2],
        "theta": theta_hist
    })
    csv_path = os.path.join(args.out, "data", f"threebody3d_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.csv")
    df_main.to_csv(csv_path, index=False)

    df_diag = pd.DataFrame({"t": T, "energy": E_series, "drift": drift,
                            "rmin": rmin_series, "r12": r12, "theta": theta_hist})
    csv_diag = os.path.join(args.out, "data", f"diagnostics_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.csv")
    df_diag.to_csv(csv_diag, index=False)

    # plots
    if not args.no_plots:
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x[0],y[0],z[0],label="Body 1")
        ax.plot(x[1],y[1],z[1],label="Body 2")
        ax.plot(x[2],y[2],z[2],label="Body 3")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"3D Three-Body (ic={args.ic}, α={args.alpha}, dtg={args.dtg})")
        ax.legend()
        try: ax.set_box_aspect((1,1,1)); ax.set_proj_type('ortho')
        except Exception: pass
        ax.view_init(elev=args.elev, azim=args.azim)
        fig3d = os.path.join(args.out, "figures", f"threebody3d_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
        fig.savefig(fig3d, dpi=160); plt.close(fig)

        plt.figure(figsize=(6,5))
        for i, lab in enumerate(["Body 1","Body 2","Body 3"]):
            plt.plot(x[i], y[i], label=lab)
        plt.axis("equal"); plt.xlabel("X"); plt.ylabel("Y"); plt.legend()
        plt.title(f"XY Projection (ic={args.ic}, α={args.alpha}, dtg={args.dtg})")
        fig_xy = os.path.join(args.out, "figures", f"xy_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
        plt.savefig(fig_xy, dpi=160); plt.close()

        plt.figure(figsize=(7,4)); plt.plot(T, drift)
        plt.xlabel("Time"); plt.ylabel("Relative Energy Drift")
        plt.title("Total Energy Drift (reference)")
        fig_drift = os.path.join(args.out, "figures", f"energy_drift_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
        plt.savefig(fig_drift, dpi=160); plt.close()

        plt.figure(figsize=(7,4)); plt.plot(T, theta_hist)
        plt.xlabel("Time"); plt.ylabel("Theta (DTG)")
        plt.title("DTG Theta")
        fig_theta = os.path.join(args.out, "figures", f"theta_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
        plt.savefig(fig_theta, dpi=160); plt.close()
    else:
        fig3d = fig_drift = fig_theta = fig_xy = None

    # meta
    meta = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        args=vars(args),
        E0=float(E0),
        lyap_top=float(lyap_top) if lyap_top is not None else None,
        lyap_spectrum=(list(map(float, lyap_spec)) if lyap_spec is not None else None),
        outputs=dict(csv=csv_path, csv_diagnostics=csv_diag,
                     fig3d=fig3d, fig_xy=fig_xy, drift=fig_drift, theta=fig_theta),
        events=dict(
            escaped=(len(sol.t_events[0])>0),
            t_escape=(float(sol.t_events[0][0]) if len(sol.t_events[0]) else None),
            collided=(len(sol.t_events[1])>0),
            t_collision=(float(sol.t_events[1][0]) if len(sol.t_events[1]) else None)
        ),
        binary=dict(flag=bool(binary_flag), r_bin=float(args.r_bin), hold=float(args.bin_hold)),
        end_state=dict(rmin=float(rmin_series[-1]), drift=float(drift[-1]), theta=float(theta_hist[-1]))
    )
    meta["provenance"] = dict(
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=pd.__version__,   # NOTE: will show pandas ver here; override below
        pandas=pd.__version__,
        cmd=" ".join(sys.argv)
    )
    try:
        import scipy
        meta["provenance"]["scipy"] = scipy.__version__
    except Exception:
        pass

    meta_path = os.path.join(args.out, "meta", f"meta_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] CSV   : {csv_path}")
    print(f"[OK] DIAG  : {csv_diag}")
    if fig3d:   print(f"[OK] FIG3D : {fig3d}")
    if fig_xy:  print(f"[OK] XY    : {fig_xy}")
    if fig_drift: print(f"[OK] DRIFT : {fig_drift}")
    if fig_theta: print(f"[OK] THETA : {fig_theta}")
    if meta["events"]["escaped"]:
        print(f"[event] escape at t≈{meta['events']['t_escape']:.3f}")
    if meta["events"]["collided"]:
        print(f"[event] collision at t≈{meta['events']['t_collision']:.3f}")
    if binary_flag:
        print(f"[info] binary formation detected (r12<{args.r_bin}, hold≥{args.bin_hold}s)")

    return meta, csv_diag, (lyap_spec if lyap_spec is not None else [lyap_top] if lyap_top is not None else []), drift, theta_hist

# ========================== Compare / Analyze ==========================
def run_and_collect(args):
    return run_simulation(args)

def compare_runs(csv_b, csv_d, meta_b, meta_d, drift_key="drift"):
    b = pd.read_csv(csv_b); d = pd.read_csv(csv_d)
    b_max = b[drift_key].abs().max(); d_max = d[drift_key].abs().max()
    imp_drift = 100*(b_max - d_max)/max(1e-12, b_max)

    esc_b = 0 if meta_b["events"]["escaped"] else 1
    esc_d = 0 if meta_d["events"]["escaped"] else 1
    imp_escape = 100*(esc_d - esc_b)  # 0->1이면 +100

    col_b = 0 if meta_b["events"]["collided"] else 1
    col_d = 0 if meta_d["events"]["collided"] else 1
    imp_collision = 100*(col_d - col_b)

    score = sum(v > 30 for v in [imp_drift, imp_escape, imp_collision])
    verdict = "PASS" if score >= 1 else "NO-GAIN"
    return dict(
        verdict=verdict,
        drift_improve=imp_drift,
        escape_improve=imp_escape,
        collision_improve=imp_collision
    )

# ============================ CLI ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", choices=["exp1","exp2","exp3","figure8"], default="exp2")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tmax", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--out", default="runs/out")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero_init", action="store_true")
    ap.add_argument("--no_plots", action="store_true")

    # drag / turbulence
    ap.add_argument("--nu0", type=float, default=0.003)
    ap.add_argument("--nu_max", type=float, default=1.0)
    ap.add_argument("--turb", type=float, default=0.0)

    # DTG
    ap.add_argument("--dtg", type=int, default=1)
    ap.add_argument("--theta0", type=float, default=1.0)
    ap.add_argument("--alpha_dtg", type=float, default=0.7)
    ap.add_argument("--beta_dtg", type=float, default=0.5)
    ap.add_argument("--b_dtg", type=float, default=0.0)
    ap.add_argument("--tau_theta", type=float, default=1.0)
    ap.add_argument("--theta_min", type=float, default=-1.5)
    ap.add_argument("--theta_max", type=float, default=1.5)
    ap.add_argument("--dtheta_max", type=float, default=0.05)
    ap.add_argument("--k_nu", type=float, default=0.0)
    ap.add_argument("--k_nu_ed", type=float, default=0.0)
    ap.add_argument("--k_ctrl", type=float, default=0.0)
    ap.add_argument("--k_ctrl_e", type=float, default=0.0)
    ap.add_argument("--lyap_const", type=float, default=0.0)
    ap.add_argument("--no_soften_when_clamped", action="store_true")

    # events / binary
    ap.add_argument("--R_esc", type=float, default=10.0)
    ap.add_argument("--r_collide", type=float, default=1e-3)
    ap.add_argument("--r_bin", type=float, default=0.5)
    ap.add_argument("--bin_hold", type=float, default=1.0)

    # Lyapunov options
    ap.add_argument("--lyap_light", action="store_true")
    ap.add_argument("--lyap_light_dt", type=float, default=0.01)
    ap.add_argument("--lyap_light_renorm", type=int, default=50)
    ap.add_argument("--lyap_spectrum", action="store_true")
    ap.add_argument("--lyap_k", type=int, default=3)

    # Compare
    ap.add_argument("--compare", action="store_true",
                    help="baseline(dtg=0) vs dtg=1 자동 비교 실행")

    args = ap.parse_args()

    if args.compare:
        # baseline
        b_args = argparse.Namespace(**vars(args))
        b_args.dtg = 0; b_args.out = os.path.join(args.out, "baseline")
        meta_b, csv_b, *_ = run_and_collect(b_args)
        # dtg
        d_args = argparse.Namespace(**vars(args))
        d_args.dtg = 1; d_args.out = os.path.join(args.out, "dtg")
        meta_d, csv_d, *_ = run_and_collect(d_args)
        # compare
        res = compare_runs(meta_b["outputs"]["csv_diagnostics"], meta_d["outputs"]["csv_diagnostics"], meta_b, meta_d)
        print(f"[COMPARE] verdict={res['verdict']} | drift={res['drift_improve']:.1f}% | "
              f"escape={res['escape_improve']:.1f}% | collision={res['collision_improve']:.1f}%")
        return

    run_simulation(args)

if __name__ == "__main__":
    main()