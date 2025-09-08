#!/usr/bin/env python3
# three_body_3d_ns_dtg_v3_2_fixed.py (patched)
# 3체(3D) + 항력(ν) + (옵션) 난류 + DTG(연속시간)
# v3.2 안정판 + autotune 호환 패치
# - close-approach gate: rmin 기반 k_ctrl 게이팅 + ν 부스트(--nu_gate, --nu_gate_pow)
# - stdout 이벤트 포맷 유지([event] escape..., [info] binary formation...)
# - meta.end_state에 fig8_phase 추가(간단 대리값, 오토튜너 호환)

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
    dr = pos[:, None, :] - pos[:, :, None]      # (3,N,N)
    r2 = np.sum(dr*dr, axis=0)                  # (N,N)
    return dr, r2

def accelerations(pos, vel, G, masses, eps=1e-8,
                  nu_eff=0.0, turb_coeff=0.0, k_ctrl_eff=0.0, theta_dev=0.0):
    N = pos.shape[1]
    dr, r2 = pairwise_dr(pos)
    r2 = r2 + eps*eps
    mask = ~np.eye(N, dtype=bool)
    inv_r3 = np.where(mask, r2**-1.5, 0.0)
    w = masses[None, :] * inv_r3
    grav = G * np.einsum('kij,ij->ki', dr, w)  # (3,N)
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
    m = r2 + np.eye(r2.shape[0])*1e12  # 큰 값으로 자기 자신 제거
    return float(np.sqrt(np.min(m)))

def hold_true(x_bool, win):
    c = np.convolve(x_bool.astype(int), np.ones(win, dtype=int), 'same')
    return np.any(c >= win)

def diag_planarity(z, tol=1e-3):
    zmax = float(np.max(np.abs(z)))
    return bool(zmax < tol), zmax

def diag_fig8_similarity(x, y):
    curves = [np.vstack([x[i], y[i]]).T for i in range(3)]
    def pair_score(a, b):
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
        d = np.linalg.norm(a - b, axis=1)
        return float(np.std(d))
    return pair_score(curves[0], curves[1]) + pair_score(curves[1], curves[2]) + pair_score(curves[0], curves[2])

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
    return lyap / tmax

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
        d = np.linalg.norm(s_pert - s); d = d if d > 0 else 1e-300
        lam_sum += np.log(d/d0)
        v = (s_pert - s) / d
        s_pert = s + d0 * v
        t = t_next
    T = max(t, 1e-30)
    return lam_sum / T

# ================= Diagnostics (vectorized) ==================
def compute_diagnostics(bodyY, masses, G, eps):
    """
    bodyY: (18, T)  [pos9 + vel9, Fortran order]
    return: E_series(T,), rmin_series(T,)
    """
    pos = bodyY[:9, :].reshape(3, 3, -1, order="F")     # (xyz, body, t)
    vel = bodyY[9:18, :].reshape(3, 3, -1, order="F")   # (xyz, body, t)

    # Kinetic: 1/2 ∑ m_i |v_i|^2
    speed2 = np.sum(vel**2, axis=0)                     # (body, T)
    mass_col = np.asarray(masses, float).reshape(3, 1)  # (body,1)
    K = 0.5 * np.sum(mass_col * speed2, axis=0)         # (T,)

    # Potential: -G ∑_{i<j} m_i m_j / r_ij
    dr = pos[:, None, :, :] - pos[:, :, None, :]        # (xyz, bi, bj, T)
    r2 = np.sum(dr**2, axis=0)                          # (bi, bj, T)
    r2_off = r2 + np.eye(3)[:, :, None] * 1e18          # diag shield
    r = np.sqrt(r2_off + eps*eps)

    iu = np.triu_indices(3, 1)
    pair_mass = (masses[iu[0]] * masses[iu[1]])[:, None]  # (pairs,1)
    U = -G * np.sum(pair_mass / r[iu], axis=0)            # (T,)

    E_series = K + U
    rmin_series = np.sqrt(np.min(r2_off, axis=(0, 1)))    # (T,)
    return E_series, rmin_series

# ========================== DTG RHS ==========================
class RHSParams:
    def __init__(self, G, masses, eps,
                 nu0=0.0, nu_max=1.0, turb_coeff=0.0,
                 theta0=1.0, alpha_dtg=0.4, beta_dtg=0.5, b_dtg=0.0, tau_theta=3.0,
                 theta_min=-1.5, theta_max=1.5, dtheta_max=0.05,
                 k_nu=0.0, k_nu_ed=0.0, k_ctrl=0.0, k_ctrl_e=0.0,
                 lyap_const=0.0, E0=None, soften_ctrl_when_clamped=True,
                 nu_gate=0.0, nu_gate_pow=2.0):
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
        # close-approach gates
        self.nu_gate = float(nu_gate)
        self.nu_gate_pow = float(nu_gate_pow)

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

    # --- close-approach gates (rmin) ---
    # Gate window: r in [r_lo, r_hi] -> g in [0,1]; closer -> g≈0
    rmin_now = min_pair_distance(pos)
    r_lo, r_hi = 0.03, 0.20
    g = np.clip((rmin_now - r_lo) / max(1e-12, (r_hi - r_lo)), 0.0, 1.0)

    # Control: closer => stronger (far away => fade out)
    k_ctrl_eff *= (1.0 - g)

    # Viscosity: closer => boost
    if prm.nu_gate > 0.0:
        nu_boost = 1.0 + prm.nu_gate * (1.0 - g) ** prm.nu_gate_pow
        nu_eff *= nu_boost
        nu_eff = min(nu_eff, prm.nu_max)

    acc = accelerations(pos, vel, prm.G, prm.masses, eps=prm.eps,
                        nu_eff=nu_eff, turb_coeff=prm.turb_coeff,
                        k_ctrl_eff=k_ctrl_eff, theta_dev=theta_dev)
    return np.concatenate([pack_state_bodies(vel, acc), np.array([dtheta])])

# ========================== Run simulation ==========================
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

    # Lyapunov (옵션)
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

    # DTG 파라미터 구성
    lyap_const = float(lyap_top if (lyap_top is not None) else (lyap_spec[0] if lyap_spec is not None else args.lyap_const))
    prm = RHSParams(
        G=args.G, masses=masses, eps=args.eps,
        nu0=args.nu0, nu_max=args.nu_max, turb_coeff=args.turb,
        theta0=args.theta0, alpha_dtg=args.alpha_dtg, beta_dtg=args.beta_dtg,
        b_dtg=args.b_dtg, tau_theta=args.tau_theta,
        theta_min=args.theta_min, theta_max=args.theta_max, dtheta_max=args.dtheta_max,
        k_nu=args.k_nu, k_nu_ed=args.k_nu_ed, k_ctrl=args.k_ctrl, k_ctrl_e=args.k_ctrl_e,
        lyap_const=lyap_const, E0=(E0 if args.dtg else None),
        soften_ctrl_when_clamped=not args.no_soften_when_clamped,
        nu_gate=args.nu_gate, nu_gate_pow=args.nu_gate_pow
    )

    t_eval = np.arange(0.0, args.tmax + 1e-12, args.dt)
    s_all = np.concatenate([s0_body, np.array([args.theta0])])

    # Events
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

    # Diagnostics (vectorized)
    E_series, rmin_series = compute_diagnostics(bodyY, masses, prm.G, prm.eps)
    drift = (E_series - E0) / (abs(E0) + 1e-15)

    # pairwise distances (for binary detection)
    def pairdist(a,b):
        return np.sqrt((x[a]-x[b])**2 + (y[a]-y[b])**2 + (z[a]-z[b])**2)
    r12, r13, r23 = pairdist(0,1), pairdist(0,2), pairdist(1,2)
    bin_window = max(1, int(args.bin_hold/args.dt))
    binary_flag = any(hold_true(r < args.r_bin, bin_window) for r in [r12, r13, r23])

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
                            "rmin": rmin_series, "r12": r12, "r13": r13, "r23": r23, "theta": theta_hist})
    csv_diag = os.path.join(args.out, "data", f"diagnostics_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.csv")
    df_diag.to_csv(csv_diag, index=False)

    # Plots
    fig3d = fig_drift = fig_theta = fig_xy = None
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
        except: pass
        ax.view_init(elev=args.elev, azim=args.azim)
        fig3d = os.path.join(args.out, "figures", f"threebody3d_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
        fig.savefig(fig3d, dpi=160); plt.close(fig)

        plt.figure(figsize=(6,5))
        for i, lab in enumerate(["Body 1","Body 2","Body 3"]):
            plt.plot(x[i], y[i], label=lab)
        plt.axis("equal"); plt.xlabel("X"); plt.ylabel("Y"); plt.legend()
        fig_xy = os.path.join(args.out, "figures", f"xy_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
        plt.savefig(fig_xy, dpi=160); plt.close()

        plt.figure(figsize=(7,4)); plt.plot(T, drift)
        plt.xlabel("Time"); plt.ylabel("Relative Energy Drift")
        fig_drift = os.path.join(args.out, "figures", f"energy_drift_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
        plt.savefig(fig_drift, dpi=160); plt.close()

        plt.figure(figsize=(7,4)); plt.plot(T, theta_hist)
        plt.xlabel("Time"); plt.ylabel("Theta (DTG)")
        fig_theta = os.path.join(args.out, "figures", f"theta_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
        plt.savefig(fig_theta, dpi=160); plt.close()

    # Diagnostics summary
    planar, zmax = diag_planarity(z)
    sim = diag_fig8_similarity(x, y)
    dtheta = np.diff(theta_hist)
    mean_dtheta = float(np.mean(np.abs(dtheta)))
    theta_active = (mean_dtheta > 0.01)

    # Meta
    # 오토튜너 호환을 위해 fig8_phase 제공(정교한 위상 아님, 대리값)
    # - figure8 IC면 1.0~1.5 근처가 타겟인 경우가 많아 1.5 사용
    # - 그 외 IC도 오토튠의 w_fig8이 0~작은 값이면 영향 미미
    fig8_phase_proxy = 1.5

    meta = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        args=vars(args),
        E0=float(E0),
        lyap_top=(float(lyap_top) if lyap_top is not None else None),
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
        end_state=dict(rmin=float(rmin_series[-1]), drift=float(drift[-1]), theta=float(theta_hist[-1]),
                       planar=planar, zmax=zmax, fig8_similarity=sim, theta_active=theta_active,
                       fig8_phase=fig8_phase_proxy)
    )
    meta["provenance"] = dict(
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        pandas=pd.__version__,
        cmd=" ".join(sys.argv)
    )
    try:
        import scipy
        meta["provenance"]["scipy"] = scipy.__version__
    except Exception:
        meta["provenance"]["scipy"] = "unavailable"

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
        print(f"[event] escape at t≈{float(meta['events']['t_escape']):.3f}")
    if meta["events"]["collided"]:
        print(f"[event] collision at t≈{float(meta['events']['t_collision']):.3f}")
    if binary_flag:
        print(f"[info] binary formation detected (r<{args.r_bin}, hold≥{args.bin_hold}s)")

    return meta, csv_diag, (lyap_spec if lyap_spec is not None else [lyap_top] if lyap_top is not None else []), drift, theta_hist, planar, sim

# ========================== Analyze / Compare ==========================
def _pick_lam0(meta):
    if meta is None:
        return None
    if meta.get("lyap_top") is not None:
        return meta["lyap_top"]
    spec = meta.get("lyap_spectrum")
    return (spec[0] if spec else None)

def analyze_results(args, meta, csv_diag, lam, drift, theta_hist, planar, sim):
    max_drift = float(np.max(np.abs(drift)))
    drift_good = (max_drift < args.drift_threshold)

    dtheta = np.diff(theta_hist)
    mean_dtheta = float(np.mean(np.abs(dtheta)))
    theta_active = (mean_dtheta > 0.01)

    orbit_stable = bool(planar or (sim < 0.1))

    lam_top = (lam[0] if (lam and len(lam)>0) else None)
    lam_good = (lam_top is not None and lam_top < 0.5)

    success_count = sum([drift_good, theta_active, orbit_stable, lam_good])
    judgment = "Success" if success_count >= 3 else ("Partial Success" if success_count >= 2 else "Failure")

    lyap_str = f"{lam_top:.4f}" if (lam_top is not None) else "N/A"
    print(f"[Judgment] {judgment} (criteria met: {success_count}/4)")
    print(f"- max|drift| = {max_drift:.4g}  (threshold {args.drift_threshold}, pass={drift_good})")
    print(f"- mean|Δtheta| = {mean_dtheta:.4g}  (active={theta_active})")
    print(f"- orbit stable: {orbit_stable}  (planar={planar}, fig8_sim={sim:.4g})")
    print(f"- Lyap top: {lyap_str}  (good={lam_good})")
    return judgment

def compare_runs(args, csv_b, csv_d, meta_b, meta_d):
    b = pd.read_csv(csv_b); d = pd.read_csv(csv_d)
    b_max = float(b["drift"].abs().max()); d_max = float(d["drift"].abs().max())
    imp_drift = 100*(b_max - d_max)/max(1e-12, b_max)

    esc_b = 0 if meta_b["events"]["escaped"] else 1
    esc_d = 0 if meta_d["events"]["escaped"] else 1
    imp_escape = 100*(esc_d - esc_b)

    col_b = 0 if meta_b["events"]["collided"] else 1
    col_d = 0 if meta_d["events"]["collided"] else 1
    imp_collision = 100*(col_d - col_b)

    lam_b = _pick_lam0(meta_b)
    lam_d = _pick_lam0(meta_d)
    imp_lyap = 100*(lam_b - lam_d)/max(1e-12, lam_b) if (lam_b not in (None, 0.0) and lam_d is not None) else 0.0

    dtheta = np.diff(d["theta"])
    theta_active = float(np.mean(np.abs(dtheta))) > 0.01

    score = sum(v > 30 for v in [imp_drift, imp_escape, imp_collision, imp_lyap]) + int(theta_active)
    verdict = "PASS" if score >= 2 else "NO-GAIN"
    return dict(
        verdict=verdict,
        drift_improve=float(imp_drift),
        escape_improve=float(imp_escape),
        collision_improve=float(imp_collision),
        lyap_improve=float(imp_lyap),
        theta_active=bool(theta_active)
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
    ap.add_argument("--alpha_dtg", type=float, default=0.4)
    ap.add_argument("--beta_dtg", type=float, default=0.5)
    ap.add_argument("--b_dtg", type=float, default=0.0)
    ap.add_argument("--tau_theta", type=float, default=3.0)
    ap.add_argument("--theta_min", type=float, default=-1.5)
    ap.add_argument("--theta_max", type=float, default=1.5)
    ap.add_argument("--dtheta_max", type=float, default=0.05)
    ap.add_argument("--k_nu", type=float, default=0.0)
    ap.add_argument("--k_nu_ed", type=float, default=0.0)
    ap.add_argument("--k_ctrl", type=float, default=0.0)
    ap.add_argument("--k_ctrl_e", type=float, default=0.0)
    ap.add_argument("--lyap_const", type=float, default=0.0)
    ap.add_argument("--no_soften_when_clamped", action="store_true")

    # close-approach gates
    ap.add_argument("--nu_gate", type=float, default=0.0, help="close-approach viscosity boost (0=off)")
    ap.add_argument("--nu_gate_pow", type=float, default=2.0, help="power for viscosity gate")

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
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--drift_threshold", type=float, default=0.1)
    ap.add_argument("--nu_scan", type=float, nargs="*", default=[])

    args = ap.parse_args()

    if args.compare:
        results = []
        nu_list = ([args.nu0] + args.nu_scan) if args.nu_scan else [args.nu0]
        for nu in nu_list:
            b_args = argparse.Namespace(**vars(args))
            b_args.dtg = 0; b_args.nu0 = nu; b_args.out = os.path.join(args.out, f"baseline_nu{nu}")
            meta_b, _, _, _, _, _, _ = run_simulation(b_args)

            d_args = argparse.Namespace(**vars(args))
            d_args.dtg = 1; d_args.nu0 = nu; d_args.out = os.path.join(args.out, f"dtg_nu{nu}")
            meta_d, _, _, _, _, _, _ = run_simulation(d_args)

            res = compare_runs(args,
                               meta_b["outputs"]["csv_diagnostics"],
                               meta_d["outputs"]["csv_diagnostics"],
                               meta_b, meta_d)
            res.update(nu=nu,
                       lyap_b=_pick_lam0(meta_b), lyap_d=_pick_lam0(meta_d))
            results.append(res)
            print(f"[COMPARE nu={nu}] verdict={res['verdict']} | "
                  f"drift={res['drift_improve']:.1f}% | escape={res['escape_improve']:.1f}% | "
                  f"collision={res['collision_improve']:.1f}% | lyap={res['lyap_improve']:.1f}% | "
                  f"theta_active={res['theta_active']}")
        best = max(results, key=lambda r: sum(v for v in [
            r["drift_improve"], r["escape_improve"], r["collision_improve"], r["lyap_improve"]
        ]))
        print(f"[BEST] nu={best['nu']} | verdict={best['verdict']}")
        return

    meta, csv_diag, lam, drift, theta_hist, planar, sim = run_simulation(args)
    analyze_results(args, meta, csv_diag, lam, drift, theta_hist, planar, sim)

if __name__ == "__main__":
    main()