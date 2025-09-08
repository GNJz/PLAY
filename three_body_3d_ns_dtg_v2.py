#!/usr/bin/env python3
# three_body_3d_ns_dtg_v2.py
# 3체(3D) + 항력(ν) + 난류 + DTG(연속시간) + 안정화 보강판
# - θ clamp + slew-rate 제한
# - ν = ν0 + k_nu*θ_dev + k_nu_ed*energy_dev  (에너지 오차에 직접 연동)
# - 제어가속도 스케일링: k_ctrl_eff = k_ctrl * (1 + k_ctrl_e*|energy_dev|)
# - softening ε 노출
# - integrator 선택: DOP853 | LEAPFROG
# - 초기 COM/총운동량 제로화 옵션
# - provenance/메타 강화

import argparse, os, json, time, math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- 글로벌 설정 ----------------
DEFAULT_EPS = 1e-12  # 기본 softening (필요시 CLI로 덮어씀)

# ---------------- 상태 유틸 ----------------
def unpack_state_bodies(s, N=3):
    s = np.asarray(s, float).reshape(-1)
    if s.size != 6 * N:
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

# ---------------- 물리량 ----------------
# --- 빠른 중력/에너지 (drop-in) ---
def accelerations(pos, vel, G, masses, eps=1e-12,
                  nu_eff=0.0, turb_coeff=0.0, k_ctrl_eff=0.0, theta_dev=0.0):
    # pos, vel: (3,N)
    # pairwise r
    dr = pos[:, None, :] - pos[:, :, None]             # (3,N,N)
    r2 = np.sum(dr*dr, axis=0) + eps*eps               # (N,N)
    msk = ~np.eye(r2.shape[0], dtype=bool)
    inv_r3 = np.where(msk, r2**-1.5, 0.0)              # diag=0 처리
    w = masses[None, :] * inv_r3                        # (1,N)*(N,N)->(N,N)
    grav = G * np.einsum('kij,ij->ki', dr, w)          # (3,N)

    # linear drag
    visc = -nu_eff * vel

    # synthetic turbulence (옵션)
    if turb_coeff:
        turb = turb_coeff * np.sin(np.sum(pos*pos, axis=0))[None, :] * vel
    else:
        turb = 0.0

    # small feedback
    ctrl = -k_ctrl_eff * theta_dev * vel
    return grav + visc + turb + ctrl


def total_energy_body(s_body, G=1.0, masses=(1.0,1.0,1.0), eps=1e-12):
    pos, vel = unpack_state_bodies(s_body, N=3)
    K = 0.5 * np.sum(masses * np.sum(vel*vel, axis=0))
    dr = pos[:, None, :] - pos[:, :, None]
    r = np.sqrt(np.sum(dr*dr, axis=0) + eps*eps)
    iu = np.triu_indices(len(masses), 1)
    U = -G * np.sum(masses[iu[0]] * masses[iu[1]] / r[iu])
    return K + U

# ---------------- 초기조건/정규화 ----------------
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
    # COM shift
    com = (pos * m).sum(axis=1) / m.sum()
    pos = pos - com.reshape(3,1)
    # total momentum zero
    p = (vel * m).sum(axis=1, keepdims=True)    # (3,1)
    vel = vel - p / m.sum()
    return pos, vel

# ---------------- 진단 ----------------
def positions_from_sol_bodyY(Y, N=3):
    rows = np.arange(N)
    x_rows = 3*rows + 0
    y_rows = 3*rows + 1
    z_rows = 3*rows + 2
    x = Y[x_rows, :]
    y = Y[y_rows, :]
    z = Y[z_rows, :]
    return x, y, z

def diag_com_momentum(s_body, masses):
    pos, vel = unpack_state_bodies(s_body, N=3)
    m = masses.reshape(1, -1)
    com = (pos * m).sum(axis=1) / m.sum()
    p = (vel * m).sum(axis=1)
    return float(np.linalg.norm(com)), float(np.linalg.norm(p))

def diag_planarity(z, tol=1e-3):
    zmax = float(np.max(np.abs(z)))
    return bool(zmax < tol), zmax

def diag_fig8_similarity(x, y):
    curves = [np.vstack([x[i], y[i]]).T for i in range(3)]
    def pair_score(a, b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        d = np.linalg.norm(a - b, axis=1)
        return float(np.std(d))
    return pair_score(curves[0], curves[1]) + pair_score(curves[1], curves[2]) + pair_score(curves[0], curves[2])

# ---------------- Lyapunov (Benettin, 보존계) ----------------
def rhs_body_conservative(t, s_body, args_rhs):
    G, masses, eps = args_rhs["G"], args_rhs["masses"], args_rhs["eps"]
    pos, vel = unpack_state_bodies(s_body, N=3)
    acc = accelerations(pos, vel, G, masses, eps=eps,
                        nu_eff=0.0, turb_coeff=0.0, k_ctrl_eff=0.0, theta_dev=0.0)
    return pack_state_bodies(vel, acc)

def lyapunov_benettin(s0_body, rhs_body, tmax=12.0, dt=0.002, delta0=1e-8,
                      renorm_every=50, args_rhs=None):
    rng = np.random.default_rng(0)
    v = rng.normal(size=s0_body.size); v /= np.linalg.norm(v)

    t = 0.0
    s = s0_body.copy()
    s_pert = s + delta0 * v
    d0 = delta0
    lam_sum = 0.0
    n_renorm = 0

    while t < tmax - 1e-12:
        t_next = min(t + renorm_every * dt, tmax)
        sol1 = solve_ivp(lambda tt, ss: rhs_body(tt, ss, args_rhs),
                         (t, t_next), s, method="DOP853",
                         rtol=1e-9, atol=1e-12, max_step=dt)
        sol2 = solve_ivp(lambda tt, ss: rhs_body(tt, ss, args_rhs),
                         (t, t_next), s_pert, method="DOP853",
                         rtol=1e-9, atol=1e-12, max_step=dt)
        s = sol1.y[:, -1]
        s_pert = sol2.y[:, -1]
        d = np.linalg.norm(s_pert - s)
        if d <= 0: d = 1e-300
        lam_sum += np.log(d / d0); n_renorm += 1
        v = (s_pert - s) / d
        s_pert = s + d0 * v
        t = t_next

    T = n_renorm * renorm_every * dt
    return lam_sum / max(T, 1e-30)

# ---------------- 연속 DTG RHS ----------------
class RHSParams:
    def __init__(self, G, masses, eps,
                 nu0=0.0, nu_max=1.0, turb_coeff=0.0,
                 # DTG
                 theta0=1.0, alpha_dtg=0.7, beta_dtg=0.5, b_dtg=0.0, tau_theta=1.0,
                 theta_min=-1.5, theta_max=1.5, dtheta_max=0.05,
                 k_nu=0.0, k_nu_ed=0.0,    # θ/에너지-연동 ν 이득
                 k_ctrl=0.0, k_ctrl_e=0.0, # 제어가속도 이득 (에너지 가중)
                 lyap_const=0.0, E0=None):
        self.G, self.masses, self.eps = G, np.asarray(masses, float), float(eps)
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

# --- 더 가벼운 DTG RHS (slew/clamp 한 번에) ---
def rhs_with_dtg(t, s_all, prm):
    pos, vel, theta = split_state_with_theta(s_all, N=3)

    # energy dev (필요할 때만)
    if prm.E0 is not None:
        E = total_energy_body(pack_state_bodies(pos, vel), prm.G, prm.masses, eps=prm.eps)
        ed = (E - prm.E0) / (abs(prm.E0) + 1e-15)
    else:
        ed = 0.0

    # θ: target→미분→slew→다음 스텝 clamp
    target = prm.b_dtg + prm.alpha_dtg*ed - prm.beta_dtg*prm.lyap_const
    dtheta = np.clip((target - theta)/max(prm.tau_theta, 1e-9), -prm.dtheta_max, prm.dtheta_max)
    # nu / control 이득
    theta_dev = np.clip(theta - prm.theta0, prm.theta_min-prm.theta0, prm.theta_max-prm.theta0)
    nu_eff = np.clip(prm.nu0 + prm.k_nu*theta_dev + prm.k_nu_ed*ed, 0.0, prm.nu_max)
    k_ctrl_eff = prm.k_ctrl * (1.0 + prm.k_ctrl_e*abs(ed))

    acc = accelerations(pos, vel, prm.G, prm.masses, eps=prm.eps,
                        nu_eff=nu_eff, turb_coeff=prm.turb_coeff,
                        k_ctrl_eff=k_ctrl_eff, theta_dev=theta_dev)
    return np.concatenate([pack_state_bodies(vel, acc), np.array([dtheta])])

    acc = accelerations(pos, vel, prm.G, prm.masses, eps=prm.eps,
                        nu_eff=nu_eff,
                        turb_coeff=prm.turb_coeff,
                        k_ctrl_eff=k_ctrl_eff,
                        theta_dev=theta_dev)

    dpos = vel
    dvel = acc
    return np.concatenate([pack_state_bodies(dpos, dvel), np.array([dtheta])])

# ---------------- Leapfrog(옵션) ----------------
def integrate_leapfrog(s0_all, t_eval, prm: RHSParams):
    """
    간단한 explicit leapfrog 변형.
    - drag/ctrl이 v에 의존하므로 v-half 예측자 사용
    """
    N = 3
    pos, vel, theta = split_state_with_theta(s0_all, N=N)
    Y = np.zeros((6*N+1, len(t_eval)))
    Y[:,0] = s0_all
    for i in range(1, len(t_eval)):
        dt = t_eval[i] - t_eval[i-1]
        # 에너지 편차
        s_body = pack_state_bodies(pos, vel)
        if prm.E0 is not None:
            E = total_energy_body(s_body, prm.G, prm.masses, eps=prm.eps)
            energy_dev = (E - prm.E0) / (abs(prm.E0) + 1e-15)
        else:
            energy_dev = 0.0

        # θ 미분 + 제한
        target = prm.b_dtg + prm.alpha_dtg*energy_dev - prm.beta_dtg*prm.lyap_const
        dtheta_raw = (target - theta) / max(prm.tau_theta, 1e-9)
        dtheta = float(np.clip(dtheta_raw, -prm.dtheta_max, prm.dtheta_max))
        theta = float(np.clip(theta + dtheta*dt, prm.theta_min, prm.theta_max))

        # ν, 제어이득
        theta_dev = (theta - prm.theta0)
        nu_eff = prm.nu0 + prm.k_nu*theta_dev + prm.k_nu_ed*energy_dev
        nu_eff = float(np.clip(nu_eff, 0.0, prm.nu_max))
        k_ctrl_eff = prm.k_ctrl * (1.0 + prm.k_ctrl_e * abs(energy_dev))

        # v_half
        acc_now = accelerations(pos, vel, prm.G, prm.masses, eps=prm.eps,
                                nu_eff=nu_eff, turb_coeff=prm.turb_coeff,
                                k_ctrl_eff=k_ctrl_eff, theta_dev=theta_dev)
        v_half = vel + 0.5*dt*acc_now
        # x_{n+1}
        pos_new = pos + dt*v_half
        # acc at new
        acc_new = accelerations(pos_new, v_half, prm.G, prm.masses, eps=prm.eps,
                                nu_eff=nu_eff, turb_coeff=prm.turb_coeff,
                                k_ctrl_eff=k_ctrl_eff, theta_dev=theta_dev)
        # v_{n+1}
        vel_new = v_half + 0.5*dt*acc_new

        pos, vel = pos_new, vel_new
        Y[:, i] = np.concatenate([pack_state_bodies(pos, vel), np.array([theta])])
    return Y

# ---------------- 실행 ----------------
def run(args):
    os.makedirs(args.out, exist_ok=True)
    for sub in ["figures","data","meta"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    masses = np.array([1.0,1.0,1.0], float)
    pos0, vel0 = make_ic(args.ic, args.alpha)

    if args.zero_init:
        pos0, vel0 = zero_com_and_momentum(pos0, vel0, masses)

    s0_body = pack_state_bodies(pos0, vel0)
    E0 = total_energy_body(s0_body, args.G, masses, eps=args.eps)

    prm = RHSParams(
        G=args.G, masses=masses, eps=args.eps,
        nu0=args.nu0, nu_max=args.nu_max, turb_coeff=args.turb,
        theta0=args.theta0, alpha_dtg=args.alpha_dtg, beta_dtg=args.beta_dtg,
        b_dtg=args.b_dtg, tau_theta=args.tau_theta,
        theta_min=args.theta_min, theta_max=args.theta_max, dtheta_max=args.dtheta_max,
        k_nu=args.k_nu, k_nu_ed=args.k_nu_ed,
        k_ctrl=args.k_ctrl, k_ctrl_e=args.k_ctrl_e,
        lyap_const=args.lyap_const, E0=(E0 if args.dtg else None)
    )

    t_eval = np.arange(0.0, args.tmax + 1e-12, args.dt)
    s0_all = np.concatenate([s0_body, np.array([args.theta0])])

    if args.scheme.lower() == "leapfrog":
        Y = integrate_leapfrog(s0_all, t_eval, prm)
        sol_t = t_eval
    else:
        sol = solve_ivp(lambda t, s: rhs_with_dtg(t, s, prm),
                        (0.0, args.tmax), s0_all, t_eval=t_eval,
                        method="DOP853", rtol=1e-9, atol=1e-12,
                        max_step=args.dt)
        Y, sol_t = sol.y, sol.t

    bodyY = Y[:-1, :]
    theta_hist = Y[-1, :]

    # 좌표
    x, y, z = positions_from_sol_bodyY(bodyY, N=3)
    x1,y1,z1 = x[0], y[0], z[0]
    x2,y2,z2 = x[1], y[1], z[1]
    x3,y3,z3 = x[2], y[2], z[2]

    # CSV
    df = pd.DataFrame({"t": sol_t,
                       "x1": x1,"y1": y1,"z1": z1,
                       "x2": x2,"y2": y2,"z2": z2,
                       "x3": x3,"y3": y3,"z3": z3,
                       "theta": theta_hist})
    csv_path = os.path.join(args.out, "data", f"threebody3d_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.csv")
    df.to_csv(csv_path, index=False)

    # 그림들
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x1,y1,z1,label="Body 1")
    ax.plot(x2,y2,z2,label="Body 2")
    ax.plot(x3,y3,z3,label="Body 3")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"3D Three-Body (ic={args.ic}, α={args.alpha}, dtg={args.dtg}, "
                 f"nu0={args.nu0}, turb={args.turb}, scheme={args.scheme})")
    ax.legend()
    try:
        ax.set_box_aspect((1,1,1)); ax.set_proj_type('ortho')
    except Exception:
        pass
    ax.view_init(elev=args.elev, azim=args.azim)
    fig3d = os.path.join(args.out, "figures", f"threebody3d_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
    fig.savefig(fig3d, dpi=160); plt.close(fig)

    plt.figure(figsize=(6,5))
    for xi, yi, lab in [(x1,y1,"Body 1"), (x2,y2,"Body 2"), (x3,y3,"Body 3")]:
        plt.plot(xi, yi, label=lab)
    plt.xlabel("X"); plt.ylabel("Y"); plt.axis('equal'); plt.legend()
    plt.title(f"XY Projection (ic={args.ic}, α={args.alpha}, dtg={args.dtg})")
    fig_xy = os.path.join(args.out, "figures", f"xy_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
    plt.savefig(fig_xy, dpi=160); plt.close()

    # 에너지 드리프트(참고)
    E_series = np.array([total_energy_body(bodyY[:,i], args.G, masses, eps=args.eps)
                         for i in range(bodyY.shape[1])])
    drift = (E_series - E0) / (abs(E0) + 1e-15)
    plt.figure(figsize=(7,4))
    plt.plot(sol_t, drift)
    plt.xlabel("Time"); plt.ylabel("Relative Energy Drift")
    plt.title("Total Energy Drift (reference)")
    fig_drift = os.path.join(args.out, "figures", f"energy_drift_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
    plt.savefig(fig_drift, dpi=160); plt.close()

    plt.figure(figsize=(7,4))
    plt.plot(sol_t, theta_hist)
    plt.xlabel("Time"); plt.ylabel("Theta (DTG)")
    plt.title("DTG Theta")
    fig_theta = os.path.join(args.out, "figures", f"theta_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.png")
    plt.savefig(fig_theta, dpi=160); plt.close()

    # 진단
    com0, p0 = diag_com_momentum(bodyY[:,0], masses)
    comT, pT = diag_com_momentum(bodyY[:,-1], masses)
    planar, zmax = diag_planarity(z, tol=1e-3)
    sim = diag_fig8_similarity(x, y)

    # 메타
    meta = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        args=vars(args),
        E0=E0,
        outputs=dict(csv=csv_path, fig3d=fig3d, fig_xy=fig_xy,
                     drift=fig_drift, theta=fig_theta),
        diagnostics=dict(COM_t0=com0, P_t0=p0, COM_tT=comT, P_tT=pT,
                         planar=planar, zmax=zmax, fig8_similarity=sim),
    )
    with open(os.path.join(args.out, "meta", f"meta_{args.ic}_a{args.alpha}_dtg{int(args.dtg)}.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] CSV   : {csv_path}")
    print(f"[OK] FIG3D : {fig3d}")
    print(f"[OK] XY    : {fig_xy}")
    print(f"[OK] DRIFT : {fig_drift}")
    print(f"[OK] THETA : {fig_theta}")
    print(f"[diag] |COM(t0)|={com0:.3e}, |P(t0)|={p0:.3e} |COM(tT)|={comT:.3e}, |P(tT)|={pT:.3e}")
    print(f"[diag] planar={planar} (max|z|={zmax:.3e}), fig8_similarity={sim:.3e}")

    return sol_t, drift, theta_hist

# ---------------- 메인 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", choices=["exp1","exp2","exp3","figure8"], default="exp2")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tmax", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--out", default="runs/out")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--scheme", choices=["dop853","leapfrog"], default="dop853")
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    ap.add_argument("--G", type=float, default=1.0)

    # 초기 정규화
    ap.add_argument("--zero_init", action="store_true", help="초기 COM/총운동량 제로화")

    # 항력/난류
    ap.add_argument("--nu0", type=float, default=0.0)
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
    ap.add_argument("--theta_max", type=float, default= 1.5)
    ap.add_argument("--dtheta_max", type=float, default=0.05)

    # ν/제어 이득
    ap.add_argument("--k_nu", type=float, default=0.0)
    ap.add_argument("--k_nu_ed", type=float, default=0.0)
    ap.add_argument("--k_ctrl", type=float, default=0.0)
    ap.add_argument("--k_ctrl_e", type=float, default=0.0)

    # Lyapunov (외부 추정치 사용 또는 내부 근사)
    ap.add_argument("--lyap_const", type=float, default=0.0)
    ap.add_argument("--lyap", action="store_true")

    args = ap.parse_args()

    # (선택) Lyapunov 근사
    if args.lyap:
        pos0, vel0 = make_ic(args.ic, args.alpha)
        s0_body = pack_state_bodies(pos0, vel0)
        lam_est = lyapunov_benettin(
            s0_body, rhs_body_conservative,
            tmax=max(6.0, args.tmax), dt=args.dt,
            args_rhs=dict(G=args.G, masses=np.array([1.0,1.0,1.0]), eps=args.eps)
        )
        args.lyap_const = float(lam_est)
        print(f"[Lyapunov_Benettin ≈] {args.lyap_const:.6f} (주기해면 ~0, 양수면 혼돈 경향)")

    run(args)

if __name__ == "__main__":
    main()