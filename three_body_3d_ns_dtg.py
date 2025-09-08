#!/usr/bin/env python3
# three_body_3d_ns_dtg.py
# 3체(3D) + 항력(ν) + (옵션) 인공 난류 섭동 + DTG(연속시간) 피드백 제어
# - 보존계 검증 가능(ν=0, turb=0, k_ctrl=0)
# - DTG를 θ 상태로 포함(dθ/dt = ((b + α*energy_dev - β*lyap) - θ)/τθ)
# - ν_eff, a_ctrl(제어가속도)로 동역학에 반영

import argparse, os, json
import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12

# ---------------- 상태 유틸 ----------------
def unpack_state_bodies(s, N=3):
    """s_body: 길이 6N (x,y,z, vx,vy,vz)"""
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

def pack_state_with_theta(pos, vel, theta):
    body = pack_state_bodies(pos, vel)
    return np.concatenate([body, np.array([float(theta)], float)])

def split_state_with_theta(s, N=3):
    s = np.asarray(s, float).reshape(-1)
    body, theta = s[:-1], float(s[-1])
    pos, vel = unpack_state_bodies(body, N=N)
    return pos, vel, theta

# ---------------- 물리량 ----------------
def accelerations(pos, vel, G, masses, nu_eff=0.0, turb_coeff=0.0, k_ctrl=0.0, theta_dev=0.0):
    """
    grav_acc: Newtonian gravity
    visc_acc: linear drag  (-nu_eff * v)
    turb_acc: artificial perturbation (for experiments)  turb_coeff * sin(|r|^2) * v
    ctrl_acc: simple feedback control  (-k_ctrl * theta_dev * v)
    """
    # Gravity
    dr = pos[:, None, :] - pos[:, :, None]                 # (3, N, N): r_i - r_j
    r2 = np.sum(dr * dr, axis=0) + EPS
    np.fill_diagonal(r2, np.inf)
    inv_r3 = r2 ** (-1.5)
    w = masses[None, :] * inv_r3                            # (1,N)*(N,N)->(N,N)
    grav_acc = G * np.einsum("kij,ij->ki", dr, w)           # (3,N)

    visc_acc = -nu_eff * vel                                # (3,N)

    if turb_coeff != 0.0:
        r2sum = np.sum(pos * pos, axis=0)                   # (N,)
        turb_acc = turb_coeff * np.sin(r2sum)[None, :] * vel
    else:
        turb_acc = 0.0

    ctrl_acc = -k_ctrl * theta_dev * vel

    return grav_acc + visc_acc + turb_acc + ctrl_acc

def total_energy_body(s_body, G=1.0, masses=(1.0,1.0,1.0)):
    m = np.asarray(masses, float)
    pos, vel = unpack_state_bodies(s_body, N=3)
    v2 = np.sum(vel * vel, axis=0)
    K = 0.5 * np.sum(m * v2)
    dr = pos[:, None, :] - pos[:, :, None]
    r = np.sqrt(np.sum(dr*dr, axis=0) + EPS)
    iu = np.triu_indices(m.size, k=1)
    U = -G * np.sum(m[iu[0]] * m[iu[1]] / r[iu])
    return K + U

# ---------------- 초기조건 ----------------
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
    return pack_state_bodies(pos, vel)

# ---------------- 좌표 추출 ----------------
def positions_from_body_state(s_body, N=3):
    Y = np.asarray(s_body, float).reshape(-1)
    # 이 함수는 solve_ivp의 Y행렬 용이 아님 — 아래에서 별도 처리
    raise NotImplementedError

def positions_from_sol_bodyY(Y, N=3):
    """
    Y: shape (6N, T) for bodies (without theta)
    return x(3,N,T) -> we provide arrays for each body
    """
    rows = np.arange(N)
    x_rows = 3*rows + 0
    y_rows = 3*rows + 1
    z_rows = 3*rows + 2
    x = Y[x_rows, :]
    y = Y[y_rows, :]
    z = Y[z_rows, :]
    return x, y, z

# ---------------- 진단 ----------------
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

# ---------------- Lyapunov (Benettin) ----------------
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
                         (t, t_next), s,
                         method="DOP853", rtol=1e-9, atol=1e-12, max_step=dt)
        sol2 = solve_ivp(lambda tt, ss: rhs_body(tt, ss, args_rhs),
                         (t, t_next), s_pert,
                         method="DOP853", rtol=1e-9, atol=1e-12, max_step=dt)

        s = sol1.y[:, -1]
        s_pert = sol2.y[:, -1]

        d = np.linalg.norm(s_pert - s)
        if d <= 0:
            d = 1e-300
        lam_sum += np.log(d / d0)
        n_renorm += 1

        v = (s_pert - s) / d
        s_pert = s + d0 * v

        t = t_next

    T = n_renorm * renorm_every * dt
    return lam_sum / max(T, 1e-30)

# ---------------- DTG 연동 RHS ----------------
class RHSParams:
    def __init__(self, G, masses,
                 nu0=0.0, nu_max=1.0,
                 turb_coeff=0.0,
                 # DTG
                 theta0=1.0, alpha_dtg=0.7, beta_dtg=0.5, b_dtg=0.0, tau_theta=1.0,
                 k_nu=0.0, k_ctrl=0.0,
                 lyap_const=0.0,
                 E0=None):
        self.G = G
        self.masses = np.asarray(masses, float)
        self.nu0 = float(nu0)
        self.nu_max = float(nu_max)
        self.turb_coeff = float(turb_coeff)
        self.theta0 = float(theta0)
        self.alpha_dtg = float(alpha_dtg)
        self.beta_dtg = float(beta_dtg)
        self.b_dtg = float(b_dtg)
        self.tau_theta = float(tau_theta)
        self.k_nu = float(k_nu)
        self.k_ctrl = float(k_ctrl)
        self.lyap_const = float(lyap_const)
        self.E0 = float(E0) if E0 is not None else None

def rhs_with_dtg(t, s_all, prm: RHSParams):
    """
    s_all = [ bodies(6N), theta ]
    d/dt bodies = [v, a_total]
    d/dt theta  = ( (b + α*energy_dev - β*lyap) - θ ) / τθ
    nu_eff = clip(nu0 + k_nu*(θ-θ0), 0, nu_max)
    a_ctrl = -k_ctrl*(θ-θ0)*v
    """
    pos, vel, theta = split_state_with_theta(s_all, N=3)

    # 에너지 편차(보존계 기준): E0가 없으면 0으로 취급
    s_body = pack_state_bodies(pos, vel)
    if prm.E0 is not None:
        E = total_energy_body(s_body, prm.G, prm.masses)
        energy_dev = (E - prm.E0) / (abs(prm.E0) + 1e-15)
    else:
        energy_dev = 0.0

    # DTG 연속 업데이트
    target = prm.b_dtg + prm.alpha_dtg * energy_dev - prm.beta_dtg * prm.lyap_const
    dtheta = (target - theta) / max(prm.tau_theta, 1e-9)

    # θ 기반 파생량
    theta_dev = (theta - prm.theta0)
    nu_eff = prm.nu0 + prm.k_nu * theta_dev
    nu_eff = float(np.clip(nu_eff, 0.0, prm.nu_max))

    acc = accelerations(pos, vel, prm.G, prm.masses,
                        nu_eff=nu_eff,
                        turb_coeff=prm.turb_coeff,
                        k_ctrl=prm.k_ctrl,
                        theta_dev=theta_dev)

    dpos = vel
    dvel = acc
    return np.concatenate([pack_state_bodies(dpos, dvel), np.array([dtheta])])

# 보존계(몸체만) RHS (Lyapunov용)
def rhs_body_conservative(t, s_body, args_rhs):
    G, masses = args_rhs["G"], args_rhs["masses"]
    pos, vel = unpack_state_bodies(s_body, N=3)
    acc = accelerations(pos, vel, G, masses, nu_eff=0.0, turb_coeff=0.0, k_ctrl=0.0, theta_dev=0.0)
    return pack_state_bodies(vel, acc)

# ---------------- 실행 ----------------
def run(ic_mode, alpha, t_max, dt, out_root,
        elev=20, azim=-60,
        # 베이스 파라미터
        G=1.0, masses=(1.0,1.0,1.0),
        # 난류/항력
        nu0=0.0, nu_max=1.0, turb_coeff=0.0,
        # DTG
        use_dtg=True, theta0=1.0, alpha_dtg=0.7, beta_dtg=0.5, b_dtg=0.0, tau_theta=1.0,
        k_nu=0.0, k_ctrl=0.0,
        # Lyapunov(상수로 사용)
        lyap_const=0.0,
        max_step=None):

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "data"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "meta"), exist_ok=True)

    masses = np.asarray(masses, float)
    s0_body = make_ic(ic_mode, alpha)
    E0 = total_energy_body(s0_body, G, masses)

    # 초기 θ
    theta_init = float(theta0)

    # 파라미터 객체
    prm = RHSParams(G=G, masses=masses,
                    nu0=nu0, nu_max=nu_max, turb_coeff=turb_coeff,
                    theta0=theta0, alpha_dtg=alpha_dtg, beta_dtg=beta_dtg, b_dtg=b_dtg, tau_theta=tau_theta,
                    k_nu=k_nu, k_ctrl=k_ctrl,
                    lyap_const=lyap_const,
                    E0=(E0 if use_dtg else None))

    # 적분
    t_eval = np.arange(0.0, t_max + 1e-12, dt)
    s0_all = np.concatenate([s0_body, np.array([theta_init])])
    sol = solve_ivp(lambda t, s: rhs_with_dtg(t, s, prm),
                    (0.0, t_max), s0_all, t_eval=t_eval,
                    method="DOP853", rtol=1e-9, atol=1e-12,
                    max_step=(dt if max_step is None else max_step))

    # 분해
    Y = sol.y   # shape: (6N+1, T)
    bodyY = Y[:-1, :]
    theta_hist = Y[-1, :]

    # 좌표/그림
    x, y, z = positions_from_sol_bodyY(bodyY, N=3)
    x1,y1,z1 = x[0], y[0], z[0]
    x2,y2,z2 = x[1], y[1], z[1]
    x3,y3,z3 = x[2], y[2], z[2]

    # CSV
    df = pd.DataFrame({"t": sol.t,
                       "x1": x1,"y1": y1,"z1": z1,
                       "x2": x2,"y2": y2,"z2": z2,
                       "x3": x3,"y3": y3,"z3": z3,
                       "theta": theta_hist})
    csv_path = os.path.join(out_root, "data", f"threebody3d_{ic_mode}_a{alpha}_dtg{int(use_dtg)}.csv")
    df.to_csv(csv_path, index=False)

    # 3D 궤적
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x1,y1,z1,label="Body 1")
    ax.plot(x2,y2,z2,label="Body 2")
    ax.plot(x3,y3,z3,label="Body 3")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"3D Three-Body (ic={ic_mode}, α={alpha}, dtg={use_dtg}, nu0={nu0}, turb={turb_coeff})")
    ax.legend()
    try:
        ax.set_box_aspect((1,1,1))
        ax.set_proj_type('ortho')
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    fig_path = os.path.join(out_root, "figures", f"threebody3d_{ic_mode}_a{alpha}_dtg{int(use_dtg)}.png")
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    # XY 투영
    plt.figure(figsize=(6,5))
    for xi, yi, lab in [(x1,y1,"Body 1"), (x2,y2,"Body 2"), (x3,y3,"Body 3")]:
        plt.plot(xi, yi, label=lab)
    plt.xlabel("X"); plt.ylabel("Y"); plt.axis('equal'); plt.legend()
    plt.title(f"XY Projection (ic={ic_mode}, α={alpha}, dtg={use_dtg})")
    fig_xy_path = os.path.join(out_root, "figures", f"xy_{ic_mode}_a{alpha}_dtg{int(use_dtg)}.png")
    plt.savefig(fig_xy_path, dpi=160)
    plt.close()

    # 에너지 드리프트(참고용: 항력/제어 있으면 보존X)
    E_series = np.array([total_energy_body(bodyY[:,i], G, masses) for i in range(bodyY.shape[1])])
    drift = (E_series - E0) / (abs(E0) + 1e-15)
    plt.figure(figsize=(7,4))
    plt.plot(sol.t, drift)
    plt.xlabel("Time"); plt.ylabel("Relative Energy Drift")
    plt.title("Total Energy Drift (reference)")
    drift_path = os.path.join(out_root, "figures", f"energy_drift_{ic_mode}_a{alpha}_dtg{int(use_dtg)}.png")
    plt.savefig(drift_path, dpi=160)
    plt.close()

    # θ 히스토리
    plt.figure(figsize=(7,4))
    plt.plot(sol.t, theta_hist)
    plt.xlabel("Time"); plt.ylabel("Theta (DTG)")
    plt.title("DTG Theta")
    theta_path = os.path.join(out_root, "figures", f"theta_{ic_mode}_a{alpha}_dtg{int(use_dtg)}.png")
    plt.savefig(theta_path, dpi=160)
    plt.close()

    # 진단
    com0, p0 = diag_com_momentum(bodyY[:,0], masses)
    comT, pT = diag_com_momentum(bodyY[:,-1], masses)
    planar, zmax = diag_planarity(z, tol=1e-3)
    sim = diag_fig8_similarity(x, y)

    # 메타정보
    meta = dict(
        ic_mode=ic_mode, alpha=alpha, t_max=t_max, dt=dt,
        G=G, masses=masses.tolist(),
        nu0=nu0, nu_max=nu_max, turb_coeff=turb_coeff,
        use_dtg=bool(use_dtg), theta0=theta0,
        alpha_dtg=alpha_dtg, beta_dtg=beta_dtg, b_dtg=b_dtg, tau_theta=tau_theta,
        k_nu=k_nu, k_ctrl=k_ctrl,
        lyap_const=lyap_const,
        E0=E0,
        outputs=dict(csv=csv_path, fig3d=fig_path, fig_xy=fig_xy_path,
                     drift=drift_path, theta=theta_path),
        diagnostics=dict(
            COM_t0=com0, P_t0=p0, COM_tT=comT, P_tT=pT,
            planar=planar, zmax=zmax, fig8_similarity=sim
        )
    )
    with open(os.path.join(out_root, "meta", f"meta_{ic_mode}_a{alpha}_dtg{int(use_dtg)}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] CSV   : {csv_path}")
    print(f"[OK] FIG3D : {fig_path}")
    print(f"[OK] XY    : {fig_xy_path}")
    print(f"[OK] DRIFT : {drift_path}")
    print(f"[OK] THETA : {theta_path}")
    print(f"[diag] |COM(t0)|={com0:.3e}, |P(t0)|={p0:.3e} |COM(tT)|={comT:.3e}, |P(tT)|={pT:.3e}")
    print(f"[diag] planar={planar} (max|z|={zmax:.3e}), fig8_similarity={sim:.3e}")

    return sol.t, drift, theta_hist

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

    # 항력/난류
    ap.add_argument("--nu0", type=float, default=0.0)
    ap.add_argument("--nu_max", type=float, default=1.0)
    ap.add_argument("--turb", type=float, default=0.0)

    # DTG
    ap.add_argument("--dtg", type=int, default=1)              # 1:on, 0:off
    ap.add_argument("--theta0", type=float, default=1.0)
    ap.add_argument("--alpha_dtg", type=float, default=0.7)
    ap.add_argument("--beta_dtg", type=float, default=0.5)
    ap.add_argument("--b_dtg", type=float, default=0.0)
    ap.add_argument("--tau_theta", type=float, default=1.0)
    ap.add_argument("--k_nu", type=float, default=0.0)         # θ→ν 연동 이득
    ap.add_argument("--k_ctrl", type=float, default=0.0)       # θ→제어가속도 이득
    ap.add_argument("--lyap_const", type=float, default=0.0)   # 외부 추정값 사용

    ap.add_argument("--lyap", action="store_true")             # 보존계에서 대략 추정
    ap.add_argument("--max_step", type=float, default=None)

    args = ap.parse_args()

    # (선택) Lyapunov 추정: 보존계(ν=0, turb=0, 제어=0)에서 근사
    lam_est = float(args.lyap_const)
    if args.lyap:
        s0_body = make_ic(args.ic, args.alpha)
        lam_est = lyapunov_benettin(
            s0_body,
            rhs_body_conservative,
            tmax=max(6.0, args.tmax),
            dt=args.dt,
            args_rhs=dict(G=1.0, masses=np.array([1.0,1.0,1.0]))
        )
        print(f"[Lyapunov_Benettin ≈] {lam_est:.6f} (주기해면 ~0, 양수면 혼돈 경향)")

    run(
        ic_mode=args.ic, alpha=args.alpha,
        t_max=args.tmax, dt=args.dt, out_root=args.out,
        elev=args.elev, azim=args.azim,
        G=1.0, masses=(1.0,1.0,1.0),
        nu0=args.nu0, nu_max=args.nu_max, turb_coeff=args.turb,
        use_dtg=bool(args.dtg),
        theta0=args.theta0, alpha_dtg=args.alpha_dtg, beta_dtg=args.beta_dtg,
        b_dtg=args.b_dtg, tau_theta=args.tau_theta,
        k_nu=args.k_nu, k_ctrl=args.k_ctrl,
        lyap_const=lam_est,
        max_step=(args.dt if args.max_step is None else args.max_step)
    )

if __name__ == "__main__":
    main()