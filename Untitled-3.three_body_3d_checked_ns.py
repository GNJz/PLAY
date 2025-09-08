#!/usr/bin/env python3
# three_body_3d_checked_ns.py
# - 벡터화 3체 시뮬레이터(3D) + Navier-Stokes 요소
# - 오해 방지용 플롯 보정, XY투영 저장
# - 수치 검증용 진단, Benettin Lyapunov
# - DTG로 난류 안정성 제어

import argparse, os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12

# ---------------- 상태 관리 ----------------
def unpack_state(s, N=3):
    s = np.asarray(s, float).reshape(-1)
    if s.size != 6 * N:
        raise ValueError(f"state length must be 6N (= {6*N}), got {s.size}")
    pos = s[:3*N].reshape(3, N, order="F")
    vel = s[3*N:6*N].reshape(3, N, order="F")
    return pos, vel

def pack_state(pos, vel):
    pos = np.asarray(pos, float)
    vel = np.asarray(vel, float)
    if pos.shape != vel.shape or pos.shape[0] != 3:
        raise ValueError("pos/vel must have shape (3,N)")
    N = pos.shape[1]
    return np.concatenate([pos.reshape(3*N, order="F"),
                           vel.reshape(3*N, order="F")])

# ---------------- 물리 코어 ----------------
def accelerations(pos, vel, G, masses, nu=0.01, eps=EPS):
    # 중력 가속도
    dr = pos[:, None, :] - pos[:, :, None]
    r2 = np.sum(dr * dr, axis=0) + eps
    np.fill_diagonal(r2, np.inf)
    inv_r3 = r2 ** (-1.5)
    w = masses[None, :] * inv_r3
    grav_acc = G * np.einsum("kij,ij->ki", dr, w)
    # 점성 항 (Navier-Stokes 간소화, Laplacian 근사)
    visc_acc = -nu * np.gradient(vel, axis=1, edge_order=2)  # 2차 차분
    # 난류 효과 (간단한 비선형 섭동)
    turb_acc = 0.001 * np.sin(np.sum(pos * pos, axis=0)) * vel
    return grav_acc + visc_acc + turb_acc

def rhs(t, s, G=1.0, masses=(1.0,1.0,1.0), nu=0.01):
    pos, vel = unpack_state(s, N=3)
    acc = accelerations(pos, vel, G, np.asarray(masses, float), nu)
    return pack_state(vel, acc)

def total_energy(s, G=1.0, masses=(1.0,1.0,1.0)):
    m = np.asarray(masses, float)
    pos, vel = unpack_state(s, N=3)
    v2 = np.sum(vel * vel, axis=0)
    K = 0.5 * np.sum(m * v2)
    dr = pos[:, None, :] - pos[:, :, None]
    r = np.sqrt(np.sum(dr*dr, axis=0) + EPS)
    iu = np.triu_indices(m.size, k=1)
    U = -G * np.sum(m[iu[0]] * m[iu[1]] / r[iu])
    return K + U

# ---------------- 초기조건 ----------------
def make_ic(mode="exp1", alpha=1.0):
    if mode == "exp1":
        raw = [0,0,0, 0,0,0, 1,0,0, 0,0.6,0.1, -1,0,0, 0,-0.6,-0.1]
    elif mode == "exp2":
        raw = [0,0,0, 0,0,0, 1,0,0, 0,0.8,0.2, -1,0,0, 0,-0.5,-0.05]
    elif mode == "exp3":
        raw = [0,0,0, 0,0,0, 1,0,0, 0,1.0,0.4, -1,0,0, 0,-0.2,0.0]
    elif mode == "figure8":
        raw = [
            -0.97000436, 0.24308753, 0, 0.466203685, 0.43236573, 0,
             0.97000436, -0.24308753, 0, 0.466203685, 0.43236573, 0,
             0.0, 0.0, 0, -0.93240737, -0.86473146, 0
        ]
    else:
        raise ValueError("mode must be exp1|exp2|exp3|figure8")

    s = np.array(raw, float)
    pos = np.vstack([s[[0,6,12]], s[[1,7,13]], s[[2,8,14]]])
    vel = np.vstack([s[[3,9,15]], s[[4,10,16]], s[[5,11,17]]])
    vel *= alpha
    return pack_state(pos, vel)

# ---------------- 좌표 추출 ----------------
def positions_from_sol(sol, N=3):
    Y = np.asarray(sol.y, float)
    rows = np.arange(N)
    x_rows = 3*rows + 0
    y_rows = 3*rows + 1
    z_rows = 3*rows + 2
    return Y[x_rows], Y[y_rows], Y[z_rows]

# ---------------- 진단 함수 ----------------
def diag_com_momentum(s, masses):
    pos, vel = unpack_state(s, N=3)
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

# ---------------- DTG 업데이트 ----------------
def dtg_update(V_0, alpha, beta, lambda_, b, energy_dev, lyap, theta_t):
    return (1 - lambda_) * theta_t + lambda_ * (b + alpha * energy_dev - beta * lyap)

# ---------------- 실행 ----------------
def run(ic_mode, alpha, t_max, dt, out_root, elev=20, azim=-60, nu=0.01):
    masses = np.array([1.0,1.0,1.0])
    s0 = make_ic(ic_mode, alpha)
    t_eval = np.arange(0.0, t_max + 1e-12, dt)

    sol = solve_ivp(lambda t,s: rhs(t,s,1.0,masses,nu),
                    (0.0, t_max), s0, t_eval=t_eval,
                    method="DOP853", rtol=1e-9, atol=1e-12)

    x, y, z = positions_from_sol(sol, N=3)
    x1,y1,z1 = x[0], y[0], z[0]
    x2,y2,z2 = x[1], y[1], z[1]
    x3,y3,z3 = x[2], y[2], z[2]

    df = pd.DataFrame({"t": sol.t,
                       "x1": x1,"y1": y1,"z1": z1,
                       "x2": x2,"y2": y2,"z2": z2,
                       "x3": x3,"y3": y3,"z3": z3})
    csv_path = os.path.join(out_root, "data", f"threebody3d_{ic_mode}_a{alpha}_ns.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x1,y1,z1,label="Body 1")
    ax.plot(x2,y2,z2,label="Body 2")
    ax.plot(x3,y3,z3,label="Body 3")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"3D Three-Body (ic={ic_mode}, α={alpha}, nu={nu})")
    ax.legend()
    try:
        ax.set_box_aspect((1,1,1))
        ax.set_proj_type('ortho')
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    fig_path = os.path.join(out_root, "figures", f"threebody3d_{ic_mode}_a{alpha}_ns.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    plt.figure(figsize=(6,5))
    for xi, yi, lab in [(x1,y1,"Body 1"), (x2,y2,"Body 2"), (x3,y3,"Body 3")]:
        plt.plot(xi, yi, label=lab)
    plt.xlabel("X"); plt.ylabel("Y"); plt.axis('equal'); plt.legend()
    plt.title(f"XY Projection (ic={ic_mode}, α={alpha}, nu={nu})")
    fig_xy_path = os.path.join(out_root, "figures", f"threebody3d_{ic_mode}_a{alpha}_ns_xy.png")
    plt.savefig(fig_xy_path, dpi=160)
    plt.close()

    E0 = total_energy(sol.y[:,0], 1.0, masses)
    E = np.array([total_energy(sol.y[:,i], 1.0, masses) for i in range(sol.y.shape[1])])
    drift = (E - E0) / (abs(E0) + 1e-15)
    plt.figure(figsize=(7,4))
    plt.plot(sol.t, drift)
    plt.xlabel("Time"); plt.ylabel("Relative Energy Drift")
    plt.title("Total Energy Drift")
    drift_path = os.path.join(out_root, "figures", f"energy_drift_{ic_mode}_a{alpha}_ns.png")
    os.makedirs(os.path.dirname(drift_path), exist_ok=True)
    plt.savefig(drift_path, dpi=160)
    plt.close()

    com0, p0 = diag_com_momentum(sol.y[:,0], masses)
    comT, pT = diag_com_momentum(sol.y[:,-1], masses)
    planar, zmax = diag_planarity(z, tol=1e-3)
    sim = diag_fig8_similarity(x, y)

    print(f"[OK] CSV  : {csv_path}")
    print(f"[OK] FIG  : {fig_path}")
    print(f"[OK] FIGXY: {fig_xy_path}")
    print(f"[OK] DRIFT: {drift_path}")
    print(f"[diag] |COM(t0)|={com0:.3e}, |P(t0)|={p0:.3e} |COM(tT)|={comT:.3e}, |P(tT)|={pT:.3e}")
    print(f"[diag] planar={planar} (max|z|={zmax:.3e}), fig8_similarity={sim:.3e}")
    return sol.t, drift, (planar, zmax, sim)

# ---------------- Lyapunov (Benettin) ----------------
def lyapunov_benettin(s0, rhs, tmax=12.0, dt=0.002, delta0=1e-8,
                      renorm_every=50, G=1.0, masses=(1.0,1.0,1.0), nu=0.01):
    rng = np.random.default_rng(0)
    v = rng.normal(size=s0.size); v /= np.linalg.norm(v)

    t = 0.0
    s = s0.copy()
    s_pert = s + delta0 * v
    d0 = delta0

    lam_sum = 0.0
    n_renorm = 0

    while t < tmax - 1e-12:
        t_next = min(t + renorm_every * dt, tmax)

        sol1 = solve_ivp(lambda tt, ss: rhs(tt, ss, G, masses, nu),
                         (t, t_next), s,
                         method="DOP853", rtol=1e-9, atol=1e-12)
        sol2 = solve_ivp(lambda tt, ss: rhs(tt, ss, G, masses, nu),
                         (t, t_next), s_pert,
                         method="DOP853", rtol=1e-9, atol=1e-12)

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

# ---------------- DTG 실행 ----------------
def run_dtg(t, drift, lyap, V_0=1.0, alpha=0.7, beta=0.5, lambda_=0.1, b=0.0):
    theta_t = V_0
    theta_history = [theta_t]
    for i in range(len(drift)):
        energy_dev = drift[i]
        theta_t = dtg_update(V_0, alpha, beta, lambda_, b, energy_dev, lyap, theta_t)
        theta_history.append(theta_t)
    return theta_history

# ---------------- 메인 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", choices=["exp1","exp2","exp3","figure8"], default="exp2")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tmax", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--out", default=".")
    ap.add_argument("--lyap", action="store_true")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--nu", type=float, default=0.01)  # 점성 계수
    args = ap.parse_args()

    t, drift, diag = run(args.ic, args.alpha, args.tmax, args.dt, args.out,
                         elev=args.elev, azim=args.azim, nu=args.nu)

    if args.lyap:
        masses = np.array([1.0,1.0,1.0])
        s0 = make_ic(args.ic, args.alpha)
        lam = lyapunov_benettin(s0, rhs, tmax=max(6.0, args.tmax),
                                dt=args.dt, G=1.0, masses=masses, nu=args.nu)
        print(f"[Lyapunov_Benettin ≈] {lam:.6f} (주기해면 ~0, 양수면 혼돈 경향)")

    # DTG 실행
    if args.lyap:
        theta_history = run_dtg(t, drift, lam)
        plt.figure(figsize=(7,4))
        plt.plot(t, theta_history)
        plt.xlabel("Time"); plt.ylabel("DTG Theta")
        plt.title(f"DTG Theta (ic={args.ic}, α={args.alpha}, nu={args.nu})")
        dtg_path = os.path.join(args.out, "figures", f"dtg_theta_{args.ic}_a{args.alpha}_ns.png")
        os.makedirs(os.path.dirname(dtg_path), exist_ok=True)
        plt.savefig(dtg_path, dpi=160)
        plt.close()
        print(f"[OK] DTG  : {dtg_path}")

if __name__ == "__main__":
    main()