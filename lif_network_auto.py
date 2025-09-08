# lif_network_auto.py — 5뉴런 SNN, 자동 저장 / summary.json 생성
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ===== 실험 저장 경로 설정 =====
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join("results", timestamp)
os.makedirs(save_dir, exist_ok=True)

# ===== LIF 파라미터 =====
dt = 1e-3
T = 1.0
steps = int(T/dt)

tau = 0.02
v_rest = 0.0
R = 40.0
theta_base = 1.0
v_reset = 0.2

N = 5
rng = np.random.default_rng(0)

# 외부 입력: 0.1~0.7초 동안 펄스 (뉴런마다 노이즈 추가)
I_ext = np.zeros((N, steps), dtype=np.float32)
for i in range(N):
    I_ext[i, int(0.10/dt):int(0.70/dt)] = 1.0 + 0.05 * rng.standard_normal()

# 연결 가중치(희소, 흥분성 위주, 소량 억제)
W = rng.uniform(0.0, 0.25, size=(N, N)).astype(np.float32)
np.fill_diagonal(W, 0.0)
for _ in range(3):
    i, j = rng.integers(0, N, size=2)
    if i != j:
        W[i, j] = -0.15

def simulate(alpha=1.0):
    """alpha * theta 로 임계값 조절 (alpha<1 => gate ON)"""
    theta = alpha * theta_base
    V = np.full(N, v_rest, dtype=np.float32)
    spikes = np.zeros((N, steps), dtype=np.int8)

    for t in range(steps):
        rec_input = (W @ spikes[:, t-1]) if t > 0 else 0.0
        I_t = I_ext[:, t] + rec_input
        dV = dt * (-(V - v_rest)/tau + R*I_t)
        V += dV
        fired = V >= theta
        spikes[fired, t] = 1
        V[fired] = v_reset
    return spikes

def plot_raster(spikes, title, outpath):
    """래스터 플롯 생성 및 저장"""
    t_ms = np.arange(spikes.shape[1]) * dt * 1000.0
    fig, ax = plt.subplots(figsize=(9, 4))
    rows, cols = np.where(spikes == 1)
    ax.scatter(t_ms[cols], rows, s=8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron id")
    ax.set_title(title)
    ax.set_ylim(-0.5, N-0.5)
    ax.set_yticks(range(N))
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

# ===== 실행: Baseline vs IG Gate =====
spk_base = simulate(alpha=1.0)
spk_gate = simulate(alpha=0.7)

# ===== 결과 저장 =====
plot_raster(spk_base, "Raster — Baseline (alpha=1.0)",
            os.path.join(save_dir, "raster_baseline.png"))
plot_raster(spk_gate, "Raster — IG on (alpha=0.7)",
            os.path.join(save_dir, "raster_ig.png"))

# ===== summary.json 자동 저장 =====
summary = {
    "timestamp": timestamp,
    "total_spikes_baseline": int(spk_base.sum()),
    "total_spikes_ig_on": int(spk_gate.sum()),
    "alpha_baseline": 1.0,
    "alpha_ig_on": 0.7,
    "duration_sec": T
}
with open(os.path.join(save_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

# ===== 콘솔 출력 =====
print("=== 실험 완료 ===")
print(f"결과 폴더: {save_dir}")
print(f"Baseline spikes : {spk_base.sum()}")
print(f"IG on spikes   : {spk_gate.sum()}")
print(f"Summary saved  : {os.path.join(save_dir, 'summary.json')}")
