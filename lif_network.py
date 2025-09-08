# lif_network.py — 5뉴런 SNN, 래스터 플롯 / 파일 저장
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("figs", exist_ok=True)
plt.savefig(os.path.join("figs", "raster_baseline.png"), bbox_inches="tight")

# ===== LIF 파라미터 =====
dt = 1e-3
T = 1.0
steps = int(T/dt)

tau = 0.02
v_rest = 0.0
R = 40.0
theta_base = 1.0
v_reset = 0.2

N = 5                    # 뉴런 수
rng = np.random.default_rng(0)

# 외부 입력: 0.1~0.7초 동안 펄스 (뉴런마다 조금씩 노이즈)
I_ext = np.zeros((N, steps), dtype=np.float32)
for i in range(N):
    I_ext[i, int(0.10/dt):int(0.70/dt)] = 1.0 + 0.05*rng.standard_normal()

# 연결 가중치(희소, 흥분성 위주, 소량 억제)
W = rng.uniform(0.0, 0.25, size=(N, N)).astype(np.float32)
np.fill_diagonal(W, 0.0)
# 소량 억제 연결 추가
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
        # 이전 시점 스파이크가 다음 시점 전류에 미치는 영향 (한 스텝 지연)
        rec_input = (W @ spikes[:, t-1]) if t > 0 else 0.0
        I_t = I_ext[:, t] + rec_input

        # LIF 적분
        dV = dt * (-(V - v_rest)/tau + R*I_t)
        V += dV

        # 스파이크 & 리셋
        fired = V >= theta
        spikes[fired, t] = 1
        V[fired] = v_reset

    return spikes

def plot_raster(spikes, title, outpath):
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
    # 필요하면 창으로도 보기
    # plt.show()
    plt.close(fig)

# ===== 실행: Baseline vs Gate(IG) =====
spk_base = simulate(alpha=1.0)
spk_gate = simulate(alpha=0.7)  # IG on: 임계값 낮춤 → 민감도↑

plot_raster(spk_base, "Raster — Baseline (alpha=1.0)", "figs/raster_baseline.png")
plot_raster(spk_gate, "Raster — IG on (alpha=0.7)", "figs/raster_ig.png")

print("Saved figures:")
print(" - figs/raster_baseline.png")
print(" - figs/raster_ig.png")

# 간단한 요약(뉴런별 총 스파이크 수)
print("Total spikes (baseline):", spk_base.sum())
print("Total spikes (IG on)  :", spk_gate.sum())
