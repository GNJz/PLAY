# PLAY/DHT_hair.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# DTG-based hair loss progression model
def dht_progression(t, theta_t, dht_level, genetic_risk, stress, nutrition, treatment):
    alpha, beta, lambda_, V_0 = 0.7, 0.5, 0.1, theta_t[0]
    treatment_effect = {'none': 0, 'minoxidil': 0.3, 'finasteride': 0.4, 'stemcell': 0.5}[treatment]
    genetic_factor = {'high': 1.5, 'medium': 1.0, 'low': 0.5}[genetic_risk]
    E_t = dht_level * genetic_factor * (stress / 10)  # Progression rate
    I_t = (nutrition / 10) + treatment_effect  # Inhibition rate
    dtheta_dt = (1 - lambda_) * theta_t + lambda_ * (V_0 + alpha * E_t - beta * I_t)
    return dtheta_dt

# Simulate hair loss over time
def simulate_hair_loss(dht_level, genetic_risk, stress, nutrition, treatment, years, initial_stage=1.0):
    t_span = (0, years)
    t_eval = np.arange(0, years + 0.01, 0.01)
    sol = solve_ivp(
        dht_progression, t_span, [initial_stage],
        args=(dht_level, genetic_risk, stress, nutrition, treatment),
        t_eval=t_eval, method='RK45'
    )
    stages = np.clip(sol.y[0], 1, 7)
    return t_eval, stages

# Visualize progression
def plot_hair_loss(t, stages, treatment, filename='PLAY/hair_loss_progression.png'):
    plt.figure(figsize=(8, 5))
    plt.plot(t, stages, label=f'Hair Loss Stage ({treatment})', color='#3b82f6')
    plt.xlabel('Years')
    plt.ylabel('Hamilton-Norwood Stage')
    plt.title('Hair Loss Progression Simulation')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Export results for React
def export_results(t, stages, filename='PLAY/hair_loss_data.json'):
    data = {'time': t.tolist(), 'stages': stages.tolist()}
    pd.DataFrame(data).to_json(filename, orient='records')

# Example usage
if __name__ == "__main__":
    dht_level = 0.8  # ng/mL
    genetic_risk = 'high'
    stress = 7
    nutrition = 5
    treatment = 'minoxidil'
    years = 5

    t, stages = simulate_hair_loss(dht_level, genetic_risk, stress, nutrition, treatment, years)
    final_stage = int(round(stages[-1]))
    print(f"Final Hair Loss Stage after {years} years: {final_stage}")
    plot_hair_loss(t, stages, treatment)
    export_results(t, stages)