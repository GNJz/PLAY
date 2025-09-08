#!/usr/bin/env bash
set -e
BEST=(--center_k 0.02 --k_planar 0.12 \
  --dtg 1 --alpha_dtg 1.1 --beta_dtg 0.0 \
  --k_ctrl 0.08 --k_nu 0.08 --nu_max 0.05 \
  --tau_theta 1.1 --theta_init 1.0 \
  --ctrl_signed --gate_floor 0.30 \
  --r_collide 0.05 --recenter_mode momentum --recenter_period 1.0)
ADAPT=(--adaptive_step --adapt_r_trigger 0.03 --adapt_factor 0.5 --min_max_step 1e-4)

for turb in 0 5e-4 1e-3 1.5e-3 2e-3 3e-3 4e-3 5e-3; do
  out="runs/turb_scan_t${turb}"
  python3 three_body_3d_ns_dtg_v3_5_full_FIXED.py \
    --compare --nu_scan 0.003 0.0035 \
    --tmax 300 --dt 0.0005 --out "$out" --no_plots --turb $turb \
    "${ADAPT[@]}" --seed 42 "${BEST[@]}"
done
python3 validate_summary.py
