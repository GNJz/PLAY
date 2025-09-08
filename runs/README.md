# DTG Sweep Summary
- Relaxed best: runs/report_relaxed/best_case (theta_active_thresh=0.004)
- Strict best:  runs/report_strict/best_case  (theta_active_thresh=0.01)
Reproduce strict best:
python three_body_3d_ns_dtg_v3_5_full.py --ic exp2 --tmax 1 --dt 0.0015 --use_symplectic --dtg 1 --k_planar 0.6 --alpha_dtg 2.1 --tau_theta 1.4 --k_ctrl 0.12 --ctrl_signed --dtheta_max 0.12 --out runs/plots_sweep4_strict/ad2.10_tt1.40_kc0.12
