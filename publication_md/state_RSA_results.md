# Results — State RSA companion analysis

This analysis corresponds to the state-RSA panel used as the fourth column of Fig. 11. I report the run saved in `DSR_RSA_simple_ROI/2026-06-26_11-30-30-final-State`, focusing on the variant you specified most recently: `test = split_halves_z`, `sub_model = state`, `combo = ctrl_dsrFULL`. In the script, this is the z-scored cross-run RSA: population vectors from run 1 and run 2 are correlated across halves, so the RDM includes both within-configuration-across-run cells and between-configuration cells. The sample comprised 536 neurons across seven ROIs.

State structure was again positive in every analysed ROI and survived BH-FDR correction across the seven ROIs when permutation p values were corrected within the requested `ctrl_dsrFULL` model. The strongest effects were observed in HC_mid (beta = 0.0982, t = 6.62, p_perm = 0.001996, q = 0.00349), medialOFC (beta = 0.0828, t = 5.53, p_perm = 0.001996, q = 0.00349), ACC (beta = 0.0732, t = 4.94, p_perm = 0.001996, q = 0.00349), and HC_anterior (beta = 0.0707, t = 4.73, p_perm = 0.001996, q = 0.00349). PCC remained reliable (beta = 0.0465, p_perm = 0.00599, q = 0.00838), and smaller but still significant effects were present in Parahippocampal cortex (beta = 0.0372, p_perm = 0.01996, q = 0.02329) and EC (beta = 0.0310, p_perm = 0.04192, q = 0.04192).

| ROI | n neurons | beta | t | p_perm | q_FDR |
|---|---:|---:|---:|---:|---:|
| ACC | 68 | 0.073156 | 4.939 | 0.001996 | 0.003493 |
| EC | 11 | 0.030951 | 2.061 | 0.041916 | 0.041916 |
| Parahippocampal | 46 | 0.037231 | 2.486 | 0.019960 | 0.023287 |
| HC_anterior | 179 | 0.070654 | 4.733 | 0.001996 | 0.003493 |
| HC_mid | 88 | 0.098188 | 6.619 | 0.001996 | 0.003493 |
| medialOFC | 90 | 0.082753 | 5.533 | 0.001996 | 0.003493 |
| PCC | 54 | 0.046464 | 3.105 | 0.005988 | 0.008383 |

The same qualitative ranking held when the control stack was expanded to include `phase` (`ctrl_dsrFULL_phase`): all seven ROIs again survived BH-FDR, and the beta estimates changed only minimally. Adding the stricter `state_phase` regressor (`ctrl_dsrFULL_state-phase`) attenuated the state term more noticeably, but state remained significant in medialOFC, HC_anterior, HC_mid, ACC, and PCC. This pattern points in the same direction as the existing sustained-state summary in `results_ephys_sustained_state.md`, which argues for widespread state coding across PFC and MTL with particularly strong effects in medialOFC and ACC, and it is also compatible with the fMRI state write-up in `results_fMRI_state.md`, where the state effect is detectable in vmPFC and left EC. The cross-method overlap is therefore not that every region peaks in the same place, but that state information is distributed across frontal and medial temporal structures in all three analyses.
