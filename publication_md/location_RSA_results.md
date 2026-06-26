# Results — Location RSA companion analysis

This note summarises the single-model `location` RSA from the DSR comparison run `DSR_RSA_simple_ROI/2026-06-22_16-17-15-DSR`, with emphasis on the hippocampal and parahippocampal regions. I focus on the base-model summaries because they isolate the location regressor without additional competition from the multi-regressor control stacks.

The clearest location effects were found in the medial temporal lobe. In the primary `split_halves_z` analysis, Parahippocampal cortex showed the largest effect (beta = 0.0643, t = 4.35, p_perm = 0.000999, q_FDR = 0.00699), followed by HC_anterior (beta = 0.0434, t = 2.93, p_perm = 0.001998, q_FDR = 0.00699) and HC_mid (beta = 0.0399, t = 2.69, p_perm = 0.007992, q_FDR = 0.01865). PCC, medialOFC, EC, and ACC were not significant after correction, and ACC in fact showed a small negative beta (beta = -0.0162, p_perm = 0.841). Thus, the location code in this RSA is concentrated in hippocampal and parahippocampal populations rather than frontal cortex.

| ROI | n neurons | beta | t | p_perm | q_FDR |
|---|---:|---:|---:|---:|---:|
| Parahippocampal | 46 | 0.064330 | 4.352 | 0.000999 | 0.006993 |
| HC_anterior | 179 | 0.043414 | 2.934 | 0.001998 | 0.006993 |
| HC_mid | 88 | 0.039875 | 2.694 | 0.007992 | 0.018648 |
| PCC | 54 | 0.021510 | 1.453 | 0.103896 | 0.181818 |
| medialOFC | 90 | 0.015293 | 1.033 | 0.148851 | 0.208392 |
| EC | 11 | 0.007312 | 0.494 | 0.300699 | 0.350816 |
| ACC | 68 | -0.016201 | -1.094 | 0.841159 | 0.841159 |

The same regional ordering was preserved in the secondary `between_tasks_z` analysis, which keeps only between-configuration cells. Parahippocampal cortex again carried the strongest location effect (beta = 0.0530, p_perm = 0.000999, q = 0.00699), with HC_anterior (beta = 0.0461, p_perm = 0.002997, q = 0.01049) and HC_mid (beta = 0.0441, p_perm = 0.008991, q = 0.02098) also surviving FDR correction. This convergence across `split_halves_z` and `between_tasks_z` argues that the location regressor is not being driven only by within-configuration structure or run-specific idiosyncrasies.

For the manuscript, the cleanest claim is therefore that single-unit population geometry carries a robust location signal in parahippocampal cortex and along the hippocampal long axis, with the strongest and most reproducible effects in Parahippocampal cortex, HC_anterior, and HC_mid. This complements the DSR results by showing that medial temporal populations encode both where the subject is in the environment (`location`) and, in more model-dependent fashion, aspects of future action structure (`dsr_fmri`, `dsr_fmri_informed`).
