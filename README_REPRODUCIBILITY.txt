Reproducibility notes for the QMC evaluation
============================================

The script reproduce_qmc_results.py records the parameter table, latency
surrogates, Monte Carlo generator, active-stressor settings, transcript-service
queueing model, event-driven transcript-service emulator, expanded robustness
sweep, and deterministic seeds used for the evaluation.

Run:

    python reproduce_qmc_results.py

Outputs are written to the repro_output directory as CSV files, including:

    probability_grid.csv
    latency_curves.csv
    sensitivity.csv
    active_stressors.csv
    transcript_queueing_screen.csv
    transcript_service_emulator.csv
    break_even_transcript_overhead.csv
    robustness_hotstuff_ranking.csv

If matplotlib is installed in the Python environment, the script also writes
two diagnostic PNG figures. The reference seed is 20260512, with independent
sub-seeds derived from each configuration tuple.

The script is a reproducibility artifact for the system-level model in the
manuscript. It is not a claim of calibration to a specific quantum hardware
platform.
