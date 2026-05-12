#!/usr/bin/env python3
"""Reproduce the QMC Monte Carlo tables and optional figures.

The script uses only the Python standard library for CSV generation. If
matplotlib is installed, it also writes figure files to the output directory.
It is intended as an auditable reference implementation of the equations and
parameter choices reported in the manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from pathlib import Path


BASE_SEED = 20260512
ROUNDS = 60_000
OUT_DIR = Path("repro_output")

BASELINE = {
    "eps_s": 0.008,
    "p_d": 0.004,
    "sigma_t": 0.7,
    "p_cp": 1.5e-3,
    "alpha": 0.005,
    "F_min": 0.820,
    "tau_0": 3.0,
    "sigma_F": 0.018,
    "Delta_max": 4.0,
}

N_VALUES = [4, 7, 10, 13, 16]
NOISE_VALUES = [0.5, 1.0, 1.5, 2.0]


def seed_for(*items: object) -> int:
    material = "|".join(str(item) for item in (BASE_SEED,) + items)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def with_noise(multiplier: float, base: dict[str, float] | None = None) -> dict[str, float]:
    params = dict(BASELINE if base is None else base)
    # Aggregate noise changes preparation, detection, and orchestration loss.
    # Timing-specific stress is modeled separately in the sensitivity sweep.
    for key in ("eps_s", "p_d", "alpha"):
        params[key] *= multiplier
    return params


def visibility(n: int, params: dict[str, float]) -> float:
    return (
        math.exp(-params["alpha"] * (n - 1))
        * (1.0 - params["eps_s"]) ** n
        * (1.0 - 2.0 * params["p_d"]) ** n
        * math.exp(-((params["sigma_t"] / params["tau_0"]) ** 2))
    )


def fidelity(n: int, params: dict[str, float]) -> float:
    return 0.5 * (1.0 + visibility(n, params))


def latency_pbft(n: int) -> float:
    return 10.70 + 1.33 * n


def latency_hotstuff(n: int) -> float:
    return 14.73 + 0.09375 * n


def latency_qmc_pre(n: int) -> float:
    return 9.30 + 0.12 * n


def latency_qmc_demand(n: int) -> float:
    return latency_qmc_pre(n) + 3.50 + 0.24 * n


def qmc_expected_latency(n: int, p_commit: float, qmc_latency: float) -> float:
    t_abort = 0.7
    t_fallback = latency_hotstuff(n)
    return p_commit * qmc_latency + (1.0 - p_commit) * (t_abort + t_fallback)


def tolerated_extra_transcript_overhead(n: int, p_commit: float) -> float:
    """Extra QMC transcript latency tolerated before HotStuff wins."""
    if p_commit <= 0.0:
        return 0.0
    base_expected = qmc_expected_latency(n, p_commit, latency_qmc_pre(n))
    return max(0.0, (latency_hotstuff(n) - base_expected) / p_commit)


def simulate_rounds(n: int, params: dict[str, float], scenario: str, rounds: int = ROUNDS) -> dict[str, float]:
    rng = random.Random(seed_for(n, scenario, tuple(sorted(params.items()))))
    v = visibility(n, params)
    f = 0.5 * (1.0 + v)
    counts = {"commit": 0, "abort": 0, "no_finality": 0}

    for _ in range(rounds):
        offsets = [rng.gauss(0.0, params["sigma_t"]) for _ in range(n)]
        delta = max(offsets) - min(offsets)
        f_hat = min(1.0, max(0.0, rng.gauss(f, params["sigma_F"])))
        control_ok = rng.random() > params["p_cp"]
        parity_plus = rng.random() < 0.5 * (1.0 + v)

        if control_ok and f_hat >= params["F_min"] and delta <= params["Delta_max"] and parity_plus:
            counts["commit"] += 1
        elif (not control_ok) or f_hat < params["F_min"] or delta > params["Delta_max"]:
            counts["abort"] += 1
        else:
            counts["no_finality"] += 1

    return {
        "p_commit": counts["commit"] / rounds,
        "p_abort": counts["abort"] / rounds,
        "p_no_finality": counts["no_finality"] / rounds,
        "visibility": v,
        "fidelity": f,
    }


def emulate_transcript_service(
    arrival_rate: float,
    service_ms: float,
    servers: int,
    duration_s: float = 30.0,
    timeout_ms: float = 12.0,
) -> dict[str, float]:
    """Discrete-event transcript-service emulator.

    It models transcript requests as Poisson arrivals and service time as a
    lightly jittered deterministic threshold-combination plus multicast step.
    The emulator is deliberately narrow: it screens queueing and timeout
    pressure at the transcript layer, not the quantum physical channel.
    """
    rng = random.Random(seed_for("transcript-emulator", arrival_rate, service_ms, servers, duration_s, timeout_ms))
    now = 0.0
    server_free = [0.0 for _ in range(servers)]
    latencies_ms: list[float] = []

    while now < duration_s:
        now += rng.expovariate(arrival_rate)
        if now >= duration_s:
            break
        service_s = max(0.0001, rng.gauss(service_ms / 1000.0, service_ms / 1000.0 * 0.08))
        idx = min(range(servers), key=lambda k: server_free[k])
        start = max(now, server_free[idx])
        finish = start + service_s
        server_free[idx] = finish
        latencies_ms.append((finish - now) * 1000.0)

    if not latencies_ms:
        return {
            "samples": 0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "timeout_fraction": 0.0,
        }

    ordered = sorted(latencies_ms)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    timeouts = sum(1 for value in latencies_ms if value > timeout_ms)
    return {
        "samples": len(latencies_ms),
        "mean_latency_ms": sum(latencies_ms) / len(latencies_ms),
        "p95_latency_ms": p95,
        "timeout_fraction": timeouts / len(latencies_ms),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_outputs() -> dict[str, list[dict[str, object]]]:
    outputs: dict[str, list[dict[str, object]]] = {}

    probability_rows = []
    latency_rows = []
    for n in N_VALUES:
        for noise in NOISE_VALUES:
            params = with_noise(noise)
            result = simulate_rounds(n, params, f"noise={noise}")
            probability_rows.append({"n": n, "noise": noise, **result})
            if noise == 1.0:
                latency_rows.append(
                    {
                        "n": n,
                        "PBFT": latency_pbft(n),
                        "HotStuff": latency_hotstuff(n),
                        "QMC_pre_expected": qmc_expected_latency(n, result["p_commit"], latency_qmc_pre(n)),
                        "QMC_demand_expected": qmc_expected_latency(n, result["p_commit"], latency_qmc_demand(n)),
                    }
                )
    outputs["probability_grid.csv"] = probability_rows
    outputs["latency_curves.csv"] = latency_rows

    break_even_rows = []
    for row in probability_rows:
        if row["noise"] == 1.0:
            n = int(row["n"])
            p_commit = float(row["p_commit"])
            qmc_expected = qmc_expected_latency(n, p_commit, latency_qmc_pre(n))
            break_even_rows.append(
                {
                    "n": n,
                    "p_commit": p_commit,
                    "hotstuff_latency_ms": latency_hotstuff(n),
                    "qmc_pre_expected_latency_ms": qmc_expected,
                    "all_attempt_extra_overhead_margin_ms": latency_hotstuff(n) - qmc_expected,
                    "success_branch_extra_overhead_margin_ms": tolerated_extra_transcript_overhead(n, p_commit),
                }
            )
    outputs["break_even_transcript_overhead.csv"] = break_even_rows

    robustness_rows = []
    cached_results: dict[tuple[int, float], dict[str, float]] = {}
    for n in N_VALUES:
        for noise in [0.75, 1.0, 1.25, 1.5]:
            cached_results[(n, noise)] = simulate_rounds(n, with_noise(noise), f"robustness:noise={noise}")

    for n in N_VALUES:
        for noise in [0.75, 1.0, 1.25, 1.5]:
            p_commit = cached_results[(n, noise)]["p_commit"]
            for hotstuff_factor in [0.70, 0.85, 1.00, 1.15, 1.30]:
                hotstuff_latency = latency_hotstuff(n) * hotstuff_factor
                for qmc_factor in [0.70, 0.85, 1.00, 1.15, 1.30]:
                    for extra_transcript_ms in [0.0, 2.0, 4.0, 6.0, 8.0]:
                        qmc_latency = latency_qmc_pre(n) * qmc_factor
                        expected = (
                            p_commit * qmc_latency
                            + (1.0 - p_commit) * (0.7 + hotstuff_latency)
                            + extra_transcript_ms
                        )
                        robustness_rows.append(
                            {
                                "n": n,
                                "noise": noise,
                                "hotstuff_factor": hotstuff_factor,
                                "qmc_factor": qmc_factor,
                                "extra_transcript_ms": extra_transcript_ms,
                                "p_commit": p_commit,
                                "hotstuff_latency_ms": hotstuff_latency,
                                "qmc_expected_latency_ms": expected,
                                "qmc_faster": expected < hotstuff_latency,
                                "advantage_ms": hotstuff_latency - expected,
                            }
                        )
    outputs["robustness_hotstuff_ranking.csv"] = robustness_rows

    emulator_rows = []
    for service_ms in [2.0, 5.0, 10.0]:
        for arrival_rate in [25, 50, 75, 100, 150, 200]:
            for servers in [1, 2, 4]:
                result = emulate_transcript_service(arrival_rate, service_ms, servers)
                emulator_rows.append(
                    {
                        "arrival_rate_rounds_per_s": arrival_rate,
                        "service_time_ms": service_ms,
                        "servers": servers,
                        "traffic_intensity_per_server": arrival_rate * (service_ms / 1000.0) / servers,
                        **result,
                    }
                )
    outputs["transcript_service_emulator.csv"] = emulator_rows

    sensitivity_rows = []
    for key, values in {
        "F_min": [0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90],
        "sigma_t": [0.2, 0.4, 0.7, 1.0, 1.4, 1.7, 2.0],
        "sigma_F": [0.005, 0.010, 0.018, 0.024, 0.030],
    }.items():
        for value in values:
            params = dict(BASELINE)
            params[key] = value
            result = simulate_rounds(10, params, f"sensitivity:{key}={value}")
            sensitivity_rows.append({"parameter": key, "value": value, **result})
    outputs["sensitivity.csv"] = sensitivity_rows

    stress_rows = []
    stressors = {
        "baseline": dict(BASELINE),
        "local_phase_inversion": dict(BASELINE),
        "selective_fidelity_degradation": {**BASELINE, "alpha": BASELINE["alpha"] / 0.90},
        "coordinated_timing_misalignment": {**BASELINE, "sigma_t": BASELINE["sigma_t"] * 2.0},
    }
    for name, params in stressors.items():
        result = simulate_rounds(10, params, f"stress:{name}")
        if name == "local_phase_inversion":
            shifted_commit = result["p_commit"] * 0.55
            result["p_no_finality"] += result["p_commit"] - shifted_commit
            result["p_commit"] = shifted_commit
        stress_rows.append({"stressor": name, **result})
    outputs["active_stressors.csv"] = stress_rows

    queue_rows = []
    for service_ms in [2.0, 5.0, 10.0]:
        service_s = service_ms / 1000.0
        for arrival_rate in [10, 25, 50, 75, 100, 150]:
            rho = arrival_rate * service_s
            if rho >= 1.0:
                wait_ms = float("inf")
                total_ms = float("inf")
            else:
                # M/D/1 waiting approximation. This is a screening model for
                # transcript-service saturation, not a deployment benchmark.
                wait_s = (arrival_rate * service_s * service_s) / (2.0 * (1.0 - rho))
                wait_ms = wait_s * 1000.0
                total_ms = service_ms + wait_ms
            queue_rows.append(
                {
                    "arrival_rate_rounds_per_s": arrival_rate,
                    "service_time_ms": service_ms,
                    "traffic_intensity": rho,
                    "expected_queue_wait_ms": wait_ms,
                    "expected_transcript_latency_ms": total_ms,
                }
            )
    outputs["transcript_queueing_screen.csv"] = queue_rows

    return outputs


def maybe_plot(outputs: dict[str, list[dict[str, object]]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    latency = outputs["latency_curves.csv"]
    xs = [int(row["n"]) for row in latency]
    plt.figure(figsize=(7, 4))
    for key in ("PBFT", "HotStuff", "QMC_pre_expected", "QMC_demand_expected"):
        plt.plot(xs, [float(row[key]) for row in latency], marker="o", label=key)
    plt.xlabel("Committee size n")
    plt.ylabel("Post-certification finality-interval latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "latency_curves.png", dpi=220)
    plt.close()

    probabilities = outputs["probability_grid.csv"]
    plt.figure(figsize=(7, 4))
    for noise in NOISE_VALUES:
        rows = [row for row in probabilities if float(row["noise"]) == noise]
        plt.plot([int(row["n"]) for row in rows], [float(row["p_commit"]) for row in rows], marker="o", label=f"{noise}x")
    plt.xlabel("Committee size n")
    plt.ylabel("Fast-commit probability")
    plt.legend(title="Noise")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fast_commit_probability.png", dpi=220)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    outputs = build_outputs()
    for filename, rows in outputs.items():
        write_csv(OUT_DIR / filename, rows)
    maybe_plot(outputs)
    print(f"Wrote reproducibility outputs to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
