#!/usr/bin/env python3
"""Reservoir Sampling Uniformity Experiment.

Tests correct reservoir sampler (Algorithm R) and two buggy variants on synthetic streams,
measuring position-wise inclusion rates and deviation from expected k/N probability.
"""

from loguru import logger
from pathlib import Path
import json
import random
import sys
from typing import List, Dict, Any
from collections import defaultdict
import math


logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")


# =============================================================================
# Reservoir Sampler Implementations
# =============================================================================

def reservoir_sampling_correct(stream: List[int], k: int, seed: int) -> List[int]:
    """Algorithm R: Correct reservoir sampling.
    
    Fill first k items, then for each item i (1-indexed after reservoir full),
    draw j ~ Uniform{1..i}; replace if j <= k.
    """
    rng = random.Random(seed)
    reservoir = []
    
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # i is 0-indexed, so we're at position i+1 (1-indexed)
            j = rng.randint(1, i + 1)
            if j <= k:
                reservoir[j - 1] = item
    
    return reservoir


def reservoir_sampling_replacement_bug(stream: List[int], k: int, seed: int) -> List[int]:
    """Buggy variant: Uses wrong replacement probability k/(i+1) instead of k/i.
    
    The correct probability at step i (0-indexed, after reservoir is full) is k/(i+1)
    since i+1 is the 1-indexed position. But this variant uses an off-by-one
    that changes the replacement rate.
    """
    rng = random.Random(seed)
    reservoir = []
    
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # BUG: Uses k/(i+1) but should be k/(i+1) - wait, that's correct
            # The bug is using k/(i+2) or similar off-by-one
            # Let's use k/(i+2) which is wrong
            prob = k / (i + 2)
            if rng.random() < prob:
                # Randomly choose which slot to replace
                idx = rng.randint(0, k - 1)
                reservoir[idx] = item
    
    return reservoir


def reservoir_sampling_index_bug(stream: List[int], k: int, seed: int) -> List[int]:
    """Buggy variant: Draws replacement indices from wrong range.
    
    Uses Uniform{0..k} or Uniform{1..k+1} instead of Uniform{1..i+1} for j,
    so some reservoir slots are favored or never chosen.
    """
    rng = random.Random(seed)
    reservoir = []
    
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # BUG: Draws j from 1..k+1 instead of 1..i+1
            j = rng.randint(1, k + 1)
            if j <= k:
                reservoir[j - 1] = item
    
    return reservoir


# =============================================================================
# Experiment Functions
# =============================================================================

def run_trials(sampler_func, stream: List[int], k: int, num_trials: int, seed_offset: int = 0) -> List[List[int]]:
    """Run multiple trials of a sampler, returning list of reservoirs."""
    reservoirs = []
    for trial in range(num_trials):
        seed = seed_offset + trial * 1000003  # Large prime for good separation
        reservoir = sampler_func(stream, k, seed)
        reservoirs.append(reservoir)
    return reservoirs


def compute_inclusion_rates(reservoirs: List[List[int]], N: int, k: int) -> List[float]:
    """Compute empirical inclusion frequency for each stream position."""
    counts = [0] * N
    for reservoir in reservoirs:
        for item in reservoir:
            if 0 <= item < N:
                counts[item] += 1
    return [c / len(reservoirs) for c in counts]


def compute_metrics(inclusion_rates: List[float], expected_rate: float) -> Dict[str, float]:
    """Compute deviation metrics from expected uniform rate."""
    deviations = [abs(rate - expected_rate) for rate in inclusion_rates]
    max_dev = max(deviations)
    mean_dev = sum(deviations) / len(deviations)
    
    # Correlation of inclusion rate with stream index (positional trend)
    n = len(inclusion_rates)
    mean_idx = (n - 1) / 2
    mean_rate = sum(inclusion_rates) / n
    cov = sum((i - mean_idx) * (r - mean_rate) for i, r in enumerate(inclusion_rates))
    var_idx = sum((i - mean_idx) ** 2 for i in range(n))
    var_rate = sum((r - mean_rate) ** 2 for r in inclusion_rates)
    
    if var_idx > 0 and var_rate > 0:
        correlation = cov / math.sqrt(var_idx * var_rate)
    else:
        correlation = 0.0
    
    # Chi-square statistic
    chi2 = sum((rate - expected_rate) ** 2 / expected_rate for rate in inclusion_rates) if expected_rate > 0 else 0.0
    
    return {
        "max_abs_deviation": max_dev,
        "mean_abs_deviation": mean_dev,
        "positional_correlation": correlation,
        "chi_square": chi2,
    }


def run_experiment(N: int, k: int, num_trials: int, seed_base: int = 42) -> Dict[str, Any]:
    """Run full experiment for given N, k."""
    stream = list(range(N))
    expected_rate = k / N
    
    # Run all three samplers
    samplers = {
        "correct": reservoir_sampling_correct,
        "replacement_bug": reservoir_sampling_replacement_bug,
        "index_bug": reservoir_sampling_index_bug,
    }
    
    results = {}
    for name, sampler in samplers.items():
        logger.info(f"Running {name} sampler: N={N}, k={k}, trials={num_trials}")
        reservoirs = run_trials(sampler, stream, k, num_trials, seed_base)
        
        # Verify reservoir size
        for i, res in enumerate(reservoirs):
            if len(res) != k:
                logger.warning(f"Trial {i}: reservoir size {len(res)} != k={k}")
        
        inclusion_rates = compute_inclusion_rates(reservoirs, N, k)
        metrics = compute_metrics(inclusion_rates, expected_rate)
        
        results[name] = {
            "inclusion_rates": inclusion_rates,
            "metrics": metrics,
            "expected_rate": expected_rate,
        }
    
    return results


def run_extended_experiments() -> Dict[str, Any]:
    """Run extended experiments with more conditions to get 50+ examples."""
    # More granular N and k values for more examples
    N_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    k_values = [1, 2, 3, 5, 7, 10]
    num_trials = 10_000
    seed_base = 42
    
    all_results = {}
    
    for N in N_values:
        for k in k_values:
            if k > N:
                continue
            
            key = f"N_{N}_k_{k}"
            logger.info(f"\n=== Running {key} ===")
            
            results = run_experiment(N, k, num_trials, seed_base)
            all_results[key] = {
                "N": N,
                "k": k,
                "num_trials": num_trials,
                "samplers": results,
            }
            
            # Log summary
            for sampler_name, sampler_results in results.items():
                m = sampler_results["metrics"]
                logger.info(f"  {sampler_name}: max_dev={m['max_abs_deviation']:.6f}, "
                          f"mean_dev={m['mean_abs_deviation']:.6f}, "
                          f"corr={m['positional_correlation']:.6f}, "
                          f"chi2={m['chi_square']:.2f}")
    
    return all_results


# =============================================================================
# Main
# =============================================================================

@logger.catch(reraise=True)
def main():
    # Run extended experiments for more examples
    logger.info(f"Starting reservoir sampling experiment (extended)")
    
    all_results = run_extended_experiments()
    
    # Create summary table
    logger.info("\n=== SUMMARY TABLE ===")
    logger.info(f"{'Condition':<15} {'Sampler':<20} {'Max Dev':<12} {'Mean Dev':<12} {'Corr':<10} {'Chi2':<10}")
    logger.info("-" * 80)
    
    for key, data in all_results.items():
        for sampler_name, sampler_results in data["samplers"].items():
            m = sampler_results["metrics"]
            logger.info(f"{key:<15} {sampler_name:<20} {m['max_abs_deviation']:<12.6f} "
                      f"{m['mean_abs_deviation']:<12.6f} {m['positional_correlation']:<10.6f} "
                      f"{m['chi_square']:<10.2f}")
    
    # Prepare output in exp_gen_sol_out.json schema format
    # We'll use "input" as the experiment condition, "output" as metrics
    # Add predict_* fields for each method/sampler combination
    examples = []
    
    for key, data in all_results.items():
        for sampler_name, sampler_results in data["samplers"].items():
            # Create input description
            input_desc = f"N={data['N']}, k={data['k']}, trials={data['num_trials']}, sampler={sampler_name}"
            
            # Create output with metrics
            m = sampler_results["metrics"]
            output_data = {
                "max_abs_deviation": m["max_abs_deviation"],
                "mean_abs_deviation": m["mean_abs_deviation"],
                "positional_correlation": m["positional_correlation"],
                "chi_square": m["chi_square"],
                "expected_rate": sampler_results["expected_rate"],
                "N": data["N"],
                "k": data["k"],
                "sampler": sampler_name,
                "num_trials": data["num_trials"],
            }
            
            examples.append({
                "input": input_desc,
                "output": json.dumps(output_data),
                "metadata_condition": key,
                "metadata_sampler": sampler_name,
                "metadata_N": data["N"],
                "metadata_k": data["k"],
                "predict_correct_sampler": json.dumps(all_results[key]["samplers"]["correct"]["metrics"]) if "correct" in all_results[key]["samplers"] else "",
                "predict_replacement_bug_sampler": json.dumps(all_results[key]["samplers"]["replacement_bug"]["metrics"]) if "replacement_bug" in all_results[key]["samplers"] else "",
                "predict_index_bug_sampler": json.dumps(all_results[key]["samplers"]["index_bug"]["metrics"]) if "index_bug" in all_results[key]["samplers"] else "",
            })
    
    # Also save detailed per-position rates for the largest case
    largest_key = max(all_results.keys(), key=lambda k: all_results[k]["N"] * all_results[k]["k"])
    if largest_key in all_results:
        largest_data = all_results[largest_key]
        detail_output = {
            "condition": largest_key,
            "N": largest_data["N"],
            "k": largest_data["k"],
            "samplers": {}
        }
        for sampler_name, sampler_results in largest_data["samplers"].items():
            detail_output["samplers"][sampler_name] = {
                "inclusion_rates": sampler_results["inclusion_rates"],
                "expected_rate": sampler_results["expected_rate"],
            }
        
        detail_path = Path("detailed_rates.json")
        detail_path.write_text(json.dumps(detail_output, indent=2))
        logger.info(f"Saved detailed per-position rates to {detail_path}")
    
    # Output in required schema format
    # Use parameters from the first result or defaults
    first_key = next(iter(all_results))
    first_data = all_results[first_key]
    
    output = {
        "metadata": {
            "method_name": "reservoir_sampling_uniformity_test",
            "description": "Empirical uniformity test of reservoir sampling algorithms",
            "parameters": {
                "N_values": [50, 100, 200, 500, 1000, 2000, 5000, 10000],
                "k_values": [1, 2, 3, 5, 7, 10],
                "num_trials": 10000,
                "seed_base": 42,
            },
        },
        "datasets": [
            {
                "dataset": "synthetic_stream_positions",
                "examples": examples,
            }
        ],
    }
    
    output_path = Path("method_out.json")
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved results to {output_path}")
    logger.info(f"Total examples generated: {len(examples)}")
    
    # Validate against schema
    import subprocess
    result = subprocess.run([
        sys.executable,
        "/ai-inventor/.claude/skills/aii-json/scripts/aii_json_validate_schema.py",
        "--format", "exp_gen_sol_out",
        "--file", str(output_path.absolute()),
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Schema validation PASSED")
    else:
        logger.error(f"Schema validation FAILED: {result.stderr}")
        raise RuntimeError(f"Schema validation failed: {result.stderr}")


if __name__ == "__main__":
    main()