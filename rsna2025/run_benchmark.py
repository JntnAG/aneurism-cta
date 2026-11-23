#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark and Inference Runner for RSNA 2025 Brain Aneurysm Detection
1st Place Winning Solution (tomoon33 / uchiyama33)

Usage:
    # Run benchmark on all series in data/series
    python run_benchmark.py

    # Run benchmark on a specific DICOM series folder
    python run_benchmark.py --series-path "data/series/1.2.826.0.1.3680043.8.498.XXXXXX"

    # Limit benchmark to first N series
    python run_benchmark.py --max-series 10 --output benchmark_results.csv
"""

import os
import sys
import time
import argparse
from pathlib import Path
import polars as pl
from dotenv import load_dotenv

# Ensure PROJECT_ROOT and rootutils setup
import rootutils
rootutils.setup_root(Path(__file__).resolve().parent, indicator=".project-root", pythonpath=True)

# Load environment variables from .env
env_file = Path(__file__).resolve().parent / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file, override=False)

# Ensure bin directory (with dcm2niix.exe) is in PATH
bin_dir = str(Path(__file__).resolve().parent / "bin")
if bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

# Import the official submission predict function
from scripts.rsna_submission_roi import predict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RSNA 2025 Intracranial Aneurysm Detection Benchmark (tomoon33 1st place)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/series",
        help="Path to folder containing DICOM series directories (default: data/series)",
    )
    parser.add_argument(
        "--series-path",
        type=str,
        default=None,
        help="Path to a single DICOM series directory",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Maximum number of series to evaluate during benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file path for predictions (default: benchmark_results.csv)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., cuda:0 or cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device:
        os.environ["VESSEL_DEVICE"] = args.device

    print("=" * 80)
    print("RSNA 2025 Brain Aneurysm Detection - 1st Place Solution Benchmark")
    print("=" * 80)
    print(f"Device: {os.environ.get('VESSEL_DEVICE', 'cuda:0')}")
    print(f"ROI Experiment: {os.environ.get('ROI_EXPERIMENTS', 'default')}")
    print(f"Sparse Model: {os.environ.get('VESSEL_NNUNET_SPARSE_MODEL_DIR', 'default')}")
    print(f"Dense Model: {os.environ.get('VESSEL_NNUNET_MODEL_DIR', 'default')}")

    # Collect series directories
    series_list = []
    if args.series_path:
        p = Path(args.series_path)
        if not p.is_dir():
            print(f"Error: Specified series path '{args.series_path}' is not a directory.")
            sys.exit(1)
        series_list.append(p)
    else:
        input_root = Path(args.input_dir)
        if not input_root.exists():
            print(f"Error: Input directory '{args.input_dir}' does not exist.")
            print("Please place your DICOM series under 'data/series/<SeriesInstanceUID>/'")
            sys.exit(1)

        for item in sorted(input_root.iterdir()):
            if item.is_dir():
                series_list.append(item)

    if not series_list:
        print(f"\n[!] No DICOM series directories found in '{args.input_dir}'.")
        print("\nExpected directory structure:")
        print("  data/")
        print("    series/")
        print("      <SeriesInstanceUID_1>/")
        print("        *.dcm")
        print("      <SeriesInstanceUID_2>/")
        print("        *.dcm")
        print("\nTo test, place your CTA/DICOM series inside 'data/series/' and run:")
        print("  python run_benchmark.py")
        sys.exit(0)

    if args.max_series:
        series_list = series_list[:args.max_series]

    total_count = len(series_list)
    print(f"\nFound {total_count} series to benchmark.")
    print("-" * 80)

    results = []
    latencies = []

    for idx, s_path in enumerate(series_list, 1):
        s_name = s_path.name
        print(f"[{idx}/{total_count}] Processing series: {s_name} ...", end="", flush=True)
        t0 = time.perf_counter()

        try:
            df_pred = predict(str(s_path))
            t_elapsed = time.perf_counter() - t0
            latencies.append(t_elapsed)

            # Add metadata columns if not present
            if "SeriesInstanceUID" not in df_pred.columns:
                df_pred = df_pred.with_columns(pl.lit(s_name).alias("SeriesInstanceUID"))
            df_pred = df_pred.with_columns(pl.lit(t_elapsed).alias("inference_time_sec"))

            results.append(df_pred)
            ap_score = df_pred["Aneurysm Present"][0] if "Aneurysm Present" in df_pred.columns else "N/A"
            print(f" Done ({t_elapsed:.2f}s) | P(Aneurysm Present) = {ap_score:.4f}" if isinstance(ap_score, float) else f" Done ({t_elapsed:.2f}s)")

        except Exception as e:
            t_elapsed = time.perf_counter() - t0
            print(f" FAILED ({t_elapsed:.2f}s): {e}")

    if results:
        all_preds = pl.concat(results)

        # Save results
        out_csv = Path(args.output)
        out_parquet = out_csv.with_suffix(".parquet")
        all_preds.write_csv(out_csv)
        all_preds.write_parquet(out_parquet)

        avg_time = sum(latencies) / len(latencies) if latencies else 0
        print("=" * 80)
        print("BENCHMARK COMPLETED")
        print("=" * 80)
        print(f"Evaluated Series : {len(results)} / {total_count}")
        print(f"Total Time       : {sum(latencies):.2f} s")
        print(f"Average Latency  : {avg_time:.2f} s / series")
        print(f"Throughput       : {1.0 / avg_time:.2f} series / s" if avg_time > 0 else "")
        print(f"Saved Predictions: {out_csv.resolve()}")
        print(f"Parquet Output   : {out_parquet.resolve()}")
        print("=" * 80)
        print("\nPredictions preview:")
        print(all_preds.head(10))
    else:
        print("\nNo series were successfully processed.")


if __name__ == "__main__":
    main()
