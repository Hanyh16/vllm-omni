#!/usr/bin/env python3
"""
Analyze GPU SM utilization from nvidia-smi dmon log.

Usage:
    python3 analyze_gpu_sm.py <log_file> [--gpus 0,1] [--output report.txt]

Examples:
    python3 analyze_gpu_sm.py gpu_sm_log.csv
    python3 analyze_gpu_sm.py gpu_sm_log.csv --gpus 2,3
    python3 analyze_gpu_sm.py gpu_sm_log.csv --gpus 0,1 --output report.txt
"""

import argparse
import sys
from collections import defaultdict


def parse_dmon_log(filepath, gpu_ids):
    """Parse nvidia-smi dmon log and extract SM utilization per GPU."""
    gpu_sm = defaultdict(list)

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                gpu_id = int(parts[0])
                sm = int(parts[1])
            except (ValueError, IndexError):
                continue
            if gpu_id in gpu_ids:
                gpu_sm[gpu_id].append(sm)

    return gpu_sm


def analyze(gpu_sm, gpu_ids):
    """Analyze SM utilization and return report lines."""
    lines = []

    ar_gpu = gpu_ids[0]
    dit_gpu = gpu_ids[1] if len(gpu_ids) > 1 else None

    # ----- Per-GPU stats -----
    for gid in gpu_ids:
        data = gpu_sm.get(gid, [])
        if not data:
            lines.append(f"  [WARNING] No data for GPU {gid}")
            continue
        active = [v for v in data if v > 0]
        role = "AR/LLM" if gid == ar_gpu else "DiT/Diffusion"
        lines.append(f"--- GPU {gid} (Stage: {role}) ---")
        lines.append(f"  Samples:               {len(data)}")
        lines.append(f"  Overall avg SM:        {sum(data)/len(data):.1f}%")
        lines.append(f"  Active samples:        {len(active)}/{len(data)}"
                      f" ({100*len(active)/len(data):.1f}%)")
        if active:
            lines.append(f"  Active avg SM:         {sum(active)/len(active):.1f}%")
            lines.append(f"  Active min/max SM:     {min(active)}% / {max(active)}%")
        lines.append("")

    # ----- Pipeline phase analysis (requires 2 GPUs) -----
    if dit_gpu is not None:
        ar_data = gpu_sm.get(ar_gpu, [])
        dit_data = gpu_sm.get(dit_gpu, [])
        n = min(len(ar_data), len(dit_data))
        if n == 0:
            lines.append("  [WARNING] Not enough paired samples for pipeline analysis")
            return lines

        both_active = gpu0_only = gpu1_only = both_idle = 0
        for j in range(n):
            g0, g1 = ar_data[j], dit_data[j]
            if g0 > 0 and g1 > 0:
                both_active += 1
            elif g0 > 0 and g1 == 0:
                gpu0_only += 1
            elif g0 == 0 and g1 > 0:
                gpu1_only += 1
            else:
                both_idle += 1

        lines.append("--- Pipeline Phase Analysis ---")
        lines.append(f"  Sample pairs:          {n}")
        lines.append(f"  Both GPU active:       {both_active:4d}/{n}"
                      f" ({100*both_active/n:5.1f}%)  <- pipeline overlap")
        lines.append(f"  GPU {ar_gpu} only (AR):      {gpu0_only:4d}/{n}"
                      f" ({100*gpu0_only/n:5.1f}%)  <- AR prefill, DiT idle")
        lines.append(f"  GPU {dit_gpu} only (DiT):     {gpu1_only:4d}/{n}"
                      f" ({100*gpu1_only/n:5.1f}%)  <- DiT denoising, AR idle")
        lines.append(f"  Both idle:             {both_idle:4d}/{n}"
                      f" ({100*both_idle/n:5.1f}%)")
        lines.append("")

        # ----- DiT SM distribution -----
        dit_active = [v for v in dit_data if v > 0]
        if dit_active:
            bins = [(0, 50, "<50%"), (50, 70, "50-70%"),
                    (70, 90, "70-90%"), (90, 101, "90-100%")]
            lines.append(f"--- GPU {dit_gpu} SM Distribution (when active) ---")
            for lo, hi, label in bins:
                c = sum(1 for v in dit_active if lo <= v < hi)
                bar = "#" * int(40 * c / len(dit_active))
                lines.append(f"  {label:>10s}: {c:4d}"
                              f" ({100*c/len(dit_active):5.1f}%) {bar}")
            lines.append("")

        # ----- Key findings -----
        lines.append("=" * 60)
        lines.append("  Key Findings")
        lines.append("=" * 60)
        ar_pct = 100 * (gpu0_only + both_active) / n
        dit_pct = 100 * (gpu1_only + both_active) / n
        overlap_pct = 100 * both_active / n
        lines.append(f"  * GPU {ar_gpu} busy {ar_pct:.1f}% of time (AR stage)")
        lines.append(f"  * GPU {dit_gpu} busy {dit_pct:.1f}% of time (DiT stage)")
        lines.append(f"  * Pipeline overlap: {overlap_pct:.1f}% of time")
        bottleneck = f"DiT (GPU {dit_gpu})" if dit_pct > ar_pct else f"AR (GPU {ar_gpu})"
        lines.append(f"  * Bottleneck: {bottleneck}")
        idle_pct = 100 * both_idle / n
        if idle_pct > 5:
            lines.append(f"  * WARNING: Both GPUs idle {idle_pct:.1f}%"
                          " - possible stage handoff overhead")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GPU SM utilization from nvidia-smi dmon log")
    parser.add_argument("log_file", help="Path to nvidia-smi dmon log file")
    parser.add_argument("--gpus", default="0,1",
                        help="Comma-separated GPU IDs to analyze (default: 0,1)")
    parser.add_argument("--output", "-o", default=None,
                        help="Write report to file (default: stdout only)")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]

    gpu_sm = parse_dmon_log(args.log_file, gpu_ids)

    header = [
        "=" * 60,
        "  GPU SM Utilization Analysis",
        f"  Source: {args.log_file}",
        f"  GPUs:   {gpu_ids}",
        "=" * 60,
        "",
    ]
    body = analyze(gpu_sm, gpu_ids)
    report = "\n".join(header + body)

    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
