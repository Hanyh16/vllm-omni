#!/usr/bin/env python3
"""Analyze GPU SM utilization from nvidia-smi dmon log."""

gpu0_sm = []
gpu1_sm = []
gpu0_active = []
gpu1_active = []

with open("gpu_sm_log.csv") as f:
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
        except ValueError:
            continue
        if gpu_id == 0:
            gpu0_sm.append(sm)
            if sm > 0:
                gpu0_active.append(sm)
        elif gpu_id == 1:
            gpu1_sm.append(sm)
            if sm > 0:
                gpu1_active.append(sm)

# Pair samples
n = min(len(gpu0_sm), len(gpu1_sm))
both_active = gpu0_only = gpu1_only = both_idle = 0
for j in range(n):
    g0, g1 = gpu0_sm[j], gpu1_sm[j]
    if g0 > 0 and g1 > 0:
        both_active += 1
    elif g0 > 0 and g1 == 0:
        gpu0_only += 1
    elif g0 == 0 and g1 > 0:
        gpu1_only += 1
    else:
        both_idle += 1

print("=" * 60)
print("  GPU SM Utilization Analysis")
print("=" * 60)
print()
print(f"Total sample pairs: {n}")
print()

print("--- GPU 0 (Stage-0: AR/LLM) ---")
print(f"  Overall avg SM:        {sum(gpu0_sm)/len(gpu0_sm):.1f}%")
print(f"  Active samples:        {len(gpu0_active)}/{len(gpu0_sm)} ({100*len(gpu0_active)/len(gpu0_sm):.1f}%)")
if gpu0_active:
    print(f"  Active avg SM:         {sum(gpu0_active)/len(gpu0_active):.1f}%")
    print(f"  Active min/max SM:     {min(gpu0_active)}% / {max(gpu0_active)}%")
print()

print("--- GPU 1 (Stage-1: DiT/Diffusion) ---")
print(f"  Overall avg SM:        {sum(gpu1_sm)/len(gpu1_sm):.1f}%")
print(f"  Active samples:        {len(gpu1_active)}/{len(gpu1_sm)} ({100*len(gpu1_active)/len(gpu1_sm):.1f}%)")
if gpu1_active:
    print(f"  Active avg SM:         {sum(gpu1_active)/len(gpu1_active):.1f}%")
    print(f"  Active min/max SM:     {min(gpu1_active)}% / {max(gpu1_active)}%")
print()

print("--- Pipeline Phase Analysis ---")
print(f"  Both GPU active:       {both_active:4d}/{n} ({100*both_active/n:5.1f}%)  ← pipeline overlap")
print(f"  GPU 0 only (AR):       {gpu0_only:4d}/{n} ({100*gpu0_only/n:5.1f}%)  ← AR prefill, DiT idle")
print(f"  GPU 1 only (DiT):      {gpu1_only:4d}/{n} ({100*gpu1_only/n:5.1f}%)  ← DiT denoising, AR idle")
print(f"  Both idle:             {both_idle:4d}/{n} ({100*both_idle/n:5.1f}%)")
print()

# GPU1 SM distribution when active
if gpu1_active:
    bins = [(0, 50, "<50%"), (50, 70, "50-70%"), (70, 90, "70-90%"), (90, 101, "90-100%")]
    print("--- GPU 1 SM Distribution (when active) ---")
    for lo, hi, label in bins:
        c = sum(1 for v in gpu1_active if lo <= v < hi)
        bar = "█" * int(40 * c / len(gpu1_active))
        print(f"  {label:>10s}: {c:4d} ({100*c/len(gpu1_active):5.1f}%) {bar}")
print()

# Summary
print("=" * 60)
print("  Key Findings")
print("=" * 60)
ar_pct = 100 * (gpu0_only + both_active) / n
dit_pct = 100 * (gpu1_only + both_active) / n
overlap_pct = 100 * both_active / n
print(f"  • GPU 0 busy {ar_pct:.1f}% of time (AR stage)")
print(f"  • GPU 1 busy {dit_pct:.1f}% of time (DiT stage)")
print(f"  • Pipeline overlap: {overlap_pct:.1f}% of time")
print(f"  • Bottleneck: {'DiT (GPU 1)' if dit_pct > ar_pct else 'AR (GPU 0)'}")
idle_pct = 100 * both_idle / n
if idle_pct > 5:
    print(f"  • ⚠ Both GPUs idle {idle_pct:.1f}% - possible stage handoff overhead")
