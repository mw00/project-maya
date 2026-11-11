# DeepSeek: ordering CPU-lane workers across chiplets

`MAYA_DS4_CPU_LANE_CPUS` sets the order in which a single CPU-lane part pins its workers. It is intended for CPUs where the natural numeric CPU order leaves some CCD/L3 groups idle even though other groups have several workers. Only the expert workers are reordered; the rest of the engine keeps its existing affinity.

The setting is off by default. With no list, behavior is unchanged. With a list:

- CPUs are used in the given order, after filtering to the CPUs the process may use.
- Duplicate and disallowed entries are ignored.
- Every unlisted allowed CPU is appended in its original order.
- `--cpu-threads N` still limits the total lane size; it does not need to equal the list length.
- It applies to one CPU-lane part. It does not assign workers between NUMA parts.
- It changes worker affinity only, not the arithmetic, expert placement, routing or draft settings.

The engine logs `ds4: CPU lane order: DS4_CPU_LANE_CPUS (...)` when the option is active.

## Example: eight L3 groups of four cores

On an EPYC 7452 with 32 physical CPUs visible as 0–31 and L3 groups 0–3, 4–7, …, 28–31, the tested 22-thread layout used three workers from most groups and two from the others:

```json
{
  "args": ["--cpu-threads", "22"],
  "env": {
    "MAYA_DS4_CPU_LANE_CPUS": "0,1,2,4,5,6,8,9,10,12,13,14,16,17,20,21,24,25,28,29,30,31"
  }
}
```

The list includes every allowed CPU, but the 22-thread limit uses only the service CPU and first 21 workers. Check the machine's actual topology rather than copying these IDs: `/sys/devices/system/cpu/cpuN/cache/index3/shared_cpu_list` shows each CPU's L3 group, and `taskset -pc PID` verifies the process's allowed CPUs.

Do **not** apply the same small CPU mask to the complete engine with `taskset`. That also restricts GPU, Engram, reader and server threads and was much slower in the test below. Likewise, increasing the lane to every core reduced isolated expert time but regressed whole-model decode.

## Measurements, 2026-11-11

Hardware and model: EPYC 7452 (32 physical cores, eight L3 groups), one RTX 4090D 48GB, DeepSeek-V4.1-Flash UD-IQ1_M + MXFP8 Engram, 262K context, parallel 2, DSpark 0.65. The implementation tested was the v1.1.57 backport of this option.

Every arm used the same frozen read-only expert profile, fresh lane calibration, empty conversation slots, three fresh 32,768-token prompts and exactly 2,048 greedy output tokens, with zero cache reuse. Each A/B pair used the same binary. Six pairs were run with GPU temperature at most 36°C before each arm and alternating order.

| Metric | Natural order | Ordered spread, 22 threads | Change |
|---|---:|---:|---:|
| Median decode over 18 runs | 66.442 tok/s | 72.979 tok/s | **+9.84%** |
| Median 32K prefill over 18 runs | 2543.2 tok/s | 2549.8 tok/s | **+0.26%** |
| Prompt-matched decode ratio | — | 1.092x | **+9.2%** |
| DSpark acceptance | 91.38% | 93.67% | +2.29 points |

Five of six A/B pairs favored the ordered layout. Their median-decode changes were +10.63%, +4.84%, +14.31%, +40.86%, -11.16% and +12.89%; 13 of 18 prompt-matched runs favored it. These values show a repeatable direction, not a guaranteed per-request gain. Acceptance differences and startup expert timings show that the result is not a pure memory-bandwidth measurement.

An independent read-only bandwidth test gave 95.33GB/s with contiguous 22 threads, 112.92GB/s with spread 22, 114.49GB/s with spread 24 and 115.96GB/s with all 32. Thus spread-22 already reached 97.4% of the all-32 result. Additional workers added little bandwidth but increased coordination and contention with the rest of the engine.
