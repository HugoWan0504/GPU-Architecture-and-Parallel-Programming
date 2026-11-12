# A2 Reduction Report

## Q1 Naive reduction and divergence

### Q1.1

**Answer: 1 step without warp divergence.**

At stride = 1, all 1024 threads participate in the reduction, so all 32 threads within each warp are active.

### Q1.2

**Answer: 10 steps with warp divergence.**

The reduction requires log₂(2048) = 11 steps. Starting at stride = 2, the condition `tid % stride == 0` causes active and inactive threads to become interleaved within warps, introducing warp divergence for the remaining 10 steps.


## Q2 Optimized reduction and divergence

### Q2.1

**Answer: 6 steps without warp divergence.**

For strides 1024, 512, 256, 128, 64, and 32, active threads remain contiguous and occupy complete warps of 32 threads. Therefore, these 6 steps execute without warp divergence.

### Q2.2

**Answer: 5 steps with warp divergence.**

Starting at stride = 16, fewer than 32 threads participate within the remaining active warp. Consequently, strides 16, 8, 4, 2, and 1 introduce warp divergence, resulting in 5 divergent steps.


## Q3. Number of Steps

**Answer: 10 reduction steps per block, across 500 blocks.**

With 512 threads per block and 2 elements processed per thread, each block handles 1024 elements.

- Number of blocks = 512000 / 1024 = 500
- Reduction steps per block = log₂(1024) = 10

Each block independently performs 10 reduction steps and writes one partial sum to the output array.

Therefore, the algorithm requires **10 reduction stages**, with 500 blocks performing a total of 5000 block-level reduction steps. No additional reduction stage is needed because the assignment only requires partial sums.


## Q4. Warp Occupancy Distribution

**Experimental configuration:**
- Input size: 10,240 elements
- Thread block size: 512 threads
- Simulator: GPGPU-Sim, GTX480 configuration

### Q4.1 — Naive Reduction

```text
Warp Occupancy Distribution:
Stall:1055    W0_Idle:26329    W0_Scoreboard:20580
W1:3780  W2:1920  W3:0  W4:1920  W5:0  W6:0  W7:0  W8:1920
W9:0  W10:0  W11:0  W12:0  W13:0  W14:0  W15:0  W16:1920
W17:0 W18:0 W19:0 W20:0 W21:0 W22:0 W23:0 W24:0
W25:0 W26:0 W27:0 W28:0 W29:0 W30:0 W31:0 W32:21920
```

Total simulated cycles: 4090.

### Q4.2 — Optimized Reduction

```text
Warp Occupancy Distribution:
Stall:174    W0_Idle:27201    W0_Scoreboard:17547
W1:140  W2:80  W3:0  W4:80  W5:0  W6:0  W7:0  W8:80
W9:0  W10:0  W11:0  W12:0  W13:0  W14:0  W15:0  W16:80
W17:0 W18:0 W19:0 W20:0 W21:0 W22:0 W23:0 W24:0
W25:0 W26:0 W27:0 W28:0 W29:0 W30:0 W31:0 W32:19280
```

Total simulated cycles: 3243.

### Q4.3 — Which distribution is more desirable, and why?

**The optimized reduction has the more desirable warp occupancy distribution.**

The optimized kernel keeps active threads contiguous, allowing more warps to execute with all 32 threads active and reducing partially active warp executions.

For example, W16 decreases from 1920 in the naive version to 80 in the optimized version, while W1 decreases from 3780 to 140.

Although the naive kernel has a higher absolute W32 count, the optimized version performs substantially fewer partially active warp executions.

The optimized kernel also requires fewer simulated GPU cycles (3243 versus 4090), supporting its improved execution efficiency for this experiment.


## Q5. Block Size

**Experimental configuration:**
- Input size: 10,240 elements
- Block sizes: 256, 512, and 1024 threads
- Simulator: GPGPU-Sim, GTX480 configuration

### Q5.1 — Compared with 1024 threads, what trends do you notice in the warp occupancy distribution?

| Block size | Kernel | W1 | W2 | W4 | W8 | W16 | W32 |
|---|---|---:|---:|---:|---:|---:|---:|
| 256 | Naive | 3720 | 1920 | 1920 | 1920 | 1920 | 20480 |
| 512 | Naive | 3780 | 1920 | 1920 | 1920 | 1920 | 21920 |
| 1024 | Naive | 3810 | 1920 | 1920 | 1920 | 1920 | 23360 |
| 256 | Optimized | 280 | 160 | 160 | 160 | 160 | 18080 |
| 512 | Optimized | 140 | 80 | 80 | 80 | 80 | 19280 |
| 1024 | Optimized | 70 | 40 | 40 | 40 | 40 | 20440 |

**Answer:**

For the naive reduction, decreasing the block size from 1024 to 256 slightly reduces W1 and W32, while W2, W4, W8, and W16 remain unchanged. The naive kernel consistently exhibits many partially active warp executions because of its interleaved thread indexing.

For the optimized reduction, smaller block sizes increase the counts of partially active warps (W1–W16), while reducing W32. However, the optimized reduction still has substantially fewer partially active warp executions than the naive reduction at every tested block size.

Therefore, changing block size affects the occupancy distribution, but the thread-indexing strategy has a stronger influence on warp divergence.

### Q5.2 — Compared with 1024 threads, what performance trends do you notice? Say which metric you used to measure performance.

**Performance metric:** `gpu_tot_sim_cycle`, the total number of simulated GPU cycles.

| Block size | Naive (cycles) | Optimized (cycles) |
|---|---:|---:|
| 256 | 3844 | 2956 |
| 512 | 4090 | 3243 |
| 1024 | 5967 | 4194 |

**Answer:**

For both implementations, decreasing the block size from 1024 to 512 and then to 256 reduces the total simulated GPU cycles.

The naive kernel decreases from 5967 cycles at 1024 threads to 3844 cycles at 256 threads. Similarly, the optimized kernel decreases from 4194 cycles to 2956 cycles.

The optimized reduction consistently requires fewer simulated cycles than naive reduction at all three block sizes.

These results suggest that smaller block sizes provide better performance for this particular workload and simulator configuration, potentially due to differences in scheduling, resource utilization, and execution overhead.

However, this does not establish that smaller blocks are always faster for other GPU architectures or input sizes.


