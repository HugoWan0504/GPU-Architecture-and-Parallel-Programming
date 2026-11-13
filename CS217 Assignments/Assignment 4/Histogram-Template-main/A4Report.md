# A4 Histogram Report

## Q1. Privatized histogram

### Q1.1. Q1.1. Global memory reads

Each of the N input elements is read once from global memory.
Each of the 8 blocks merges all 2048 private bins into the global
histogram using atomicAdd, which reads the global bin value once.

Total global memory reads = N + 8 × 2048 = N + 16,384.

### Q1.2. Global memory writes

Each block merges all 2048 private bins into the global histogram.
Each atomicAdd performs one global memory write.

Total global memory writes = 8 × 2048 = 16,384.

### Q1.3. Atomic operations to shared memory

Each input element performs one atomicAdd to a shared-memory bin.

Total shared-memory atomic operations = N.

### Q1.4. Atomic operations to global memory

Each block performs 2048 atomicAdd operations to merge its private
histogram into the global histogram.

Total global-memory atomic operations = 8 × 2048 = 16,384.


## Q2 Basic histogram

### Q2.1. Atomic operations to shared memory

The basic histogram performs all atomic updates directly in
global memory.

Total shared-memory atomic operations = 0.

### Q2.2. Atomic operations to global memory

Each input element performs one atomicAdd directly to a global
histogram bin.

Total global-memory atomic operations = N.


## Q3 Basic vs privatized

### Q3.1. Kernel time versus number of bins

Both implementations were tested on an NVIDIA T4 GPU in Google
Colab with 1,000,000 input elements and 512 threads per block.
Each time below is the average of five runs. All runs passed
the CPU-reference correctness check.

```text
Bins    Basic (ms)    Privatized (ms)
1       0.954387      0.051309
32      0.285146      0.026150
256     0.336922      0.038739
1024    0.129325      0.039334
4096    0.054765      0.058822
```

The basic histogram generally becomes faster as the number of
bins increases. With one bin, every thread updates the same
global counter, causing heavy contention. More bins distribute
updates across more counters and generally reduce contention.

However, the trend is not strictly decreasing: 256 bins is slower
than 32 bins. Contention alone does not explain every timing
difference. Recent GPUs can combine atomic updates from threads
of a warp targeting the same address, which may benefit cases
with fewer bins. Memory-system behavior can also affect timing.
The measurements alone do not isolate the cause of this increase.

The privatized histogram is fastest with 32 bins. With one bin,
threads within each block still contend for the same shared-memory
counter. Increasing the number of bins reduces this contention,
but also increases the work required to initialize the private
histogram and merge its bins into global memory.

The implementation merges every bin, including bins with zero
counts. With 4096 bins and 512 threads per block, each thread
initializes and merges eight bins. The kernel launches
ceil(1,000,000 / 512) = 1954 blocks, so the merge performs
1954 × 4096 = 8,003,584 global atomic operations, compared with
1,000,000 global atomic operations in the basic kernel.

This growing initialization and merge overhead helps explain why
the privatized kernel becomes slower at larger bin counts and is
slightly slower than the basic kernel with 4096 bins.

### Q3.2. Which implementation is faster?

Speedup is calculated using the average kernel times:

Speedup = Basic time / Privatized time

```text
Bins    Speedup    Faster implementation
1       18.60×     Privatized
32      10.90×     Privatized
256     8.70×      Privatized
1024    3.29×      Privatized
4096    0.93×      Basic
```

Privatization helps the most with one bin, providing an 18.60×
speedup. The basic kernel has every input element update the same
global counter. Privatization moves counting into each block's
shared memory and reduces global atomic operations from 1,000,000
to 1954, one per block.

The benefit decreases as the number of bins grows. The basic
kernel generally experiences less contention, while the privatized
kernel must initialize and merge more bins per block.

Privatization helps the least with 4096 bins, where it provides
no speedup. The basic kernel is approximately 1.07× faster.
At this bin count, the privatized kernel performs 8,003,584 global
atomic operations during merging, including updates for zero-count
bins. This extra work outweighs the benefit of counting locally.


## Q4. GPGPU-Sim atomic statistics

Both kernels were simulated using the GTX 480 configuration with
20,000 input elements and 512 threads per block. All six runs
passed the CPU-reference correctness check.

```text
Bins    Basic cycles    Basic requests    Privatized cycles    Privatized requests
1       61,573          20,000            2,659                40
256     13,474          9,412             3,312                640
4096    10,444          18,902            11,289               10,240
```

Cycles were taken from gpu_sim_cycle. Global atomic requests were
calculated by summing the counts in braces in
traffic_breakdown_coretomem[GLOBAL_ATOMIC].

The privatized kernel launches ceil(20,000 / 512) = 40 blocks.
Each block merges every private bin into the global histogram,
including bins whose counts are zero.

### Q4.1. Global atomic requests with one bin

For N = 20,000 and 40 blocks, the expected global atomic
operation counts are:

```text
Basic:
One atomicAdd per input element.
Total = N = 20,000 operations.

Privatized:
Each block merges one bin using one global atomicAdd.
Total = 40 × 1 = 40 operations.
```

The simulator reports 20,000 requests for basic and 40 requests
for privatized. These match the predicted operation counts.

In the basic kernel, all threads update the same address.
In this simulation, these same-address atomic operations are
sent as separate requests. In the privatized kernel, only one
thread per block performs the global update, producing one
request per block.

### Q4.2. Global atomic operations versus requests

A global atomic operation is one thread's atomicAdd. A global
atomic request is a packet sent from an SM to the memory
partitions. One request can carry multiple atomic operations,
so the two counts need not match.

GPGPU-Sim groups operations from each half-warp of 16 threads
into memory requests when their addresses permit coalescing.

The basic kernel performs 20,000 global atomic operations for
both bin counts. However, the simulator reports 9,412 requests
with 256 bins and 18,902 requests with 4096 bins. With more bins,
the random updates are spread over a larger address range,
giving fewer opportunities to combine operations into requests.

The privatized kernel merges consecutive bins using consecutive
threads. Each half-warp's 16 atomic operations share one request:

```text
256 bins:
Operations = 40 × 256 = 10,240.
Requests = 10,240 / 16 = 640.

4096 bins:
Operations = 40 × 4096 = 163,840.
Requests = 163,840 / 16 = 10,240.
```

These request counts match the simulator results. The shared-memory
atomic operations used during local counting remain within the SM
and are not included in the global atomic request counts.



