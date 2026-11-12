# A1 Matrix Addition Report

## Q1. Grid and Block Dimensions

```
Blocks in x = ceil(1023 / 16) = 64
Blocks in y = ceil(4097 / 16) = 257

Total threads = 64 × 257 × 256 = 4,210,688
Valid elements = 4097 × 1023    = 4,191,231
Threads doing no work         = 19,457
```


## Q2. Kernel time vs matrix size

I measured seven square matrix sizes using 16 x 16 thread blocks
on a Tesla T4 GPU. Each size was tested five times, and I reported
the median time.

```
Matrix size    Elements      Data moved (MB)    Time (ms)
16 x 16      256         0.003072         0.118
64 x 64      4096         0.049152        0.114
256 x 256     65536        0.786432         0.124
1024 x 1024    1048576       12.582912        0.182
4096 x 4096    16777216      201.326592        1.001
8192 x 8192    67108864      805.306368        3.396
16384 x 16384   268435456     3221.225472        13.047
```

Each output element requires reading one float from A, reading
one float from B, and writing one float to C. Each float is
4 bytes, so the kernel moves:

Data moved = N² x (4 + 4 + 4) = 12N² bytes.

Here, 1 MB = 1,000,000 bytes. This counts GPU-memory reads and
writes, excluding transfers between the CPU and GPU.

For large matrices, time grows approximately in proportion to
the number of elements and the amount of data moved. Increasing
N from 8192 to 16384 quadruples the data, while measured time
increases by about 3.84 times. This is consistent with memory
bandwidth becoming the main performance limit.

Small matrices take approximately 0.11–0.12 ms despite having
different numbers of elements. For these sizes, fixed launch
and synchronization overhead contributes much of the reported
time. Timing variability can also explain why 64 x 64 is
slightly faster than 16 x 16.

The provided timer measures the kernel launch and synchronization,
rather than only GPU execution time. The largest matrix tested
was 16384 x 16384; this was not established as the memory limit.


## Q3. Data Transfer vs Computation

Each square matrix size was tested five times. The table reports
the median time for each stage, in milliseconds.

```
N    Host to device    Kernel    Device to host    Total
16      0.031       0.114       0.016       0.161
64      0.048       0.104       0.029       0.181
256      0.233      0.106       0.216        0.555
1024     2.172       0.182       4.936       7.290
4096     29.331      0.933       59.786       90.050
8192     118.035      3.407      258.308       379.750
16384    470.938      13.176      848.399       1332.513
```

Total is the sum of these three stage medians. It excludes
memory allocation, initialization, and verification.

1. Kernel percentage for the largest matrix

For the 16384 x 16384 matrix:

Kernel percentage = 13.176 / 1332.513 x 100 = 0.99%

The two transfers account for approximately 99.01% of the
combined time. Thus, data movement dominates this workload.

2. Data moved and comparison with the kernel

Copying A and B from the CPU to the GPU moves:
2 x N² x 4 = 8N² bytes.

Copying C from the GPU to the CPU moves:
N² x 4 = 4N² bytes.

The two copies together move 12N² bytes. For N = 16384:

12 x 16384² = 3,221,225,472 bytes = 3221.225472 MB.

Here, 1 MB = 1,000,000 bytes.

This equals the kernel's logical data movement in Q2:
reading A, reading B, and writing C.

However, the data travels along different paths. The copies
cross the CPU–GPU connection, while the kernel reads and
writes GPU memory locally. GPU memory has much higher bandwidth.

The combined transfer time is:

470.938 + 848.399 = 1319.337 ms.

This is approximately 100.13 times the kernel-stage time of
13.176 ms. Copy timings can also include host-memory staging
and other overhead, so equal byte counts do not imply equal
execution times.

3. Running the kernel 100 times before copying C back

Assume A and B are copied to the GPU once, the kernel runs
100 times, and C is copied back once. Also assume each launch
takes the same measured kernel time.

Combined kernel time = 100 x 13.176 = 1317.600 ms.

Total time = 470.938 + 1317.600 + 848.399 = 2636.937 ms.

Kernel percentage = 1317.600 / 2636.937 x 100 = 49.97%.

The kernel's share increases from 0.99% to 49.97% because the
same transfer cost is spread over more computation.

This suggests that GPU execution is more beneficial when data
stays on the GPU and is reused for substantial computation.
A CPU baseline is still needed to establish actual speedup.

The 100-launch result is an estimate. Repeated launches may
differ from the measured first launch, whose reported time
includes launch and synchronization overhead.


## Q4. Block Size

I tested a 4096 x 4096 matrix with four block sizes.
Each configuration was run five times, and I recorded
the median reported kernel time.

```
Block size   Threads per block   Kernel time (ms)
4 x 4         16           2.527
8 x 8         64           0.941
16 x 16        256           0.943
32 x 32        1024          0.930
```

The 4 x 4 configuration was approximately 2.7 times slower
than the other configurations.

A warp contains 32 threads. A 4 x 4 block contains only
16 threads, so it occupies a warp with half its lanes unused.
It also requires many more blocks to cover the matrix,
increasing block scheduling overhead. These factors help
explain its lower performance.

The other configurations contain whole multiples of 32
threads and achieve similar measured times. Although
32 x 32 had the lowest median, its advantage was small
and may reflect timing variability.

Larger blocks do not necessarily keep improving performance.
Memory bandwidth and resource limits also affect execution.
The reported timing includes launch and synchronization
overhead, which can obscure small performance differences.

For make TILE=64, compilation succeeded, but execution failed
at the kernel launch with:

- CUDA error: invalid argument

A 64 x 64 block requests 4096 threads, exceeding the GPU's
1024-thread-per-block limit. Therefore, this launch is invalid.

After the experiment, I ran plain make to restore the
default 16 x 16 block configuration.


## Q5. Block Shape and Memory Access

I tested a 4096 x 4096 matrix with five block shapes.
Each block contained 256 threads. Each configuration was
run five times, and I recorded the median reported kernel time.

```
Block shape   Threads per block   Kernel time (ms)   Correctness
256 x 1       256                 0.911              PASSED
32 x 8        256                 0.896              PASSED
16 x 16       256                 0.924              PASSED
8 x 32        256                 0.937              PASSED
1 x 256       256                 5.796              PASSED        PASSED
```

The 32 x 8 shape had the lowest median time, while 1 x 256
was approximately 6.47 times slower. The first four shapes
performed similarly, so their small timing differences
should not be treated as a definitive ranking.

The matrices use row-major storage: adjacent columns are
adjacent in memory. The kernel maps the x coordinate to
columns and the y coordinate to rows. Within a block,
thread indices advance along x first.

For each 32-thread warp:

- 256 x 1: accesses 32 adjacent columns in one row.
- 32 x 8: accesses 32 adjacent columns in one row.
- 16 x 16: accesses 16 adjacent columns in each of two rows.
- 8 x 32: accesses 8 adjacent columns in each of four rows.
- 1 x 256: accesses one column across 32 different rows.

The first four shapes access contiguous groups of elements,
allowing efficient memory transactions. In particular,
8 adjacent floats occupy 32 bytes, so the 8 x 32 shape
can still use memory transactions efficiently when aligned.

For 1 x 256, neighboring warp threads access different rows.
With 4096 columns, these addresses are 4096 floats, or
16384 bytes, apart. These scattered accesses require many
more memory transactions for the same useful data, explaining
the much longer execution time.

All five shapes produced correct results in every measured
run. Changing block shape changes memory-access efficiency,
but the indexing and bounds check still ensure that each
valid output element is computed correctly.

The experiment restored the default 16 x 16 blocks afterward.

