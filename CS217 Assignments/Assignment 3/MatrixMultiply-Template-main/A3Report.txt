# A3 Tiled Matrix Multiplication Report

## Q1. Loads per element

For a 128 × 128 matrix multiplication with 16 × 16 tiles,
each element of A and B is loaded from global memory 8 times.

Calculation:
128 / 16 = 8

Each element of A is loaded once for each of the 8 output tile
columns. Within a block, it is reused to calculate 16 output elements.

Each element of B is loaded once for each of the 8 output tile
rows. Within a block, it is also reused to calculate 16 output elements.

These counts describe loads issued by the algorithm; hardware
caching can reduce the traffic reaching physical GPU memory.


## Q2. Tiled vs non-tiled

For a 256 × 256 non-tiled matrix multiplication, each element
of A and B is loaded from global memory 256 times.

Each A element is needed for all 256 output columns, and each
B element is needed for all 256 output rows. Without shared-memory
tiling, each output thread loads the values it needs separately.

Compared with Q1:
256 / 8 = 32

Therefore, the non-tiled case in Q2 performs 32 times as many
global-memory loads per input element as the tiled case in Q1.
This comparison includes both the larger matrix dimension
(256 instead of 128) and the benefit of tiling.

For an equal-size comparison, a 256 × 256 multiplication using
16 × 16 tiles would load each input element 256 / 16 = 16 times.
Thus, tiling reduces global-memory loads per element by a factor
of 16 when comparing matrices of the same size.


## Q3. Square vs rectangular matrices

Measurements were collected on a Tesla T4 GPU in Google Colab
using 16 × 16 thread blocks.

Square:
A = 1024 × 1024, B = 1024 × 1024
C = 1024 × 1024
Kernel time = 0.003523 s (3.523 ms)

Rectangular:
A = 4096 × 256, B = 256 × 4096
C = 4096 × 4096
Kernel time = 0.017161 s (17.161 ms)

The square multiplication was approximately 4.87 times faster.

Both cases have 1,048,576 elements in each input matrix.
However, the rectangular case produces 16,777,216 output elements,
compared with 1,048,576 in the square case.

Since each thread calculates one output element, the rectangular
case launches 16 times as many threads: 16,777,216 versus 1,048,576.
These threads execute in blocks over time, rather than all at once.

Each rectangular output requires 256 multiply-adds, compared with
1024 for each square output. Therefore, total arithmetic increases
by 16 × (256 / 1024) = 4 times.

The rectangular case also writes 16 times as many output elements.
Its greater total computation and memory traffic explain why it
takes longer despite having equally sized input matrices.

These are single-run Colab measurements; the assignment requests
measurements on Bender.


## Q4. GPGPU-Sim memory statistics

Configuration: simulated NVIDIA GTX480
Matrix dimensions: 128 × 128 for A, B, and C
All four configurations passed verification.

```
Tile    Global loads   Global stores   Shared instructions   Cycles
4×4     1,048,576      16,384          5,242,880             159,651
8×8       524,288      16,384          4,718,592              40,956
16×16     262,144      16,384          4,456,448              27,204
32×32     131,072      16,384          4,325,376              59,650
```

### Q4.1. Global-memory accesses

The 32×32 tile produced the fewest global-memory accesses,
while the 4×4 tile produced the most.

Larger tiles allow each input value loaded into shared memory
to be reused for more output elements within the block.

For square matrices of dimension N and square tiles of width T,
where N is divisible by T, the total input-element load count is:

Global loads = 2 × N² × (N / T)

```text
For N = 128:
T = 4:  2 × 128² × 32 = 1,048,576
T = 8:  2 × 128² × 16 =   524,288
T = 16: 2 × 128² ×  8 =   262,144
T = 32: 2 × 128² ×  4 =   131,072
```

These values match the simulator results.

Global stores remain at 128 × 128 = 16,384 for every tile size,
because each output element is written exactly once.


### Q4.2. Performance

The 16×16 tile was fastest at 27,204 simulated cycles.
The 4×4 tile was slowest at 159,651 cycles.

The 4×4 configuration performs the most global loads and requires
the most tile iterations and synchronization steps. Each block
also contains only 16 threads, leaving half of a 32-thread warp
inactive. These factors contribute to its poor performance.

The 32×32 configuration performs the fewest global loads, but
takes 59,650 cycles, approximately 2.19 times the 16×16 result.
Therefore, reducing global loads alone does not guarantee the
best performance.

The simulated GTX480 supports at most 1536 resident threads
per SM. A 32×32 block contains 1024 threads, so only one such
block can reside on an SM, leaving capacity for 512 threads
that cannot fit another block of that size.

A 16×16 block contains 256 threads. Based on the thread limit
alone, up to six such blocks could reside on an SM, providing
more opportunities to hide memory and synchronization latency.
Actual residency also depends on registers, shared memory,
and the hardware block limit.

For this small matrix, 32×32 tiles also produce only 16 blocks
in total, compared with 64 blocks for 16×16 tiles, limiting
the available block-level parallelism.

The measured results suggest that 16×16 achieves a better
balance between data reuse and parallelism for this workload.


## Q5. Tile shape

### Q5.1. Loads for different tile shapes

Each block computes TILE_Y rows and TILE_X columns of C.

Each A element is loaded once per output block column:
Loads per A element = ceil(n / TILE_X)

Each B element is loaded once per output block row:
Loads per B element = ceil(m / TILE_Y)

```text
Total input-element loads:
m × k × ceil(n / TILE_X)
+ k × n × ceil(m / TILE_Y)
```

For 1024 × 1024 matrices:

```text
Shape     Loads/A element   Loads/B element   Total loads
32×32     32                32                 67,108,864
16×64     64                16                 83,886,080
64×16     16                64                 83,886,080
8×128     128                8                142,606,336
128×8      8               128                142,606,336
```

The 32×32 shape loads the least total input data.
The 8×128 and 128×8 shapes tie for the most.


### Q5.2. Measured performance and warp mapping

Measurements used 1024 × 1024 matrices on a Tesla T4 GPU
in Google Colab.
Each shape contains 1024 threads per block.
Times are averages of five runs after one excluded warm-up run.

```text
Shape (X×Y)    Average kernel time
32×32          3.889 ms
16×64          4.396 ms
64×16          4.871 ms
8×128          6.573 ms
128×8          6.063 ms
```

The fastest shape was 32×32, and the slowest was 8×128.
The slowest shape took approximately 1.69 times as long
as the fastest.

The overall trend agrees with Q5.1. The 32×32 shape loads
the least total input data. The 16×64 and 64×16 shapes
load more, while 8×128 and 128×8 load the most.

However, shapes with equal total load counts have different
measured times. Load count alone does not describe access
efficiency, shared-memory behavior, or scheduling overhead.

CUDA forms warps with threadIdx.x varying fastest.
Since the matrices are row-major, neighboring columns
are stored at neighboring memory addresses.

For TILE_X = 32, each warp covers 32 consecutive columns
of one output row. This provides a natural mapping to
contiguous row-major accesses.

For TILE_X = 16, each warp covers two rows, with 16
consecutive columns per row. For TILE_X = 8, each warp
covers four rows, with 8 consecutive columns per row.

For TILE_X = 64 or 128, each warp still covers 32
consecutive columns within one row. However, in this
implementation, only threads with threadIdx.x < TILE_K
load A, where TILE_K = min(TILE_X, TILE_Y). Therefore,
some warps have few or no active A-loading threads.
The wider block does not automatically make all loads
more efficient.

The extreme shapes also use TILE_K = 8, requiring 128
tile iterations for k = 1024. The 32×32 shape uses
TILE_K = 32 and needs only 32 iterations. Each iteration
includes two synchronization barriers, so extreme shapes
have greater synchronization and loop overhead.

These factors help explain why 32×32 performed best.
The smaller differences between shapes with equal load
counts may reflect access patterns, shared-memory behavior,
and measurement variability. Additional profiling would
be needed to identify their exact causes.

These measurements were collected on Colab rather than Bender.



