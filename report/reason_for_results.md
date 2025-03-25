## for the small-sized matrix multiplication
+ We set the size of shared block to 16x16, and for the test case of 128x128 matrix, we find that the speedup is much higher than the case of bigger-sized matrix(3:1 approximately). 

+ The reason we think is that:
    1. when the matrix size is small, all data is loaded into the shared memory with fewer times, emliminating repeated accesses to the global memory.
    2. the loop runs fewer times, so synchronization via `__syncthreads()` occurs fewer times, which is time-consuming.

## for the direct addition of two vectors
+ We find that the speedup is less than 1, which is not what we expected. The reason we think is that:
    1. the overhead of copying data to the shared memory is not negligible, and the time saved by shared memory is not enough to compensate for the overhead.
    2. the shared memory is not fully utilized, as the data is not reused in the next iteration.

## for the reduction addition 
+ We find that the speedup is unexpectedly high(for the test cases, the speedup is 13.97 and 18.33), the reason we think is that:
    1. Without shared memory, the reduction addition we implemented is direct but time-consuming, since every thread has to access one same memory address to update the sum. It is almost the same as the serial version.
    2. With shared memory, we use a tree-like structure to reduce the number of accesses to the global memory, every turn, the number of accesses is halved. The complexity is reduced from $O(n)$ to $O(\logn)$. And every block just needs execute the atomic operation once to add the result to the global memory.
     
