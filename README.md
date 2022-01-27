# CUDA experiments with Python

The `matmul.py` file contains Python code for matrix multiplication with NumPy,
Python, and CUDA. The program generates two matrices A and B and multiplies them
using the three approaches. For each approach the elapsed time is printed.

The CUDA code is implemented with the help of Numba, see [Numba for CUDA GPUs](https://numba.readthedocs.io/en/stable/cuda/index.html).
It still follows the concepts explained in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

## How to run

First the numpy `matmul()` function is compared with a self written naive Python
function `matmul_python()`. This can only be done for small dimensions. The
example below shows it for N = 200.

```
# python matmul.py 
Matrix dimension N = 200
[[5 0 3 ... 8 2 3]
 [9 7 5 ... 4 0 0]
 [8 4 6 ... 3 7 9]
 ...
 [4 9 1 ... 4 5 9]
 [9 4 0 ... 0 0 9]
 [4 6 0 ... 0 5 9]]
[[5 2 6 ... 6 3 0]
 [7 7 0 ... 2 1 5]
 [9 1 4 ... 7 0 5]
 ...
 [1 0 8 ... 0 9 8]
 [3 4 9 ... 4 8 3]
 [8 1 9 ... 5 1 4]]

Multiply A x B with numpy
time: 0.008651971817016602 seconds
[[4047 3968 4008 ... 3925 4108 3894]
 [4217 4156 4187 ... 4248 4187 3933]
 [3913 4041 3953 ... 4285 3895 3986]
 ...
 [3854 3993 4180 ... 4219 4201 4051]
 [3862 3935 3626 ... 4059 3779 3538]
 [4119 4409 3970 ... 4054 4188 4037]]

Multiply A x B with python
time: 3.1456005573272705 seconds
[[4047. 3968. 4008. ... 3925. 4108. 3894.]
 [4217. 4156. 4187. ... 4248. 4187. 3933.]
 [3913. 4041. 3953. ... 4285. 3895. 3986.]
 ...
 [3854. 3993. 4180. ... 4219. 4201. 4051.]
 [3862. 3935. 3626. ... 4059. 3779. 3538.]
 [4119. 4409. 3970. ... 4054. 4188. 4037.]]
```

Now the numpy `matmul()` function is compared with a CUDA Kernel which is run
on the GPU by many threads in parallel. The example below shows it for N = 2000.
The naive Python function is commented out as it would take way too long.

```
$ python matmul.py 
Matrix dimension N = 2000
[[5 0 3 ... 6 4 2]
 [3 2 0 ... 7 5 6]
 [9 9 5 ... 7 0 6]
 ...
 [2 5 2 ... 4 9 3]
 [6 9 4 ... 5 3 0]
 [7 9 5 ... 8 4 0]]
[[0 0 1 ... 6 9 8]
 [7 7 1 ... 2 1 0]
 [6 5 6 ... 4 5 4]
 ...
 [3 8 0 ... 8 0 3]
 [2 8 7 ... 2 8 5]
 [3 0 7 ... 0 4 4]]

Multiply A x B with numpy
time: 13.221363067626953 seconds
[[40057 40045 39818 ... 39471 39177 40292]
 [41191 40602 41106 ... 40001 40606 40081]
 [41209 41222 41510 ... 40900 39753 40128]
 ...
 [40808 41009 40877 ... 40428 39166 40515]
 [39352 38838 38865 ... 39285 38904 39338]
 [39667 40245 38818 ... 39127 39668 40084]]

Multiply A x B with cuda
threads_per_block=(8, 8), blocks_per_grid=(250, 250)
time: 0.6066441535949707 seconds
[[40057. 40045. 39818. ... 39471. 39177. 40292.]
 [41191. 40602. 41106. ... 40001. 40606. 40081.]
 [41209. 41222. 41510. ... 40900. 39753. 40128.]
 ...
 [40808. 41009. 40877. ... 40428. 39166. 40515.]
 [39352. 38838. 38865. ... 39285. 38904. 39338.]
 [39667. 40245. 38818. ... 39127. 39668. 40084.]]
```
