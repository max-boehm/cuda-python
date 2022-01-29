import time
import numpy as np
from numba import cuda


def matmul_python(A, B, C):
    """Perform square matrix multiplication of C = A * B"""
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp


@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B"""
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


N = 2000
print(f"Matrix dimension N = {N}")

np.random.seed(0)  # generate identical matrices each time
A = np.random.randint(10, size=(N, N))
B = np.random.randint(10, size=(N, N))
print(A)
print(B)

print()
print("Multiply A x B with numpy")
start = time.time()
C = np.matmul(A, B)
print(f"time: {time.time()-start} seconds")
print(C)

# print()
# print("Multiply A x B with python")
# C = np.zeros((N, N), np.float32)
# start = time.time()
# matmul_python(A, B, C)
# print(f"time: {time.time()-start} seconds")
# print(C)

print()
print("Multiply A x B with cuda")
C = np.zeros((N, N), np.float32)
threads_per_block = (8, 8)
blocks_per_grid = ((N + threads_per_block[0] - 1) // threads_per_block[0],
                   (N + threads_per_block[1] - 1) // threads_per_block[1])
print(f"threads_per_block={threads_per_block}, blocks_per_grid={blocks_per_grid}")
start = time.time()
matmul[blocks_per_grid, threads_per_block](A, B, C)
print(f"time: {time.time()-start} seconds")
print(C)
