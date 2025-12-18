"""
Matrix multiplication benchmark with multiprocessing
"""

import random
import time
from multiprocessing import Pool, cpu_count


def generate_matrices(n, seed=42):
    # create two random matrices as lists
    random.seed(seed)
    A = [[float(random.randint(0, 9)) for _ in range(n)] for _ in range(n)]
    B = [[float(random.randint(0, 9)) for _ in range(n)] for _ in range(n)]
    return A, B


def multiply_rows(A_rows, B):
    # compute matrix multiplication using loops
    n = len(B)
    m = len(B[0])
    num_rows = len(A_rows)
    C = [[0.0 for _ in range(m)] for _ in range(num_rows)]
    
    for i in range(num_rows):
        for j in range(m):
            s = 0.0
            for k in range(n):
                s += A_rows[i][k] * B[k][j]
            C[i][j] = s
    
    return C


def multiply_sequential(A, B):
    # use loops for matrix multiplication
    return multiply_rows(A, B)


def compute_chunk(args):
    # compute a chunk of rows using loops
    A_chunk, B = args
    return multiply_rows(A_chunk, B)


def multiply_parallel(A, B, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()
    
    n = len(A)
    rows_per_process = n // num_processes
    
    # split A into chunks for each process
    chunks = []
    for i in range(num_processes):
        start = i * rows_per_process
        if i == num_processes - 1:
            # last process gets remainder
            end = n
        else:
            end = (i + 1) * rows_per_process
        chunks.append((A[start:end], B))
    
    # compute in parallel
    with Pool(num_processes) as pool:
        results = pool.map(compute_chunk, chunks)
    
    # combine results
    C = []
    for result in results:
        C.extend(result)
    return C


def run_benchmark(n, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"\nMatrix Size: {n}x{n}")
    print(f"Processes: {num_processes}")
    
    A, B = generate_matrices(n)
    
    # sequential version
    start = time.perf_counter()
    C_seq = multiply_sequential(A, B)
    seq_time = time.perf_counter() - start
    print(f"Sequential Time: {seq_time:.6f} seconds")
    
    # parallel version
    start = time.perf_counter()
    C_par = multiply_parallel(A, B, num_processes)
    par_time = time.perf_counter() - start
    print(f"Parallel Time: {par_time:.6f} seconds")


if __name__ == "__main__":
    print(f"CPU cores: {cpu_count()}")
    
    run_benchmark(500)
    run_benchmark(1000)
    
    print("\nDone!")
