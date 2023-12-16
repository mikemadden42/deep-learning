#!/usr/bin/env python3

import tensorflow as tf
import timeit


def create_random_matrix(matrix_size):
    return tf.random.normal((matrix_size, matrix_size))


def matmul_operation(matrix):
    return tf.matmul(matrix, matrix)


def benchmark_cpu(matrix_size):
    matrix_cpu = create_random_matrix(matrix_size)
    start_time = timeit.default_timer()
    result_cpu = matmul_operation(matrix_cpu)
    end_time = timeit.default_timer()
    time_cpu = end_time - start_time
    print(f"CPU (Size {matrix_size}): {time_cpu} seconds")
    return time_cpu


def benchmark_gpu(matrix_size):
    if tf.config.list_physical_devices("GPU"):
        matrix_gpu = create_random_matrix(matrix_size)
        with tf.device("/GPU:0"):
            start_time = timeit.default_timer()
            result_gpu = matmul_operation(matrix_gpu)
            end_time = timeit.default_timer()
        time_gpu = end_time - start_time
        print(f"GPU (Size {matrix_size}): {time_gpu} seconds")
        return time_gpu
    else:
        print("No GPU available.")
        return None


def main():
    matrix_size = 10000

    # Benchmark CPU
    time_cpu = benchmark_cpu(matrix_size)

    # Benchmark GPU
    time_gpu = benchmark_gpu(matrix_size)

    # Print speedup factor if GPU is used
    if time_gpu is not None:
        speedup_factor = time_cpu / time_gpu
        print(f"Speedup Factor: {speedup_factor:.2f}")


if __name__ == "__main__":
    main()
