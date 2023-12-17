#!/usr/bin/env python3

import tensorflow as tf
import timeit
import logging


def create_random_matrix(matrix_size):
    return tf.random.normal((matrix_size, matrix_size))


def matmul_operation(matrix):
    return tf.matmul(matrix, matrix)


def benchmark_cpu(matrix_size):
    matrix_cpu = create_random_matrix(matrix_size)
    start_time = timeit.default_timer()
    matmul_operation(matrix_cpu)
    end_time = timeit.default_timer()
    time_cpu = end_time - start_time
    logging.info(f"CPU (Size {matrix_size}): {time_cpu} seconds")
    return time_cpu


def benchmark_gpu(matrix_size):
    if tf.config.list_physical_devices("GPU"):
        matrix_gpu = create_random_matrix(matrix_size)
        with tf.device("/GPU:0"):
            start_time = timeit.default_timer()
            matmul_operation(matrix_gpu)
            end_time = timeit.default_timer()
        time_gpu = end_time - start_time
        logging.info(f"GPU (Size {matrix_size}): {time_gpu} seconds")
        return time_gpu
    else:
        logging.warning("No GPU available.")
        return None


def main():
    matrix_sizes = [2**i for i in range(10, 15)]
    print(matrix_sizes)

    for size in matrix_sizes:
        # Benchmark CPU
        time_cpu = benchmark_cpu(size)

        # Benchmark GPU
        time_gpu = benchmark_gpu(size)

        # Print speedup factor if GPU is used
        if time_gpu is not None:
            speedup_factor = time_cpu / time_gpu
            logging.info(f"Speedup Factor (Size {size}): {speedup_factor:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
