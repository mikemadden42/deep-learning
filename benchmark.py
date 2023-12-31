#!/usr/bin/env python3

import tensorflow as tf
import timeit
import logging
from typing import Optional


def create_random_matrix(matrix_size: int) -> tf.Tensor:
    """
    Create a random matrix of the specified size.

    Args:
        matrix_size (int): The size of the square matrix.

    Returns:
        tf.Tensor: A random matrix of shape (matrix_size, matrix_size).
    """
    return tf.random.normal((matrix_size, matrix_size))


def matmul_operation(matrix: tf.Tensor) -> tf.Tensor:
    """
    Perform matrix multiplication on the input matrix.

    Args:
        matrix (tf.Tensor): The input matrix.

    Returns:
        tf.Tensor: The result of matrix multiplication.
    """
    return tf.matmul(matrix, matrix)


def benchmark_cpu(matrix_size: int) -> float:
    """
    Benchmark matrix multiplication on CPU.

    Args:
        matrix_size (int): The size of the square matrix.

    Returns:
        float: The time taken for matrix multiplication on CPU in seconds.
    """
    matrix_cpu = create_random_matrix(matrix_size)
    start_time = timeit.default_timer()
    matmul_operation(matrix_cpu)
    end_time = timeit.default_timer()
    time_cpu = end_time - start_time
    logging.info(f"CPU (Size {matrix_size}): {time_cpu} seconds")
    return time_cpu


def benchmark_gpu(matrix_size: int) -> Optional[float]:
    """
    Benchmark matrix multiplication on GPU if available.

    Args:
        matrix_size (int): The size of the square matrix.

    Returns:
        Optional[float]: The time taken for matrix multiplication on GPU in seconds,
        or None if no GPU is available.
    """
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
    """
    Main function to perform matrix multiplication benchmarks for different matrix sizes.
    """
    matrix_sizes = [2**i for i in range(8, 14)]
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
