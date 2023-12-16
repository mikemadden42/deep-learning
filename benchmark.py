#!/usr/bin/env python3

import tensorflow as tf
import time

# Create random matrices
matrix_size = 10000
matrix_cpu = tf.random.normal((matrix_size, matrix_size))
matrix_gpu = tf.random.normal((matrix_size, matrix_size))


# Define a simple matrix multiplication operation
def matmul_operation(matrix):
    return tf.matmul(matrix, matrix)


# Run the operation on CPU
start_time_cpu = time.time()
result_cpu = matmul_operation(matrix_cpu)
end_time_cpu = time.time()
time_cpu = end_time_cpu - start_time_cpu
print(f"CPU Execution Time: {time_cpu} seconds")

# Run the operation on GPU
# Make sure to have a GPU available for this to work
if tf.config.list_physical_devices("GPU"):
    with tf.device("/GPU:0"):
        start_time_gpu = time.time()
        result_gpu = matmul_operation(matrix_gpu)
        end_time_gpu = time.time()
    time_gpu = end_time_gpu - start_time_gpu
    print(f"GPU Execution Time: {time_gpu} seconds")
else:
    print("No GPU available.")

# Print speedup factor if GPU is used
if tf.config.list_physical_devices("GPU"):
    speedup_factor = time_cpu / time_gpu
    print(f"Speedup Factor: {speedup_factor:.2f}")
