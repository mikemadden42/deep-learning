#!/usr/bin/env python3

import tensorflow as tf


def get_cpu_info():
    try:
        # Get CPU information using TensorFlow
        cpu_info = tf.config.experimental.list_physical_devices("CPU")

        if not cpu_info:
            print("No CPU found.")
            return

        print("CPU Information:")
        for i, device in enumerate(cpu_info):
            print(f"  CPU {i + 1}:")
            print(f"    Device Name: {device.name}")
            print(f"    Device Type: {device.device_type}")
            print()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    get_cpu_info()
