#!/usr/bin/env python3

import tensorflow as tf


def get_gpu_info():
    try:
        # Get the list of available physical devices
        physical_devices = tf.config.experimental.list_physical_devices("GPU")

        if not physical_devices:
            raise RuntimeError("No GPU found.")

        for i, device in enumerate(physical_devices):
            if device.device_type != "GPU":
                print(f"Device {device.name} is not a GPU. Skipping.")
                continue

            print(f"GPU {i + 1}:")

            # Get device details
            device_details = tf.config.experimental.get_device_details(device)

            if device_details:
                for key, value in device_details.items():
                    print(f"  {key}: {value}")
            else:
                print("  No device details available.")

            print()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    get_gpu_info()
