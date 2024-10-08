#!/usr/bin/env python3

from typing import List, Union

import tensorflow as tf


def print_device_details(
    devices: List[Union[tf.config.PhysicalDevice, None]], device_type: str
) -> None:
    """
    Print details of the specified type of devices.

    Args:
        devices (List[Union[tf.config.PhysicalDevice, None]]): List of devices.
        device_type (str): Type of the devices (e.g., "CPU" or "GPU").
    """
    print(f"{device_type} Information:")
    if devices:
        for i, device in enumerate(devices):
            print(f"  {device_type} {i + 1}:")
            print(f"    Device Name: {device.name}")
            print(f"    Device Type: {device.device_type}")
            print()
    else:
        print(f"No {device_type} found.")


if __name__ == "__main__":
    cpus: List[Union[tf.config.PhysicalDevice, None]] = tf.config.list_physical_devices(
        "CPU"
    )
    gpus: List[Union[tf.config.PhysicalDevice, None]] = tf.config.list_physical_devices(
        "GPU"
    )

    print_device_details(cpus, "CPU")
    print_device_details(gpus, "GPU")
