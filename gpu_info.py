import tensorflow as tf


def get_gpu_info():
    try:
        # Get the list of available physical devices
        physical_devices = tf.config.experimental.list_physical_devices("GPU")

        if not physical_devices:
            print("No GPU found.")
            return

        for i, device in enumerate(physical_devices):
            print(f"GPU {i + 1}:")

            # Get device details
            # device_details = tf.config.experimental.get_device_details(device.name)
            device_details = tf.config.experimental.get_device_details(device)

            for key, value in device_details.items():
                print(f"  {key}: {value}")

            print()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    get_gpu_info()
