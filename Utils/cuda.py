import torch


def check():
    print('torch: version', torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available.")
        num_devices = torch.cuda.device_count()
        print(f"Number of GPU devices: {num_devices}")

        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")

    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
    print(f"Current device: {current_device}")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
