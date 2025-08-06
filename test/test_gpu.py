import torch

print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")

    for device_id in range(device_count):
        try:
            x = torch.randn(10, 10, device=f'cuda:{device_id}')
            print(f"[GPU {device_id}] Successfully allocated tensor.")
        except RuntimeError as e:
            print(f"[GPU {device_id}] Error allocating tensor: {e}")
else:
    print("No CUDA devices found. Running on CPU.")

