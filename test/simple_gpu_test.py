import os
import torch
import torch.nn as nn
import torch.distributed as dist

def run_gpu_test():
    """
    Initializes a distributed process, assigns a GPU to the current rank,
    performs a simple tensor operation, and prints the result.
    """
    # 1. Initialize the distributed process group
    # These environment variables (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
    # are typically set by the distributed launcher (e.g., torchrun, torch.distributed.launch)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize the process group
    # 'nccl' is the recommended backend for GPU communication
    print(f"Rank {rank}/{world_size} (Local Rank: {local_rank}): Initializing process group...")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank}/{world_size} (Local Rank: {local_rank}): Process group initialized.")

    # 2. Assign the correct GPU to the current process
    # This ensures each process uses a unique GPU on the node
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"Rank {rank}/{world_size} (Local Rank: {local_rank}): Assigned GPU: {device}")

    # 3. Create a simple model and move it to the assigned GPU
    # This verifies that the GPU is accessible for model operations
    model = nn.Linear(10, 10).to(device)
    print(f"Rank {rank}/{world_size} (Local Rank: {local_rank}): Model moved to {device}")

    # 4. Create dummy input data and move it to the assigned GPU
    # This verifies data transfer to the GPU
    dummy_input = torch.randn(5, 10, device=device)
    print(f"Rank {rank}/{world_size} (Local Rank: {local_rank}): Dummy input created on {device}")

    # 5. Perform a forward pass (simple computation)
    # This verifies basic computation on the GPU
    output = model(dummy_input)
    print(f"Rank {rank}/{world_size} (Local Rank: {local_rank}): Forward pass completed on {device}. Output shape: {output.shape}")

    # 6. Perform a simple all-reduce to test inter-GPU communication (optional but good for distributed)
    # This will sum a tensor across all processes. If it hangs, NCCL communication is the issue.
    test_tensor = torch.tensor([float(rank)], device=device) # Create a tensor on the assigned device
    print(f"Rank {rank}/{world_size} (Local Rank: {local_rank}): Before all_reduce, tensor: {test_tensor}")
    dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}/{world_size} (Local Rank: {local_rank}): After all_reduce, tensor: {test_tensor}")
    # Expected result for all_reduce: sum of ranks (0+1+2+3 = 6.0)

    # 7. Clean up the distributed process group
    dist.destroy_process_group()
    print(f"Rank {rank}/{world_size} (Local Rank: {local_rank}): Distributed process group destroyed.")
    print(f"Rank {rank}/{world_size} (Local Rank: {local_rank}): GPU test successful!")

if __name__ == "__main__":
    # This ensures the test only runs if the script is executed directly
    # and not imported as a module.
    run_gpu_test()
