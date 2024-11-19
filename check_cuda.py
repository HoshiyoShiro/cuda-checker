import torch

def check_cuda_support():
    """
    Check if CUDA is available and gather information about the GPUs on the system.
    """
    try:
        # Check if CUDA is available
        is_cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {is_cuda_available}")
        
        if is_cuda_available:
            # Get the number of GPUs
            device_count = torch.cuda.device_count()
            print(f"Number of GPUs: {device_count}")
            
            # Print details of each GPU
            for i in range(device_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Print current device
            print(f"Current Device: {torch.cuda.current_device()}")
        else:
            print("CUDA is not available on this machine.")
    
    except Exception as e:
        print(f"An error occurred while checking CUDA support: {e}")

if __name__ == "__main__":
    print("=== CUDA Checker ===")
    check_cuda_support()
