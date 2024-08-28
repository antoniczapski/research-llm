import torch
import time

def gpu_stress_test(device_id=1, matrix_size=1024, duration_seconds=60):
    # Set the device to the specified GPU
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    # Inform the user which device is being used
    print(f'Using device: {device}')
    
    # Create two random matrices
    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)

    # Start a timer
    start_time = time.time()
    operations = 0
    
    # Perform matrix multiplications until the specified duration is up
    while time.time() - start_time < duration_seconds:
        A @ B
        operations += 1
    
    # Output the number of operations completed
    print(f'Completed {operations} matrix multiplications in {duration_seconds} seconds.')

# Call the function to test GPU 1 for 1 minute
gpu_stress_test(device_id=1, matrix_size=1024, duration_seconds=60)
