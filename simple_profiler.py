import torch
import time
import importlib.util
import pathlib
import sys

def load_module_from_path(module_path):
    """Load a Python module from a file path."""
    module_path = pathlib.Path(module_path)
    name = module_path.stem
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def profile_model(model_path, num_warmup=10, num_runs=100):
    """Profile a PyTorch model's execution time.
    
    Args:
        model_path: Path to the Python file containing the model
        num_warmup: Number of warmup runs
        num_runs: Number of actual timing runs
    """
    print(f"\nProfiling model: {model_path}")
    
    # Load the model
    module = load_module_from_path(model_path)
    model = module.Model().cuda()
    
    # Get input tensors
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in module.get_inputs()]
    
    # Warmup runs
    print("Warming up...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(*inputs)
    
    # Actual timing runs
    print(f"Running {num_runs} iterations...")
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(*inputs)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("\nResults:")
    print(f"Average time: {avg_time*1000:.3f} ms")
    print(f"Min time:     {min_time*1000:.3f} ms")
    print(f"Max time:     {max_time*1000:.3f} ms")
    
    return {
        'average_ms': avg_time * 1000,
        'min_ms': min_time * 1000,
        'max_ms': max_time * 1000,
        'all_times_ms': [t * 1000 for t in times]
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_profiler.py <path_to_model_file>")
        sys.exit(1)
        
    profile_model(sys.argv[1]) 