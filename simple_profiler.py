import torch
import time
import importlib.util
import pathlib
import sys
import glob

def load_module_from_path(module_path):
    """Load a Python module from a file path."""
    module_path = pathlib.Path(module_path)
    name = module_path.stem
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def find_task_file(level, task, use_agent_solution=False):
    """Find the task file based on level and task number."""
    base_dir = "best_agent_solutions" if use_agent_solution else "tasks"
    pattern = f"{base_dir}/level_{level}/task_{task}.py" if use_agent_solution else f"{base_dir}/level_{level}/{task}_*.py"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No matching task file found for level {level}, task {task}")
    return matches[0]

def profile_model(model_path, num_warmup=10, num_runs=100, quiet=False):
    """Profile a PyTorch model's execution time.
    
    Args:
        model_path: Path to the Python file containing the model
        num_warmup: Number of warmup runs
        num_runs: Number of actual timing runs
        quiet: If True, suppress most output
    """
    if not quiet:
        print(f"\nProfiling model: {model_path}")
    
    # Load the model
    module = load_module_from_path(model_path)
    
    # Get initialization parameters if they exist
    init_params = []
    if hasattr(module, 'get_init_inputs'):
        init_params = module.get_init_inputs()
        if not quiet:
            print(f"Using initialization parameters: {init_params}")
    
    # Initialize model with parameters if provided
    model = module.Model(*init_params).cuda()
    
    # Get input tensors
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in module.get_inputs()]
    
    # Print input shapes for debugging
    if not quiet:
        input_shapes = [x.shape if isinstance(x, torch.Tensor) else type(x) for x in inputs]
        print(f"Input shapes: {input_shapes}")
    
    # Warmup runs
    if not quiet:
        print("Warming up...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(*inputs)
    
    # Actual timing runs
    if not quiet:
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
    
    if not quiet:
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

def compare_implementations(level, task):
    """Compare baseline and agent implementations."""
    print(f"\nComparing implementations for level {level}, task {task}")
    print("-" * 50)
    
    # Profile baseline
    baseline_path = find_task_file(level, task, use_agent_solution=False)
    print("Baseline implementation:")
    baseline_results = profile_model(baseline_path, quiet=False)
    
    print("\n" + "-" * 50)
    
    # Profile agent solution
    agent_path = find_task_file(level, task, use_agent_solution=True)
    print("Agent implementation:")
    agent_results = profile_model(agent_path, quiet=False)
    
    # Calculate speedup
    speedup = baseline_results['average_ms'] / agent_results['average_ms']
    
    print("\n" + "=" * 50)
    print("Comparison Summary:")
    print(f"Baseline average: {baseline_results['average_ms']:.3f} ms")
    print(f"Agent average:    {agent_results['average_ms']:.3f} ms")
    print(f"Speedup:         {speedup:.2f}x")
    
    if speedup > 1:
        print(f"The agent solution is {speedup:.2f}x faster!")
    else:
        print(f"The baseline is {1/speedup:.2f}x faster.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_profiler.py <level>-<task> [new|compare]")
        print("Example: python simple_profiler.py 2-28")
        print("         python simple_profiler.py 2-28 new")
        print("         python simple_profiler.py 2-28 compare")
        sys.exit(1)
    
    # Parse level and task from format like "2-28"
    try:
        level, task = sys.argv[1].split('-')
        mode = sys.argv[2].lower() if len(sys.argv) > 2 else 'baseline'
        
        if mode == 'compare':
            compare_implementations(level, task)
        else:
            use_agent_solution = mode == 'new'
            model_path = find_task_file(level, task, use_agent_solution)
            profile_model(model_path)
            
    except ValueError:
        print("Error: Please use the format <level>-<task>, e.g., '2-28'")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1) 