# KernelBenchFiltered

A repository for benchmarking and comparing PyTorch kernel implementations. This project includes baseline implementations and optimized agent solutions, with tools to measure and compare their performance.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/pavanyellow/KernelBenchFiltered.git
cd KernelBenchFiltered

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `tasks/`: Baseline kernel implementations organized by difficulty level
- `best_agent_solutions/`: Optimized solutions from ML agents
- `simple_profiler.py`: Tool for profiling and comparing implementations

## Usage Examples

### Profile Baseline Implementation
```bash
# Format: python simple_profiler.py <level>-<task>
python simple_profiler.py 2-28

# Output example:
# Profiling model: tasks/level_2/28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py
# Average time: 0.124 ms
# Min time: 0.118 ms
# Max time: 0.258 ms
```

### Profile Optimized Solution
```bash
# Format: python simple_profiler.py <level>-<task> new
python simple_profiler.py 2-28 new

# Output example:
# Profiling model: best_agent_solutions/level_2/task_28.py
# Average time: 0.037 ms
# Min time: 0.035 ms
# Max time: 0.099 ms
```

### Compare Implementations
```bash
# Format: python simple_profiler.py <level>-<task> compare
python simple_profiler.py 2-28 compare

# Output example:
# Baseline average: 0.124 ms
# Agent average: 0.037 ms
# Speedup: 3.36x
# The agent solution is 3.36x faster!
```

## Available Tasks

Tasks are organized by difficulty levels (1-5) and include various operations:
- Matrix multiplications
- Convolutions
- Activation functions
- Normalization layers
- Pooling operations
- And more...

Each task follows a standard interface:
```python
class Model(nn.Module):
    def __init__(self, *args):
        # Initialize model with optional parameters
        pass
        
    def forward(self, *inputs):
        # Perform the computation
        pass

def get_inputs():
    # Return list of input tensors for benchmarking
    pass

def get_init_inputs():
    # Return list of initialization parameters (if needed)
    pass
```

## Requirements

- Python 3.x
- PyTorch with CUDA support
- Triton
- NumPy
- Ninja build system

## Development

When adding new implementations:
1. Follow the standard model interface
2. Ensure CUDA compatibility
3. Implement both `get_inputs()` and `get_init_inputs()`
4. Place files in appropriate level directories
5. Use descriptive names for operations

## Notes

- All timing measurements include proper CUDA synchronization
- Default settings include warmup runs for stable measurements
- Results include min, max, and average execution times
- TF32 and medium precision matmul settings are used for consistency 