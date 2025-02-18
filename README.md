# KernelBenchFiltered

A repository containing kernel benchmarking tasks and evaluation tools for PyTorch-based solutions. This project focuses on measuring and comparing the performance of different kernel implementations.

## Project Structure

- `tasks/`: Contains the kernel benchmarking tasks and related assets
- `best_agent_solutions/`: Collection of top-performing solutions
- `randomly_sampled_problems/`: Set of randomly sampled benchmark problems
- `evaluate_solution.py`: Main script for evaluating solution performance
- `one_each_level.txt`: Contains problem specifications, one from each difficulty level

## Requirements

- Python 3.x
- PyTorch
- CUDA support
- Triton
- NumPy

## Evaluation

The project includes tools for:
- Measuring kernel performance
- Comparing solutions against ground truth
- Benchmarking with various input shapes and types
- Testing correctness and performance across multiple trials

## Usage

To evaluate a solution:
```python
python evaluate_solution.py [solution_path] [level] [index]
```

The evaluation includes:
- Correctness verification
- Performance benchmarking
- Comparison with baseline implementations

## Configuration

- Default settings include:
  - Multiple correctness trials
  - Warmup periods for accurate timing
  - Configurable tolerance levels for numerical comparisons
  - TF32 and medium precision matmul settings for consistent benchmarking 