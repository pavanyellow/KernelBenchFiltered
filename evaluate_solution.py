import time
measurement_start_time = time.time()
import signal
import pathlib
import random
import torch
from torch import nn
import numpy as np
import importlib.util
import sys
import os
import traceback
import json
from triton.testing import do_bench
atol = 1e-2
rtol = 1e-2
torch.set_printoptions(profile='short')

# set matmul precision to medium because it's super low hanging fruit otherwise which adds noise
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.allow_tf32 = True


NUM_CORRECTNESS_TRIALS = 3
REP_TIME = 400
range_to_count = slice(5,10)
WARMUP_TIME = 50
SEED = 13
frac_cut = 0.2

ground_truth_dir = 'tasks/kernelbench/assets'

def set_seed(seed: int):
    # NOTE: this only sets on current cuda device
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_module_from_path(module_path, add_to_sys_modules: bool = False):
    module_path = pathlib.Path(module_path)
    name = module_path.stem
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    if add_to_sys_modules:
        sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod
    
resource_management = load_module_from_path('resource_management.py', add_to_sys_modules=True)
    
kb_util = load_module_from_path('kb_util.py', add_to_sys_modules=True)

tao_triton_util = load_module_from_path('tao_triton_util.py', add_to_sys_modules=True)
    
def get_original_path(level, index):
    level_path = f'{ground_truth_dir}/level_{level}'
    file_path = [x for x in os.listdir(level_path) if x.startswith(str(index)+'_')][0]
    return f'{level_path}/{file_path}'

def get_shape_type(arg):
    if isinstance(arg, torch.Tensor):
        return f"tensor(shape={tuple(arg.shape)}, dtype={arg.dtype})"
    stringified = str(arg)
    type_name = str(type(arg).__name__)
    if not stringified.startswith('<') and not stringified.endswith(')') and len(stringified) < 20:
        return type_name + '=' + stringified
    return type_name

def get_shape_types(args):
    if not isinstance(args, (list, tuple)):
        args = [args]
    types = [get_shape_type(arg) for arg in args]
    return f"({', '.join(types)})"

@torch.no_grad()
def score(level, index, solution_path, device, lock_gpu=False, compile=False)->kb_util.Perf:
    torch.cuda.set_device(device) # set device before loading modules so they can use 'cuda' and have that point to correct device
    with torch.device(device):
        original = load_module_from_path(get_original_path(level, index), add_to_sys_modules=True)
        inits = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in original.get_init_inputs()
        ]
        result = {
            "gpu_index": device,
            "lock_gpu": lock_gpu,
        }
        result['init_shape_types'] = get_shape_types(inits)
            
        set_seed(SEED)
        solution = load_module_from_path(solution_path, add_to_sys_modules=False)
        try:
            if hasattr(solution, "ModelNew"):
                solution_model: nn.Module = solution.ModelNew(*inits)
            else:
                solution_model: nn.Module = solution.Model(*inits)
            assert isinstance(solution_model, nn.Module)
            assert hasattr(solution_model, "forward")
        except Exception as e:
            result |= {
                "error": "Solution failed to load",
                "message": traceback.format_exc(),
            }
            return result
        # do once to compile immediately so we don't spend gpu memory on the original if the solution doesnt even compile
        # this is before the lock, slightly sketchy
        test_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in original.get_inputs()
            ]
        result['input_shape_types'] = get_shape_types(test_inputs)
        test_outputs = solution_model(*test_inputs)
        result['output_shape_types'] = get_shape_types(test_outputs)
        
        # load original after checking that the solution compiled
        set_seed(SEED)
        original_model: nn.Module = original.Model(*inits).to(device)
        assert hasattr(original_model, "forward")
            
            
        if compile:
            if 'cudnn_benchmark' in compile:
                torch.backends.cudnn.benchmark = True
            if 'compile' in compile:
                solution_model = torch.compile(solution_model)
            if 'downcast_model' in compile:
                solution_model = solution_model.half()
                
                class PrecisionWrapper(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        
                    def forward(self, *args):
                        half_args = [x.half() if isinstance(x, torch.Tensor) else x for x in args]
                        return self.model(*half_args).float()
                        
                solution_model = PrecisionWrapper(solution_model)
            if 'jit_script' in compile:
                solution_model = torch.jit.script(solution_model)
            if 'autocast_half' in compile:
                solution_model = torch.autocast(device_type='cuda', dtype=torch.half)(solution_model)
            if 'autocast_bf16' in compile:
                solution_model = torch.autocast(device_type='cuda', dtype=torch.bfloat16)(solution_model)
        
        pre_lock_time = time.time()
        with resource_management.GpuLockSingle(device, set_pytorch_device=False, disable=not lock_gpu):
            # check correctness
            lock_start_time = time.time()
            result['time_waiting_for_lock'] = lock_start_time - pre_lock_time
            time.sleep(0.5) # sleep to make sure the gpu is ready
            set_seed(SEED)
            correctness_trial_seeds = [
            random.randint(0, 2 ** 32)  for _ in range(NUM_CORRECTNESS_TRIALS)
            ]
            prev_original_output = None
            correctness_start_time = time.time()
            for seed in correctness_trial_seeds:
                set_seed(seed)
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in original.get_inputs()
                ]

                set_seed(seed)
                try:
                    solution_output = solution_model(*inputs)
                except Exception as e:
                    result |= {
                        "error": "Solution failed to execute",
                        "message": traceback.format_exc(),
                    }
                    return result
                
                # compute original AFTER solution to avoid rare cheating case of solution's torch.empty allocating original's memory
                set_seed(seed)
                original_output = original_model(*inputs)

                if solution_output.shape != original_output.shape:
                    result |= {
                        "error": "Output shapes do not match",
                        "original_shape": original_output.shape,
                        "solution_shape": solution_output.shape,
                    }
                else:
                    result['max_float_deviation'] = float(torch.max(torch.abs(solution_output - original_output)))
                    frac_near_zero_original = float(torch.sum(torch.abs(original_output) < atol).item() / original_output.numel())
                    original_output_float = original_output.float()
                    frac_near_mean_original = float(torch.sum(torch.abs(original_output_float - original_output_float.mean()) < atol).item() / original_output.numel())
                    if frac_near_zero_original > 0.7:
                        result |= {
                            "error": f"Original output has too many values within {atol} of zero",
                            "frac_near_zero_original": frac_near_zero_original
                        }
                    elif original_output.numel()>100 and frac_near_mean_original > 0.7:
                        result |= {
                            "error": f"Original output has too many values within {atol} of its mean",
                            "frac_near_zero_original": frac_near_mean_original
                        }
                    elif prev_original_output is not None:
                        frac_near_prev_seed = float(torch.sum(torch.abs(original_output - prev_original_output) < atol).item() / original_output.numel())
                        if frac_near_prev_seed > 0.95:
                            result |= {
                                "error": f"Original output has too many values within {atol} of its output with a different input seed",
                                "frac_near_zero_original": frac_near_prev_seed
                            }
                    if not torch.allclose(original_output, solution_output, atol=atol, rtol=rtol):
                        diffidxs = (torch.abs(solution_output - original_output) > atol).nonzero(as_tuple=True)
                        if len(diffidxs[0]):
                            result |= {"first_different_index":diffidxs[0][0].item()}
                        magnitude_ratio = float(torch.norm(original_output.flatten() - solution_output.flatten())/ torch.norm(solution_output.flatten()))
                        print("ASCII Comparison")
                        print(tao_triton_util.ascii_compare_tensors(original_output.cpu(), solution_output.cpu(), atol=atol, rtol=rtol, scaling='truncate'))
                        print("ASCII Comparison to Zero")
                        print(tao_triton_util.ascii_compare_tensors(torch.zeros_like(original_output).cpu(), solution_output.cpu(), atol=1e-7, rtol=1e-7, scaling='truncate', same_symbol='0', different_symbol='.'))
                        frac_zero = float(torch.sum(solution_output == 0).item() / solution_output.numel())
                        last_dim_summed = solution_output.sum(dim=-1)
                        frac_zero_excluding_last = float(torch.sum(last_dim_summed == 0).item() / last_dim_summed.numel())
                        result |= {
                            "error": "Output values do not match",
                            "magnitude_ratio": magnitude_ratio,
                            "frac_zero": frac_zero,
                            "frac_zero_excluding_last": frac_zero_excluding_last,
                        }
                prev_original_output = original_output

            result |={
                "correctness_trial_time":time.time()-correctness_start_time
            }
            set_seed(SEED)
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in original.get_inputs()
            ]

            bench_mean_time = do_bench(lambda: solution_model(*inputs), rep=REP_TIME, warmup=WARMUP_TIME, return_mode='median')
            lock_duration = time.time() - lock_start_time


        result |= {
            "time": bench_mean_time,
            "measurement_start_time": measurement_start_time,
            "measurement_duration": time.time() - measurement_start_time,
            "lock_duration": lock_duration,
        }
        return result


if __name__ == "__main__":
    # example usage: time python tasks/kernelbench/assets/evaluate_solution.py 1 1 tasks/kernelbench/assets/level_1/1_Square_matrix_multiplication_.py
    result = score(sys.argv[1], sys.argv[2], sys.argv[3],int( sys.argv[4]), lock_gpu=sys.argv[5].lower() == 'true', compile=sys.argv[6].lower() if len(sys.argv) > 6 else None)
    if 'message' in result:
        print(result['message'])
    print('49j3I' + json.dumps(result, indent=4))
