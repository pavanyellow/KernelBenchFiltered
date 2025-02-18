from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class Perf:
    time: Optional[float] = None
    error: Optional[str] = None
    message: Optional[str] = None
    gpu_index: Optional[int] = None
    lock_gpu: bool = False
    measurement_start_time: Optional[float] = None
    measurement_duration: Optional[float] = None
    lock_duration: Optional[float] = None
    time_waiting_for_lock: Optional[float] = None
    input_shape_types: Optional[str] = None
    output_shape_types: Optional[str] = None
    init_shape_types: Optional[str] = None
    max_float_deviation: Optional[float] = None
    correctness_trial_time: Optional[float] = None
    first_different_index: Optional[int] = None
    magnitude_ratio: Optional[float] = None
    frac_zero: Optional[float] = None
    frac_zero_excluding_last: Optional[float] = None 