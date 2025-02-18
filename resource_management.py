import resource
import signal

def set_memory_limit(max_memory_mb):
    max_memory_bytes = max_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))

def set_time_limit(max_time_seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Time limit exceeded")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(max_time_seconds))

def remove_time_limit():
    signal.alarm(0) 