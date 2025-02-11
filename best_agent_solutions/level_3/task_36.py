# level 3 index 36 agent name: KernelAgent 4o speedup: 1.06x

import torch
import torch.nn as nn
import triton
import triton.language as tl

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        out, state = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return state[0]

@triton.jit
def lstm_elementwise_fused(
    gates_ptr, h_prev_ptr, c_prev_ptr, c_next_ptr, h_next_ptr,
    batch_size: tl.constexpr, hidden_size: tl.constexpr
):
    batch_id = tl.program_id(0)
    offset = tl.arange(0, hidden_size)

    # Load gates and previous states
    gates = tl.load(gates_ptr + batch_id * hidden_size * 4 + offset)
    
    h_prev = tl.load(h_prev_ptr + batch_id * hidden_size + offset)
    c_prev = tl.load(c_prev_ptr + batch_id * hidden_size + offset)

    # Element-wise operations
    i = tl.sigmoid(gates[offset + 0 * hidden_size])
    f = tl.sigmoid(gates[offset + 1 * hidden_size])
    g = tl.tanh(gates[offset + 2 * hidden_size])
    o = tl.sigmoid(gates[offset + 3 * hidden_size])

    c_next = f * c_prev + i * g
    h_next = o * tl.tanh(c_next)

    # Store results
    tl.store(c_next_ptr + batch_id * hidden_size + offset, c_next)
    tl.store(h_next_ptr + batch_id * hidden_size + offset, h_next)

def lstm_forward_optimized(x, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh):
    B, T, I = x.size()  # Batch, Time, Input
    H = h0.size(2)      # Hidden size elements

    # Initialize storage for h and c
    h_n = h0.clone()
    c_n = c0.clone()

    # Iterate over each time step
    for t in range(T):
        gates = x[:, t, :] @ weight_ih.t() + bias_ih + (h_n @ weight_hh.t() + bias_hh)
        
        # Allocate buffers for next states
        c_next = torch.empty_like(c_n)
        h_next = torch.empty_like(h_n)

        # Kernel Launch
        grid = (B,)
        lstm_elementwise_fused[grid](gates, h_n, c_n, c_next, h_next, batch_size=B, hidden_size=H)

        # Prepare h_n and c_n for next time step
        h_n = h_next
        c_n = c_next

    return h_n

# Updated the model to use the optimized LSTM
class OptimizedModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(OptimizedModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        weight_ih, weight_hh = self.lstm.weight_ih_l0, self.lstm.weight_hh_l0
        bias_ih, bias_hh = self.lstm.bias_ih_l0, self.lstm.bias_hh_l0
        
        h_n = lstm_forward_optimized(x, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh)
        out = self.fc(h_n[:, -1, :])
        return h_n
