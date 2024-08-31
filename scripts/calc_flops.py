import os

if os.getenv('DEBUG', '0') == '1':
    os.environ['PT_HPU_LAZY_MODE'] = '1'
    os.environ['LOG_LEVEL_PT_FALLBACK'] = '1'
    os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '1'
    os.environ['LOG_LEVEL_ALL'] = '3'
    os.environ['ENABLE_CONSOLE'] = 'true'

import time

import torch
import tqdm

try:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hpu
    device = 'hpu'
    dtype = torch.float16
except ImportError:
    device = 'cuda'
    dtype = torch.float16

def synchronize():
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'hpu':
        hpu.synchronize()
    else:
        raise Exception()

def mark_step():
    if device == 'hpu':
        htcore.mark_step()
    elif device == 'cuda':
        torch.cuda.synchronize()
    else:
        raise Exception()

def linear_working_set():
    # batch size 8, 128k Llama 3.1 8b Sequence
    x = torch.randn((1 * 131071, 4096), device=device, dtype=torch.float32).to(dtype)
    w1 = torch.randn((4096, 4096 * 4), device=device, dtype=torch.float32).to(dtype)
    w2 = torch.randn((4096 * 4, 4096), device=device, dtype=torch.float32).to(dtype)
    out1 = torch.zeros((x.shape[0], w1.shape[1]), device=device, dtype=torch.float32).to(dtype)
    out2 = torch.zeros((x.shape[0], w2.shape[1]), device=device, dtype=torch.float32).to(dtype)
    return x, w1, w2, out1, out2

def linear_worker(working_set):
    x, w1, w2, out1, out2 = working_set
    torch.mm(x, w1, out=out1)
    torch.mm(out1, w2, out=out2)

def linear_flops(working_set):
    x, w1, w2, out1, out2 = working_set
    return (
        (x.shape[0] * w1.shape[1] * (2 * w1.shape[0] - 1)) + \
        (x.shape[0] * w2.shape[1] * (2 * w2.shape[0] - 1))
    )

def conv_working_set():
    x = torch.randn((4, 512, 512, 512), dtype=torch.float32, device=device).to(dtype)
    w = torch.randn((512, 512, 3, 3,), dtype=torch.float32, device=device).to(dtype)
    out = torch.zeros(x.shape, dtype=torch.float32, device=device).to(dtype)
    return x, w,out

def conv_worker(working_set):
    x, w, out = working_set
    t = torch.nn.functional.conv2d(x, weight=w, bias=None, padding=(1, 1))
    out.copy_(t)

def conv_flops(working_set):
    x, w, out = working_set
    N, C, H, W = x.shape
    KOUT, KIN, KH, KW = w.shape
    return (N * H * W) * (2 * KIN * KH * KW - 1) * KOUT

def test(name, init_working_set_fn, worker_fn, flops_fn):
    n_steps = 50
    
    working_set = init_working_set_fn()
    synchronize()

    t = time.time()
    for i in tqdm.tqdm(range(n_steps), dynamic_ncols=True, leave=False, desc=name):
        worker_fn(working_set)
        mark_step()
    
    synchronize()
    elapsed = time.time() - t
    tflops = flops_fn(working_set) / (1024 * 1024 * 1024 * 1024) * n_steps
    print(f'[{name}, {dtype}, {device}] {tflops / elapsed:.2f} TFLOP/s, took {elapsed} sec')

def main():
    global dtype
    for test_dtype in [torch.float32, torch.bfloat16, torch.float16]:
        dtype = test_dtype
        
        test(
            'linear',
            linear_working_set,
            linear_worker,
            linear_flops,
        )

        test(
            'conv',
            conv_working_set,
            conv_worker,
            conv_flops,
        )

if __name__ == '__main__':
    main()
