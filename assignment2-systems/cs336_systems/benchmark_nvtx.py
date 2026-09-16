from basics.model import BasicsTransformerLM
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from collections import defaultdict
import argparse
import torch
import timeit
import pandas as pd
import statistics as st
import torch.cuda.nvtx as nvtx

parser = argparse.ArgumentParser()
# parser.add_argument('--num_layers', default=12, type=int, help='the number of layers in transformer model')
parser.add_argument('model_size', default='small', type=str, help='the overall size of the model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BSZ=2
VOCAB_SIZE=10000
CTX_LEN=1024

w=5     # warm up runs
nrun=10

def sync_excute(func):
    output = func()
    if device == 'cuda':
        torch.cuda.synchronize()
    return output

benchmark_stats = defaultdict(lambda: defaultdict(list))

def benchmarking(record_dict, **model_hps):
    model = BasicsTransformerLM(**model_hps, vocab_size=VOCAB_SIZE, context_length=CTX_LEN).to(device)

    batch_data = torch.randint(VOCAB_SIZE, size=(BSZ, CTX_LEN)).to(device)
    labels = torch.randint(VOCAB_SIZE, size=(BSZ, CTX_LEN)).to(device)
    optimizer = AdamW(model.parameters())

    for _ in range(w):
        output = model(batch_data)
    if device == 'cuda':
        torch.cuda.synchronize()

    for _ in range(nrun):
        optimizer.zero_grad()
        forward_start = timeit.default_timer()
        with nvtx.range("forward pass"):
            output = sync_excute(lambda: model(batch_data))
        forward_end = timeit.default_timer()

        output = output.view(-1, VOCAB_SIZE)
        labels = labels.view(-1)
        loss = cross_entropy(output, labels)
        backward_start = timeit.default_timer()
        with nvtx.range("backward pass"):
            sync_excute(loss.backward)
        backward_end = timeit.default_timer()
        opt_start = timeit.default_timer()
        with nvtx.range("optimizer step"):
            sync_excute(optimizer.step)
        opt_end = timeit.default_timer()

        record_dict['forward'].append(forward_end - forward_start)
        record_dict['backward'].append(backward_end - backward_start)
        record_dict['optimizer'].append(opt_end - opt_start)

    
def get_model_hps(model_type):
    match model_type:
        case 'small':
            model_hps = dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12)
        case 'medium':
            model_hps = dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16)
        case 'large':
            model_hps = dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20)
        case 'xl':
            model_hps = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)
        case '10B':
            model_hps = dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36)
        case _:
            raise ValueError(f'Undefined size label {args.model_size}')
    return model_hps

if __name__ == "__main__":
    if args.model_size == 'all':
        for model_size in ('small', 'medium', 'large', 'xl'):
            model_hps = get_model_hps(model_size)
            benchmarking(benchmark_stats[model_size], **model_hps)
    else:
        model_hps = get_model_hps(args.model_size)
        benchmarking(benchmark_stats[args.model_size], **model_hps)

    summary = {
        size: {
            f"{stage}_{fn.__name__}": fn(vals)
            for stage, vals in stages.items()
            for fn in (min, st.mean, st.stdev)
        }
        for size, stages in benchmark_stats.items()
    }
    df = pd.DataFrame(summary).T
    df.to_markdown('benchmark_results.md')