import triton
import triton.language as tl
import torch

@triton.jit
def weighted_sum_kernel(
    x_ptr, weight_ptr, output_ptr,
    x_row_stride, x_dim_stride,
    w_dim_stride,
    output_dim_stride,
    NUM_ROWS, D,
    ROW_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),                            # shape of the entire block
        strides=(x_row_stride, x_dim_stride),           # stride of each dimension
        offsets=(ROW_TILE_SIZE * pid, D_TILE_SIZE),     # starting point of the first tile
        block_shape=(ROW_TILE_SIZE, D_TILE_SIZE),       # the block for each load/write
        order=(1, 0)
    )
    w_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(w_dim_stride,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE),
        order=(0,)
    )
    o_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        stride=(output_dim_stride,),
        offsets=(pid*ROW_TILE_SIZE,),
        block_shape=(ROW_TILE_SIZE,),
        order=(0,)
    )

    output = tl.zeros((NUM_ROWS,), dtype=tl.float32)
    for i in range(triton.cdiv(D, D_TILE_SIZE)):
        x = tl.load(x_block_ptr, boundry_check=(0, 1), padding_option='zero')
        w = tl.load(w_block_ptr, boundry_check=(0,), padding_option='zero')
        output += tl.sum(x * w, axis=1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        w_block_ptr = w_block_ptr.advance((D_TILE_SIZE,))

    tl.store(o_block_ptr, output, boundry_check=(0,))

def weighted_sum_triton(x, w):
    n_rows, d = x.shape
    ROW_TILE_SIZE=16
    D_TILE_SIZE=64

    output = torch.zeros(n_rows).to(x.device)
    weighted_sum_kernel[triton.cdiv(n_rows, ROW_TILE_SIZE)](
        x, w, output,
        x.stride(0), x.stride(1),
        w.stride(0),
        output.stride(0),
        n_rows, d, ROW_TILE_SIZE, D_TILE_SIZE
    )

if __name__ == '__main__':
    x = torch.randn(32, 1024).cuda()
    w = torch.randn(1024).cuda()

    triton_run_time = triton.testing.do_bench(lambda: weighted_sum_triton(x,w))
    torch_run_time = triton.testing.do_bench(lambda: x*w)

    print(f"triton_run_time: {triton_run_time}\ntorch_run_time: {torch_run_time}")