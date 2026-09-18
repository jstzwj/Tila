# buggy_triton_add.py
#
# 这是一个故意包含错误的 Triton 程序。
# 目标：
#     实现逐元素向量加法
#
#         C = A + B
#
# 注意：
#     下面的 Triton kernel 故意写错了。
#     请不要把它当作正确实现。
#
# 运行环境：
#     pip install torch triton
#
# 运行：
#     python buggy_triton_add.py

import torch
import triton
import triton.language as tl


# ============================================================
# Buggy Triton Kernel
# ============================================================

@triton.jit
def buggy_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 当前 program 的 ID
    pid = tl.program_id(axis=0)

    # 当前 program 要处理的元素
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # --------------------------------------------------------
    # BUG:
    #
    # 这里没有进行边界检查。
    #
    # 当 offsets >= n_elements 时，
    # 会访问 tensor 范围之外的内存。
    # --------------------------------------------------------

    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)

    result = a + b

    # --------------------------------------------------------
    # BUG:
    #
    # store 同样没有进行边界检查。
    # --------------------------------------------------------

    tl.store(c_ptr + offsets, result)


# ============================================================
# Python wrapper
# ============================================================

def buggy_add(a: torch.Tensor, b: torch.Tensor):

    assert a.is_cuda
    assert b.is_cuda

    assert a.shape == b.shape

    n_elements = a.numel()

    output = torch.empty_like(a)

    # --------------------------------------------------------
    # BUG:
    #
    # 假设：
    #
    # n_elements = 1000
    # BLOCK_SIZE = 256
    #
    # 那么：
    #
    # 1000 // 256 = 3
    #
    # 实际只会启动：
    #
    # pid = 0
    # pid = 1
    # pid = 2
    #
    # 只能处理：
    #
    # 0 ~ 767
    #
    # 后面的：
    #
    # 768 ~ 999
    #
    # 完全没有被计算。
    # --------------------------------------------------------

    grid = lambda meta: (
        n_elements // meta["BLOCK_SIZE"],
    )

    buggy_add_kernel[grid](
        a,
        b,
        output,
        n_elements,
        BLOCK_SIZE=256,
    )

    return output


# ============================================================
# Test
# ============================================================

def main():

    torch.manual_seed(42)

    device = "cuda"

    # 故意选择一个不能被 256 整除的长度
    N = 1000

    print("=" * 60)
    print("Buggy Triton Vector Add")
    print("=" * 60)

    print(f"N          = {N}")
    print(f"BLOCK_SIZE = 256")
    print()

    # 创建输入
    a = torch.randn(
        N,
        device=device,
        dtype=torch.float32,
    )

    b = torch.randn(
        N,
        device=device,
        dtype=torch.float32,
    )

    # PyTorch 正确结果
    expected = a + b

    # Triton 错误实现
    actual = buggy_add(a, b)

    # GPU 是异步执行的
    torch.cuda.synchronize()

    print("First 10 elements:")
    print()

    print("PyTorch:")
    print(expected[:10])

    print()

    print("Triton:")
    print(actual[:10])

    print()

    # --------------------------------------------------------
    # 检查整体结果
    # --------------------------------------------------------

    correct = torch.allclose(
        actual,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )

    print("=" * 60)

    if correct:
        print("Result: PASS")
    else:
        print("Result: FAIL")

    print("=" * 60)

    # --------------------------------------------------------
    # 找出错误的位置
    # --------------------------------------------------------

    diff = torch.abs(actual - expected)

    wrong_indices = torch.nonzero(
        diff > 1e-5
    ).flatten()

    print()
    print(f"Number of wrong elements: {wrong_indices.numel()}")

    if wrong_indices.numel() > 0:

        print()

        print("First wrong indices:")

        print(
            wrong_indices[:20]
        )

        first_wrong = wrong_indices[0].item()

        print()
        print(f"First wrong index = {first_wrong}")

        print()

        start = max(0, first_wrong - 5)
        end = min(N, first_wrong + 10)

        print(f"Expected [{start}:{end}]:")

        print(
            expected[start:end]
        )

        print()

        print(f"Actual [{start}:{end}]:")

        print(
            actual[start:end]
        )


if __name__ == "__main__":
    main()