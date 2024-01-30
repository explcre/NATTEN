# import unittest
# import torch
# from neighborhood_cross_attention_2d import NeighborhoodCrossAttention2D

# class NeighborhoodCrossAttention2DTests(unittest.TestCase):
#     def test_forward_pass(self):
#         """
#         Test the forward pass of the NeighborhoodCrossAttention2D module.
#         """
#         # Define the module parameters
#         dim = 64
#         num_heads = 8
#         kernel_size = 3
#         dilation = 1

#         # Create an instance of the module
#         nca = NeighborhoodCrossAttention2D(dim, num_heads, kernel_size, dilation)

#         # Create dummy inputs for query and key-value pair
#         B, H, W, C = 2, 16, 16, 64  # Batch size, Height, Width, Channels
#         x_q = torch.randn(B, H, W, C)
#         x_kv = torch.randn(B, H, W, C)

#         # Forward pass
#         output = nca(x_q, x_kv)

#         # Check output shape
#         self.assertEqual(output.shape, (B, H, W, C))

#     def test_backward_pass(self):
#         """
#         Test the backward pass (gradient computation) of the NeighborhoodCrossAttention2D module.
#         """
#         # Define the module parameters
#         dim = 64
#         num_heads = 8
#         kernel_size = 3
#         dilation = 1

#         # Create an instance of the module
#         nca = NeighborhoodCrossAttention2D(dim, num_heads, kernel_size, dilation)

#         # Create dummy inputs for query and key-value pair
#         B, H, W, C = 2, 16, 16, 64
#         x_q = torch.randn(B, H, W, C, requires_grad=True)
#         x_kv = torch.randn(B, H, W, C, requires_grad=True)

#         # Forward pass
#         output = nca(x_q, x_kv)

#         # Backward pass
#         output.sum().backward()

#         # Check gradients
#         self.assertIsNotNone(x_q.grad)
#         self.assertIsNotNone(x_kv.grad)

# # Additional tests can be added here for different configurations, input shapes, etc.

# if __name__ == "__main__":
#     unittest.main()
import unittest
import torch
from natten import (
    enable_gemm_na, enable_tf32, enable_tiled_na, disable_gemm_na,
    disable_tf32, disable_tiled_na, has_bfloat, has_fp32_gemm,
    has_gemm, has_half
)
from natten.functional import natten2dav,natten2dqkrpb#na2d_av, na2d_qk_with_bias
from torch.autograd import gradcheck

# Assuming these utility functions are available from self-attention tests
from natten.utils.testing import (
    skip_if_cuda_is_not_supported, skip_if_gemm_does_not_support_double_precision,
    skip_if_nested_is_not_supported
)

# Global variables for hardware support
HAS_GEMM = has_gemm()
HAS_FLOAT_GEMM = has_fp32_gemm()
HAS_HALF = has_half()
HAS_BFLOAT = has_bfloat()

# import unittest
# import torch
# from torch.autograd import gradcheck
from torch.utils.cpp_extension import CUDA_HOME

# Import your custom modules here. Adjust these imports based on your actual module structure.
# Get the current working directory
import os
import sys
current_dir = os.getcwd()

# Define the path to the parent directory (higher-level directory)
parent_dir = os.path.abspath(os.path.join(current_dir, '../src/natten/'))

# Add the parent directory to the sys.path list
sys.path.append(parent_dir)

# Now, you can import the 'functional' module
#from natten.functional import natten2dav,natten2dqkrpb#,natten2dcrossqkrpb
#from functional import natten2dav,natten2dqkrpb,natten2dcrossqkrpb

#from functional import natten2dcrossqkrpb, natten2dav,natten2dqkrpb
# If you have any additional utility functions or constants to import, add them here.

def init_cpu_ref_cross(B, H, X1, Y1, X2, Y2, D, kernel_size, dilation, has_bias):
    """
    Initialize inputs for cross-attention testing on CPU.
    B: Batch size, H: Number of heads, X1, Y1: Dimensions of input 1,
    X2, Y2: Dimensions of input 2, D: Dimension of each head,
    kernel_size: Size of the kernel, dilation: Dilation factor, has_bias: If bias is used.
    """
    with torch.no_grad():
        # Randomly initialize queries, keys, and values
        q = torch.randn((B, H, X1, Y1, D)) * (D ** -0.5)
        k = torch.randn((B, H, X2, Y2, D))
        v = torch.randn((B, H, X2, Y2, D))

        # Initialize relative positional bias
        rpb = None if not has_bias else torch.randn(H, 2 * kernel_size - 1, 2 * kernel_size - 1)

        # Clone and transfer to CUDA for comparison
        q_, k_, v_ = q.clone().cuda(), k.clone().cuda(), v.clone().cuda()
        rpb_ = None if rpb is None else rpb.clone().cuda()

        # Compute reference attention and output using CPU functions
        attn_ref = natten2dqkrpb(q, k, rpb, kernel_size, dilation)
        attn_ref = attn_ref.softmax(dim=-1)
        out_ref = natten2dav(attn_ref, v, kernel_size, dilation)

        return (q_, k_, v_, rpb_, kernel_size, dilation), (attn_ref.cuda(), out_ref.cuda())



class NA2DCrossTests(unittest.TestCase):
    def _test_against_cpu(self, inputs, reference, eps, dtype):
        q, k, v, rpb, kernel_size, dilation = inputs
        attn_ref, out_ref = reference

        with torch.no_grad():
            q_, k_, v_ = q.clone().to(dtype), k.clone().to(dtype), v.clone().to(dtype)
            rpb_ = rpb if rpb is None else rpb.clone().to(dtype)

            assert q_.is_cuda and k_.is_cuda and v_.is_cuda
            attn = natten2dqkrpb(q_, k_, rpb_, kernel_size, dilation)
            attn = attn.softmax(dim=-1)
            out = natten2dav(attn, v_, kernel_size, dilation)

            torch.testing.assert_close(attn.float(), attn_ref, atol=eps, rtol=0)
            torch.testing.assert_close(out.float(), out_ref, atol=eps, rtol=0)

    def _test_all_dtypes_against_cpu(self, B, H, X1, Y1, X2, Y2, D, kernel_size, dilation, has_bias=False):
        inputs, reference = init_cpu_ref_cross(
            B, H, X1, Y1, X2, Y2, D, kernel_size, dilation, has_bias
        )

        # Test naive kernels
        disable_tiled_na()
        disable_gemm_na()
        disable_tf32()
        self._test_against_cpu(inputs, reference, dtype=torch.float32, eps=1e-4)

        if HAS_HALF:
            self._test_against_cpu(inputs, reference, dtype=torch.float16, eps=1e-1)

        if HAS_BFLOAT:
            self._test_against_cpu(inputs, reference, dtype=torch.bfloat16, eps=1e-1)

        # Test tiled kernels
        if kernel_size < 15 and D == 32:
            enable_tiled_na()
            self._test_against_cpu(inputs, reference, dtype=torch.float32, eps=1e-4)
            
            if HAS_HALF:
                self._test_against_cpu(inputs, reference, dtype=torch.float16, eps=1e-1)
            
            if HAS_BFLOAT:
                self._test_against_cpu(inputs, reference, dtype=torch.bfloat16, eps=1e-1)

        # Test GEMM-based kernels
        if HAS_GEMM:
            enable_gemm_na()
            if HAS_FLOAT_GEMM:
                self._test_against_cpu(inputs, reference, dtype=torch.float32, eps=1e-2)
            
            enable_tf32()
            self._test_against_cpu(inputs, reference, dtype=torch.float32, eps=1e-2)

            assert (HAS_HALF), "GEMM kernels must support FP16 across all supported architectures."
            self._test_against_cpu(inputs, reference, dtype=torch.float16, eps=1e-1)

            if HAS_BFLOAT:
                self._test_against_cpu(inputs, reference, dtype=torch.bfloat16, eps=1e-1)

    @skip_if_cuda_is_not_supported()
    def test_cpu_vs_cuda(self):
        # Define test scenarios for different configurations
        # These scenarios can be adjusted based on your specific testing requirements
        self._test_all_dtypes_against_cpu(B=1, H=1, X1=16, Y1=16, X2=16, Y2=16, D=32, kernel_size=7, dilation=1)
        # Add more scenarios as needed

    # Additional tests for autograd, forward mode autograd, extra tokens, etc., can be added
    # following the same pattern as in the NA2DTests class
    @unittest.expectedFailure
    def test_invalid_kernel_size(self):
        self._test_autograd(
            B=1, H=1, X=8, Y=9, D=8, kernel_size=8, dilation=1, eps=1e-6, device="cuda"
        )

if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()



# # Global variables and configuration
# HAS_CUDA = torch.cuda.is_available() and (CUDA_HOME is not None)
# # Add any other global settings or configurations that are relevant to your tests.
# class NCrossA2DTests(unittest.TestCase):
#     def _test_cross_attention_cpu_vs_cuda(self, B, H, X, Y, D, kernel_size, dilation, eps, device):
#         # Initialize random tensors for query and key-value
#         torch.manual_seed(42)
#         q = torch.randn((B, H, X, Y, D), device=device, requires_grad=True)
#         kv = torch.randn((B, H, X, Y, D), device=device, requires_grad=True)
#         rpb = torch.randn((H, 2 * kernel_size - 1, 2 * kernel_size - 1), device=device, requires_grad=True)

#         # Compute cross-attention on CPU
#         attn_cpu = natten2dcrossqkrpb(q.cpu(), kv.cpu(), rpb.cpu(), kernel_size, dilation).softmax(dim=-1)
#         out_cpu = natten2dav(attn_cpu, kv.cpu(), kernel_size, dilation)

#         # Compute cross-attention on CUDA
#         attn_cuda = natten2dcrossqkrpb(q.cuda(), kv.cuda(), rpb.cuda(), kernel_size, dilation).softmax(dim=-1)
#         out_cuda = natten2dav(attn_cuda, kv.cuda(), kernel_size, dilation)

#         # Check consistency between CPU and CUDA outputs
#         torch.testing.assert_close(attn_cpu, attn_cuda.cpu(), atol=eps, rtol=0)
#         torch.testing.assert_close(out_cpu, out_cuda.cpu(), atol=eps, rtol=0)

#     def test_cpu_vs_cuda(self):
#         if not HAS_CUDA:
#             self.skipTest("CUDA not available.")
#         self._test_cross_attention_cpu_vs_cuda(
#             B=2, H=2, X=16, Y=16, D=32, kernel_size=7, dilation=1, eps=1e-4, device='cpu'
#         )

#     def _test_cross_attention_autograd(self, B, H, X, Y, D, kernel_size, dilation, eps, device):
#         torch.manual_seed(42)
#         q = torch.randn((B, H, X, Y, D), device=device, dtype=torch.float64, requires_grad=True)
#         kv = torch.randn((B, H, X, Y, D), device=device, dtype=torch.float64, requires_grad=True)
#         rpb = torch.randn((H, 2 * kernel_size - 1, 2 * kernel_size - 1), device=device, dtype=torch.float64, requires_grad=True)

#         # Check autograd
#         variables = [q, kv, rpb, kernel_size, dilation]
#         assert gradcheck(
#             natten2dcrossqkrpb,
#             variables,
#             eps=eps,
#             atol=1e-4,
#             rtol=1e-3,
#             fast_mode=False,
#         ), "Autograd check failed for Cross-Attention."

#     def test_autograd_cpu(self):
#         self._test_cross_attention_autograd(
#             B=1, H=1, X=8, Y=7, D=8, kernel_size=5, dilation=1, eps=1e-6, device='cpu'
#         )

#     # Additional tests like test_invalid_kernel_size, test_invalid_dilation, etc., similar to NA2DTests
#     def test_neighborhood_vs_cross_attention(self):
#         B, H, X, Y, D = 2, 2, 16, 16, 32
#         kernel_size, dilation = 7, 1
#         device = 'cpu' if not HAS_CUDA else 'cuda'

#         torch.manual_seed(42)
#         q = torch.randn((B, H, X, Y, D), device=device, requires_grad=True)

#         # Neighborhood Attention
#         attn_na = natten2dqkrpb(q, q, None, kernel_size, dilation).softmax(dim=-1)
#         out_na = natten2dav(attn_na, q, kernel_size, dilation)

#         # Cross Attention with same queries and keys/values
#         attn_ca = natten2dcrossqkrpb(q, q, None, kernel_size, dilation).softmax(dim=-1)
#         out_ca = natten2dav(attn_ca, q, kernel_size, dilation)

#         # Check if outputs are similar
#         torch.testing.assert_close(out_na, out_ca, atol=1e-4, rtol=0)

# if __name__ == "__main__":
#     unittest.main()
