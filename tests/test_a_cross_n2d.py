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
from torch.autograd import gradcheck
from torch.utils.cpp_extension import CUDA_HOME

# Import your custom modules here. Adjust these imports based on your actual module structure.
# Get the current working directory
import os
import sys
current_dir = os.getcwd()

# Define the path to the parent directory (higher-level directory)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to the sys.path list
sys.path.append(parent_dir)

# Now, you can import the 'functional' module
from src.natten.functional import natten2dcrossqkrpb, natten2dav,natten2dqkrpb
#from functional import natten2dcrossqkrpb, natten2dav,natten2dqkrpb
# If you have any additional utility functions or constants to import, add them here.

# Global variables and configuration
HAS_CUDA = torch.cuda.is_available() and (CUDA_HOME is not None)
# Add any other global settings or configurations that are relevant to your tests.
class NCrossA2DTests(unittest.TestCase):
    def _test_cross_attention_cpu_vs_cuda(self, B, H, X, Y, D, kernel_size, dilation, eps, device):
        # Initialize random tensors for query and key-value
        torch.manual_seed(42)
        q = torch.randn((B, H, X, Y, D), device=device, requires_grad=True)
        kv = torch.randn((B, H, X, Y, D), device=device, requires_grad=True)
        rpb = torch.randn((H, 2 * kernel_size - 1, 2 * kernel_size - 1), device=device, requires_grad=True)

        # Compute cross-attention on CPU
        attn_cpu = natten2dcrossqkrpb(q.cpu(), kv.cpu(), rpb.cpu(), kernel_size, dilation).softmax(dim=-1)
        out_cpu = natten2dav(attn_cpu, kv.cpu(), kernel_size, dilation)

        # Compute cross-attention on CUDA
        attn_cuda = natten2dcrossqkrpb(q.cuda(), kv.cuda(), rpb.cuda(), kernel_size, dilation).softmax(dim=-1)
        out_cuda = natten2dav(attn_cuda, kv.cuda(), kernel_size, dilation)

        # Check consistency between CPU and CUDA outputs
        torch.testing.assert_close(attn_cpu, attn_cuda.cpu(), atol=eps, rtol=0)
        torch.testing.assert_close(out_cpu, out_cuda.cpu(), atol=eps, rtol=0)

    def test_cpu_vs_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        self._test_cross_attention_cpu_vs_cuda(
            B=2, H=2, X=16, Y=16, D=32, kernel_size=7, dilation=1, eps=1e-4, device='cpu'
        )

    def _test_cross_attention_autograd(self, B, H, X, Y, D, kernel_size, dilation, eps, device):
        torch.manual_seed(42)
        q = torch.randn((B, H, X, Y, D), device=device, dtype=torch.float64, requires_grad=True)
        kv = torch.randn((B, H, X, Y, D), device=device, dtype=torch.float64, requires_grad=True)
        rpb = torch.randn((H, 2 * kernel_size - 1, 2 * kernel_size - 1), device=device, dtype=torch.float64, requires_grad=True)

        # Check autograd
        variables = [q, kv, rpb, kernel_size, dilation]
        assert gradcheck(
            natten2dcrossqkrpb,
            variables,
            eps=eps,
            atol=1e-4,
            rtol=1e-3,
            fast_mode=False,
        ), "Autograd check failed for Cross-Attention."

    def test_autograd_cpu(self):
        self._test_cross_attention_autograd(
            B=1, H=1, X=8, Y=7, D=8, kernel_size=5, dilation=1, eps=1e-6, device='cpu'
        )

    # Additional tests like test_invalid_kernel_size, test_invalid_dilation, etc., similar to NA2DTests
    def test_neighborhood_vs_cross_attention(self):
        B, H, X, Y, D = 2, 2, 16, 16, 32
        kernel_size, dilation = 7, 1
        device = 'cpu' if not HAS_CUDA else 'cuda'

        torch.manual_seed(42)
        q = torch.randn((B, H, X, Y, D), device=device, requires_grad=True)

        # Neighborhood Attention
        attn_na = natten2dqkrpb(q, q, None, kernel_size, dilation).softmax(dim=-1)
        out_na = natten2dav(attn_na, q, kernel_size, dilation)

        # Cross Attention with same queries and keys/values
        attn_ca = natten2dcrossqkrpb(q, q, None, kernel_size, dilation).softmax(dim=-1)
        out_ca = natten2dav(attn_ca, q, kernel_size, dilation)

        # Check if outputs are similar
        torch.testing.assert_close(out_na, out_ca, atol=1e-4, rtol=0)

if __name__ == "__main__":
    unittest.main()
