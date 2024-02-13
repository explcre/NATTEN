
import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from natten.functional import na2d_av, na2d_qk_with_bias
import unittest
from typing import Optional

class NeighborhoodCrossAttention2D(nn.Module):
    """
    Neighborhood Cross Attention 2D Module
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # x1 is used for queries, x2 is used for keys and values
        B, H, W, C = x1.shape
        _, H_kv, W_kv, _ = x2.shape

        q = self.q_proj(x1).reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        kv = self.kv_proj(x2).reshape(B, H_kv, W_kv, 2, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        k, v = kv[0], kv[1]
        q = q * self.scale

        attn = na2d_qk_with_bias(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = na2d_av(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"has_bias={self.rpb is not None}"
        )
        

class TestNeighborhoodCrossAttention2D(unittest.TestCase):
    def setUp(self):
        self.dim = 64
        self.num_heads = 4
        self.kernel_size = 3
        self.dilation = 1
        self.batch_size = 2
        self.height = 16
        self.width = 16
        
        # Initialize the module
        self.nca_module = NeighborhoodCrossAttention2D(
            dim=self.dim,
            num_heads=self.num_heads,
            kernel_size=self.kernel_size,
            dilation=self.dilation
        )
    
    def test_forward_shape(self):
        """Test if the forward pass returns the correct shape"""
        x1 = torch.rand(self.batch_size, self.height, self.width, self.dim)
        x2 = torch.rand(self.batch_size, self.height, self.width, self.dim)
        
        output = self.nca_module(x1, x2)
        
        self.assertEqual(output.shape, (self.batch_size, self.height, self.width, self.dim))
    
    def test_gradient_flow(self):
        """Test if gradients can flow through the network"""
        x1 = torch.rand(self.batch_size, self.height, self.width, self.dim, requires_grad=True)
        x2 = torch.rand(self.batch_size, self.height, self.width, self.dim, requires_grad=True)
        
        output = self.nca_module(x1, x2)
        output.mean().backward()
        
        self.assertIsNotNone(x1.grad)
        self.assertIsNotNone(x2.grad)

if __name__ == '__main__':
    unittest.main()
    
            
# class NeighborhoodCrossAttentionPointsTests(unittest.TestCase):
#     def test_forward_pass(self):
#         """
#         Test the forward pass of the NeighborhoodCrossAttentionPoints module.
#         """
#         # Module parameters
#         dim = 64
#         num_heads = 8
#         kernel_size = 5
#         # Create an instance of the module
#         ncap = NeighborhoodCrossAttention2D(dim, num_heads, kernel_size)

#         # Create dummy inputs for query (list of points) and key-value pair
#         B, N, C = 2, 10, 64  # Batch size, Number of points, Channels
#         x1 = torch.randn(B, N, C)
#         x2 = torch.randn(B, N, C)

#         # Forward pass
#         output = ncap(x1, x2)

#         # Check output shape to match input queries shape
#         self.assertEqual(output.shape, (B, N, C))

# if __name__ == "__main__":
#     unittest.main()
