import unittest
import torch
from neighborhood_cross_attention_2d import NeighborhoodCrossAttention2D

class NeighborhoodCrossAttention2DTests(unittest.TestCase):
    def test_forward_pass(self):
        """
        Test the forward pass of the NeighborhoodCrossAttention2D module.
        """
        # Define the module parameters
        dim = 64
        num_heads = 8
        kernel_size = 3
        dilation = 1

        # Create an instance of the module
        nca = NeighborhoodCrossAttention2D(dim, num_heads, kernel_size, dilation)

        # Create dummy inputs for query and key-value pair
        B, H, W, C = 2, 16, 16, 64  # Batch size, Height, Width, Channels
        x_q = torch.randn(B, H, W, C)
        x_kv = torch.randn(B, H, W, C)

        # Forward pass
        output = nca(x_q, x_kv)

        # Check output shape
        self.assertEqual(output.shape, (B, H, W, C))

    def test_backward_pass(self):
        """
        Test the backward pass (gradient computation) of the NeighborhoodCrossAttention2D module.
        """
        # Define the module parameters
        dim = 64
        num_heads = 8
        kernel_size = 3
        dilation = 1

        # Create an instance of the module
        nca = NeighborhoodCrossAttention2D(dim, num_heads, kernel_size, dilation)

        # Create dummy inputs for query and key-value pair
        B, H, W, C = 2, 16, 16, 64
        x_q = torch.randn(B, H, W, C, requires_grad=True)
        x_kv = torch.randn(B, H, W, C, requires_grad=True)

        # Forward pass
        output = nca(x_q, x_kv)

        # Backward pass
        output.sum().backward()

        # Check gradients
        self.assertIsNotNone(x_q.grad)
        self.assertIsNotNone(x_kv.grad)

    def test_cross_attention_equivalence(self):
        """
        Test that NeighborhoodCrossAttention2D with (a, a) has the same output as NeighborhoodAttention2D with (a).
        """
        # Define the module parameters
        dim = 64
        num_heads = 8
        kernel_size = 3
        dilation = 1

        # Create instances of the modules
        na = NeighborhoodAttention2D(dim, num_heads, kernel_size, dilation)
        nca = NeighborhoodCrossAttention2D(dim, num_heads, kernel_size, dilation)

        # Create a dummy input
        B, H, W, C = 2, 16, 16, 64
        x = torch.randn(B, H, W, C)

        # Forward pass for both modules
        output_na = na(x)
        output_nca = nca(x, x)

        # Check if outputs are close enough
        torch.testing.assert_close(output_na, output_nca, rtol=1e-5, atol=1e-5)

# Additional tests can be added here for different configurations, input shapes, etc.

if __name__ == "__main__":
    unittest.main()
