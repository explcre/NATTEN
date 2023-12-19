import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .functional import natten2dav, natten2dqkrpb


def trunc_normal_(tensor, mean=0, std=1, a=-2, b=2):
    # This function fills the input Tensor with values drawn from a truncated
    # normal distribution. The values are effectively drawn from the normal
    # distribution `Normal(mean, std)` with values outside `[a, b]` redrawn
    # until they fall inside the interval.
    with torch.no_grad():
        size = tensor.size()
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < b) & (tmp > a)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class NeighborhoodCrossAttention2D(nn.Module):
    """
    Neighborhood Cross Attention 2D Module
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # Ensure kernel size and dilation are appropriate
        assert kernel_size > 1 and kernel_size % 2 == 1, f"Kernel size must be odd and greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert dilation is None or dilation >= 1, f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        # Linear layer for QKV
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Only two times dim because K and V are combined

        # Relative position bias
        if bias:
            self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter('rpb', None)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, Hq, Wq, C = x_q.shape
        _, Hkv, Wkv, _ = x_kv.shape

        # Process queries
        q = self.qkv(x_q).reshape(B, Hq, Wq, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        q = q * self.scale

        # Process keys and values
        kv = self.qkv(x_kv).reshape(B, Hkv, Wkv, 2, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        k, v = kv[0], kv[1]

        # Attention mechanism (assuming natten2dqkrpb and natten2dav functions are defined)
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.kernel_size, self.dilation)

        # Reshape and project output
        x = x.permute(0, 2, 3, 1, 4).reshape(B, Hq, Wq, C)
        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )
