import jax
import jax.numpy as jnp
import numpy as np

from hyper_task_descriptions.common.testing import get_prng_key
from hyper_task_descriptions.modeling.lora import (
    LoraDenseGeneral,
    LoraMultiHeadDotProductAttention,
    lora_linear,
)


def test_lora_linear():
    batch_size, in_features, out_features = 3, 4, 5
    rank = 2

    key = get_prng_key(23)
    inputs = jax.random.normal(key, (batch_size, in_features))
    W = jax.random.normal(key, (in_features, out_features))
    A = jax.random.normal(key, (in_features, rank))
    B = jax.random.normal(key, (rank, out_features))

    output = lora_linear(inputs, W, A, B, 1, rank)
    assert output.shape == (batch_size, out_features)
    expected_output = (inputs @ W) + (inputs @ A @ B) * (1 / rank)  # W0x + (BAx*scaling)
    assert jnp.all(output == expected_output)


def test_lora_dense_general():
    batch_size, in_features, out_features = 3, 4, 5
    rank = 2

    inputs = jnp.array(np.random.randn(batch_size, in_features))

    lora_dense = LoraDenseGeneral(out_features, rank=rank, hyper_gen=False)
    key = get_prng_key(23)
    params = lora_dense.init(key, inputs)
    assert "lora_a" in params["params"].keys()

    output = lora_dense.apply(params, inputs)
    assert output.shape == (batch_size, out_features)

    lora_dense = LoraDenseGeneral(out_features, rank=rank, hyper_gen=True)
    A = jax.random.normal(key, (in_features, rank))
    B = jax.random.normal(key, (rank, out_features))
    key = get_prng_key(23)
    params = lora_dense.init(key, inputs, lora_a=A, lora_b=B)
    assert "lora_a" not in params["params"].keys()


def test_lora_multihead_dot_product_attention():
    batch_size, q_len, q_features, kv_len, kv_features = 3, 4, 5, 6, 7
    num_heads, head_dim = 8, 16
    rank = 2

    inputs_q = jnp.array(np.random.randn(batch_size, q_len, q_features))
    inputs_kv = jnp.array(np.random.randn(batch_size, kv_len, kv_features))

    lora_multihead = LoraMultiHeadDotProductAttention(
        num_heads=num_heads, head_dim=head_dim, rank=rank
    )
    key = get_prng_key(23)
    params = lora_multihead.init(key, inputs_q, inputs_kv)

    output = lora_multihead.apply(params, inputs_q, inputs_kv)
    assert output.shape == (batch_size, q_len, q_features)


# if __name__ == "__main__":
#     test_lora_multihead_dot_product_attention()

# def test_replace_layer():
#     class FakeModel(nn.Module):
#         out_features: int
#
#         @nn.compact
#         def __call__(self, inputs):
#             out = nn.DenseGeneral(self.out_features)(inputs)
#             return out


# def test_sanity_check_lora_dense():
#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as F
#
#     import math
#
#     class LoRALayer():
#         def __init__(
#                 self,
#                 r: int,
#                 lora_alpha: int,
#                 lora_dropout: float,
#                 merge_weights: bool,
#         ):
#             self.r = r
#             self.lora_alpha = lora_alpha
#             # Optional dropout
#             if lora_dropout > 0.:
#                 self.lora_dropout = nn.Dropout(p=lora_dropout)
#             else:
#                 self.lora_dropout = lambda x: x
#             # Mark the weight as unmerged
#             self.merged = False
#             self.merge_weights = merge_weights
#
#     class Linear(nn.Linear, LoRALayer):
#         # LoRA implemented in a dense layer
#         def __init__(
#                 self,
#                 in_features: int,
#                 out_features: int,
#                 r: int = 0,
#                 lora_alpha: int = 1,
#                 lora_dropout: float = 0.,
#                 fan_in_fan_out: bool = False,
#                 # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
#                 merge_weights: bool = True,
#                 **kwargs
#         ):
#             nn.Linear.__init__(self, in_features, out_features, **kwargs)
#             LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                                merge_weights=merge_weights)
#
#             self.fan_in_fan_out = fan_in_fan_out
#             # Actual trainable parameters
#             if r > 0:
#                 self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
#                 self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
#                 self.scaling = self.lora_alpha / self.r
#                 # Freezing the pre-trained weight matrix
#                 self.weight.requires_grad = False
#             self.reset_parameters()
#             if fan_in_fan_out:
#                 self.weight.data = self.weight.data.T
#
#         def reset_parameters(self):
#             nn.Linear.reset_parameters(self)
#             if hasattr(self, 'lora_A'):
#                 # initialize A the same way as the default for nn.Linear and B to zero
#                 nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#                 nn.init.zeros_(self.lora_B)
#
#         def train(self, mode: bool = True):
#             def T(w):
#                 return w.T if self.fan_in_fan_out else w
#
#             nn.Linear.train(self, mode)
#             if self.merge_weights and self.merged:
#                 # Make sure that the weights are not merged
#                 if self.r > 0:
#                     self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
#                 self.merged = False
#
#         def eval(self):
#             def T(w):
#                 return w.T if self.fan_in_fan_out else w
#
#             nn.Linear.eval(self)
#             if self.merge_weights and not self.merged:
#                 # Merge the weights and mark it
#                 if self.r > 0:
#                     self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
#                 self.merged = True
#
#         def forward(self, x: torch.Tensor):
#             def T(w):
#                 return w.T if self.fan_in_fan_out else w
#
#             if self.r > 0 and not self.merged:
#                 result = F.linear(x, T(self.weight), bias=self.bias)
#                 if self.r > 0:
#                     result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
#                 return result
#             else:
#                 return F.linear(x, T(self.weight), bias=self.bias)
#
#     torch.manual_seed(23)
#     batch_size, in_features, out_features = 3, 4, 5
#     rank = 2
#     torch_lora = Linear(in_features, out_features, r=rank)
#
#     inputs = torch.randn(batch_size, in_features)
#     output = torch_lora.forward(inputs)
#     assert output.shape == (batch_size, out_features)
