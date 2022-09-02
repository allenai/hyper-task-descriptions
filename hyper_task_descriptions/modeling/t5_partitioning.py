"""
Partition map for t5 v1.1
See https://github.com/google-research/t5x/blob/main/docs/usage/partitioning.md
for details on how this works.
"""

t5_axes_names_override = [
    (r"hyper/encoder/shared/embedding", ("vocab", "embed")),
    (
        r"hyper/encoder/encoder/block/\d+/layer/0/SelfAttention/(q|k|v)/kernel",
        ("embed", "joined_kv"),
    ),
    (r"hyper/encoder/encoder/block/\d+/layer/0/SelfAttention/o/kernel", ("joined_kv", "embed")),
    (
        r"hyper/encoder/encoder/block/\d+/layer/0/SelfAttention/relative_attention_bias/embedding",
        ("heads", "relpos_buckets"),
    ),
    (r"hyper/encoder/encoder/block/\d+/layer/0/layer_norm/weight", ("embed",)),
    (r"hyper/encoder/encoder/block/\d+/layer/1/DenseReluDense/wi_0/kernel", ("embed", "mlp")),
    (r"hyper/encoder/encoder/block/\d+/layer/1/DenseReluDense/wi_1/kernel", ("embed", "mlp")),
    (r"hyper/encoder/encoder/block/\d+/layer/1/DenseReluDense/wo/kernel", ("mlp", "embed")),
    (r"hyper/encoder/encoder/block/\d+/layer/1/layer_norm/weight", ("embed",)),
    (r"hyper/encoder/encoder/final_layer_norm/weight", ("embed",)),
]
