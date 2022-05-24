"""
Partition map for roberta-base
See https://github.com/google-research/t5x/blob/main/docs/usage/partitioning.md
for details on how this works.
"""

roberta_axes_names_override = [
    (r"hyper/encoder/embeddings/[\w_]+/embedding", ("vocab", "embed")),
    # attention
    (
        r"hyper/encoder/encoder/layer/\d+/attention/self/(query|key|value)/kernel",
        ("embed", "joined_kv"),
    ),
    (r"hyper/encoder/encoder/layer/\d+/attention/self/(query|key|value)/bias", ("joined_kv",)),
    (r"hyper/encoder/encoder/layer/\d+/attention/output/dense/kernel", ("joined_kv", "embed")),
    (r"hyper/encoder/encoder/layer/\d+/attention/output/dense/bias", ("embed",)),
    # intermediate
    (r"hyper/encoder/encoder/layer/\d+/intermediate/dense/kernel", ("embed", "mlp")),
    (r"hyper/encoder/encoder/layer/\d+/intermediate/dense/bias", ("mlp",)),
    # output
    (r"hyper/encoder/encoder/layer/\d+/output/dense/kernel", ("mlp", "embed")),
    (r"hyper/encoder/encoder/layer/\d+/output/dense/bias", ("embed",)),
    # layer norms
    (r"hyper/encoder/encoder/layer/\d+/[\w_\/]+/LayerNorm/bias", ("embed",)),
    (r"hyper/encoder/encoder/layer/\d+/[\w_\/]+/LayerNorm/scale", ("embed",)),
    (r"hyper/encoder/embeddings/LayerNorm/bias", ("embed",)),
    (r"hyper/encoder/embeddings/LayerNorm/scale", ("embed",)),
    # pooler
    (r"hyper/encoder/pooler/dense/kernel", ("embed", "mlp")),
    (r"hyper/encoder/pooler/dense/bias", ("embed",)),
]
