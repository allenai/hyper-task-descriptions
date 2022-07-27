"""
Partition map for lora weights
See https://github.com/google-research/t5x/blob/main/docs/usage/partitioning.md
for details on how this works.
"""

lora_axes_names_override = [
    (
        r"encoder/layer_\\d+/(self_attention|attention|encoder_decoder_attention)/(query|key|value)/lora_a",
        ("embed", "joined_kv"),
    ),
    (r"encoder/layer_\\d+/(self_attention|attention|encoder_decoder_attention)/(query|key|value)/lora_b",
     ("joined_kv",)),
    (r"encoder/layer_\\d+/(self_attention|attention|encoder_decoder_attention)/out/lora_a", ("joined_kv", "embed")),
    (r"encoder/layer_\\d+/(self_attention|attention|encoder_decoder_attention)/out/lora_b", ("embed",)),
]
