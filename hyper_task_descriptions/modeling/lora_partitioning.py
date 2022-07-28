"""
Partition map for lora weights
See https://github.com/google-research/t5x/blob/main/docs/usage/partitioning.md
for details on how this works.
"""

lora_axes_names_override = [
    (
        r"encoder/layers_\d+/(self_attention|attention|encoder_decoder_attention)/(query|key|value)/lora_a",
        ("embed", "kv"),
    ),
    (r"encoder/layers_\d+/(self_attention|attention|encoder_decoder_attention)/(query|key|value)/lora_b",
     ("embed", "kv")),
    (r"encoder/layers_\d+/(self_attention|attention|encoder_decoder_attention)/out/lora_a", ("kv", "embed")),
    (r"encoder/layers_\d+/(self_attention|attention|encoder_decoder_attention)/out/lora_b", ("kv", "embed")),


    (
        r"decoder/layers_\d+/(self_attention|attention|encoder_decoder_attention)/(query|key|value)/lora_a",
        ("embed", "kv"),
    ),
    (r"decoder/layers_\d+/(self_attention|attention|encoder_decoder_attention)/(query|key|value)/lora_b",
     ("embed", "kv")),
    (r"decoder/layers_\d+/(self_attention|attention|encoder_decoder_attention)/out/lora_a", ("kv", "embed")),
    (r"decoder/layers_\d+/(self_attention|attention|encoder_decoder_attention)/out/lora_b", ("kv", "embed")),
]
