from transformers.models.t5.modeling_flax_t5 import FlaxT5EncoderModule
from flax import linen as nn
import jax.numpy as jnp


class FlaxT5EncoderModuleSharedEmbedding(FlaxT5EncoderModule):
    def __init__(self, config, dtype=jnp.float32, shared_embedding=None, **kwargs):
        super().__init__(config, dtype=dtype, **kwargs)
        self.shared_embedding = shared_embedding

    def setup(self):
        super().setup()
        self.shared = self.shared_embedding
