# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T5.1.1 Transformer model.
Altered to include hypernet stuff.
"""

from typing import Any, Callable, Iterable, Union

import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import partitioning as nn_partitioning
from jax import lax
from t5x.examples.t5 import layers
from t5x.examples.t5.layers import DenseGeneral, _convert_to_activation_function
from t5x.examples.t5.network import Decoder, T5Config
from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaModel

# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes

# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]


@struct.dataclass
class HyperT5Config(T5Config):
    adapter_size: int = 64
    hbottleneck_size: int = 128
    roberta_model: str = "roberta-base"


class SimplerLinear(nn.Module):
    """Feed-forward block that allows output values to be set.
    TODO: allow gated-gelu here?
    TODO: hypernet init

    Attributes:
      output_dim: Output dimension.
      activations: Type of activations for layer. Can be string or flax module.
      kernel_init: Kernel function, passed to the dense layers.
      deterministic: Whether the dropout layers should be deterministic.
      intermediate_dropout_rate: Dropout rate used after the intermediate layers.
      dtype: Type for the dense layer.
    """

    output_dim: int = 2048
    act_fn: Union[str, Callable] = "gelu"
    kernel_init: Initializer = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
        """Applies SimpleLinear module."""
        # Iterate over specified MLP input activation functions.
        # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
        x = DenseGeneral(
            self.output_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("embed", "mlp"),
            name="wi",
        )(inputs)
        x = _convert_to_activation_function(self.act_fn)(x)
        output = nn.Dropout(rate=self.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic
        )  # Broadcast along length.

        return output


class Hypernet(nn.Module):
    config: HyperT5Config
    shared_embedding: nn.Module

    # we setup here as loading huggingface weights
    def setup(self):
        cfg = self.config
        roberta = FlaxRobertaModel.from_pretrained(cfg.roberta_model)
        self.encoder = roberta.module
        self.embedder = jnp.asarray(
            param_with_axes(
                "embedding",
                nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                (cfg.num_encoder_layers + cfg.num_decoder_layers, cfg.emb_dim),
                jnp.float32,
                axes=("vocab", "embed"),
            ),
            jnp.float32,
        )
        self.intermediate_embedder = SimplerLinear(
            output_dim=cfg.hbottleneck_size,
            act_fn="gelu",
            dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="intermediate_hypernet",
        )
        self.adapter_down_gen = SimplerLinear(
            output_dim=cfg.emb_dim * cfg.adapter_size,
            act_fn="gelu",
            dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="adapter_down_mlp",
        )
        self.adapter_up_gen = SimplerLinear(
            output_dim=cfg.emb_dim * cfg.adapter_size,
            act_fn="gelu",
            dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="adapter_up_mlp",
        )
        self.adapter_bias_down_gen = SimplerLinear(
            output_dim=cfg.adapter_size,
            act_fn="gelu",
            dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="adapter_bias_down_mlp",
        )
        self.adapter_bias_up_gen = SimplerLinear(
            output_dim=cfg.emb_dim,
            act_fn="gelu",
            dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="adapter_bia_up_mlp",
        )

    def __call__(self, encoder_input_tokens, deterministic=False):
        cfg = self.config
        # '1' is roberta pad token.
        output = self.encoder(encoder_input_tokens, encoder_input_tokens != 1)
        pooled_output = output[1]  # jnp.mean(output, axis=1)
        # grab embeds, and
        total_layers = cfg.num_encoder_layers + cfg.num_decoder_layers
        embeds = jnp.arange(total_layers)
        embeds = self.embedder[embeds][
            None,
        ]
        embeddings = jnp.repeat(embeds, pooled_output.shape[0], axis=0)

        hyper_input = jnp.concatenate(
            [embeddings, jnp.repeat(pooled_output[:, None], embeddings.shape[1], axis=1)], axis=-1
        )

        intermediate_embeddings = self.intermediate_embedder(
            hyper_input, deterministic=deterministic
        )
        adapter_down = self.adapter_down_gen(intermediate_embeddings, deterministic=deterministic)
        adapter_down = jnp.reshape(adapter_down, (-1, total_layers, cfg.emb_dim, cfg.adapter_size))
        adapter_up = self.adapter_up_gen(intermediate_embeddings, deterministic=deterministic)
        adapter_up = jnp.reshape(adapter_up, (-1, total_layers, cfg.adapter_size, cfg.emb_dim))
        adapter_bias_down = self.adapter_bias_down_gen(
            intermediate_embeddings, deterministic=deterministic
        )
        adapter_bias_up = self.adapter_bias_up_gen(
            intermediate_embeddings, deterministic=deterministic
        )
        return adapter_down, adapter_up, adapter_bias_down, adapter_bias_up


class HyperEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    config: HyperT5Config
    relative_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        inputs,
        adapter_wd=None,
        adapter_wu=None,
        adapter_bd=None,
        adapter_bu=None,
        encoder_mask=None,
        deterministic=False,
    ):
        cfg = self.config

        # Relative position embedding as attention biases.
        encoder_bias = self.relative_embedding(inputs.shape[-2], inputs.shape[-2], True)

        # Attention block.
        # TODO: add deep prefix tuning.
        assert inputs.ndim == 3
        x = layers.LayerNorm(dtype=cfg.dtype, name="pre_attention_layer_norm")(inputs)
        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        x = layers.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="attention",
        )(x, x, encoder_mask, encoder_bias, deterministic=deterministic)
        x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        lx = layers.LayerNorm(dtype=cfg.dtype, name="pre_mlp_layer_norm")(x)
        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        y = layers.MlpBlock(
            intermediate_dim=cfg.mlp_dim,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="mlp",
        )(lx, deterministic=deterministic)
        y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)
        # adapter block
        adapter_y = (
            lax.batch_matmul(y, adapter_wd)
            + adapter_bd[
                :,
                None,
            ]
        )
        adapter_y = nn.gelu(adapter_y)
        adapter_y = (
            lax.batch_matmul(adapter_y, adapter_wu)
            + adapter_bu[
                :,
                None,
            ]
        )
        # final residual connection
        # TODO: scaled add?
        y = y + x + adapter_y
        return y


class HyperEncoder(nn.Module):
    """A stack of encoder layers."""

    config: T5Config
    shared_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        encoder_input_tokens,
        adapter_wd=None,
        adapter_wu=None,
        adapter_bd=None,
        adapter_bu=None,
        encoder_mask=None,
        deterministic=False,
    ):
        cfg = self.config
        assert encoder_input_tokens.ndim == 2  # [batch, length]
        rel_emb = layers.RelativePositionBiases(
            num_buckets=32,
            max_distance=128,
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            embedding_init=nn.initializers.variance_scaling(1.0, "fan_avg", "uniform"),
            name="relpos_bias",
        )

        # [batch, length] -> [batch, length, emb_dim]
        x = self.shared_embedding(encoder_input_tokens.astype("int32"))
        x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(x, deterministic=deterministic)
        x = x.astype(cfg.dtype)

        for lyr in range(cfg.num_encoder_layers):
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            x = HyperEncoderLayer(config=cfg, relative_embedding=rel_emb, name=f"layers_{lyr}")(
                x,
                adapter_wd[:, lyr],
                adapter_wu[:, lyr],
                adapter_bd[:, lyr],
                adapter_bu[:, lyr],
                encoder_mask,
                deterministic,
            )

        x = layers.LayerNorm(dtype=cfg.dtype, name="encoder_norm")(x)
        return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)


class HyperTransformer(nn.Module):
    """An encoder-decoder Transformer model, with hypernets."""

    config: HyperT5Config

    def setup(self):
        cfg = self.config
        self.shared_embedding = layers.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.emb_dim,
            dtype=cfg.dtype,
            attend_dtype=jnp.float32,  # for logit training stability
            embedding_init=nn.initializers.normal(stddev=1.0),
            one_hot=True,
            name="token_embedder",
        )

        self.hyper = Hypernet(config=cfg, shared_embedding=self.shared_embedding)
        self.encoder = HyperEncoder(config=cfg, shared_embedding=self.shared_embedding)
        # TODO: also condition decoder.
        self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding)

    def encode(
        self, encoder_input_tokens, awd, awu, bd, bu, encoder_segment_ids=None, enable_dropout=True
    ):
        """Applies Transformer encoder-branch on the inputs."""
        cfg = self.config
        assert encoder_input_tokens.ndim == 2  # (batch, len)

        # Make padding attention mask.
        encoder_mask = layers.make_attention_mask(
            encoder_input_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype
        )
        # Add segmentation block-diagonal attention mask if using segmented data.
        if encoder_segment_ids is not None:
            encoder_mask = layers.combine_masks(
                encoder_mask,
                layers.make_attention_mask(
                    encoder_segment_ids, encoder_segment_ids, jnp.equal, dtype=cfg.dtype
                ),
            )

        return self.encoder(
            encoder_input_tokens, awd, awu, bd, bu, encoder_mask, deterministic=not enable_dropout
        )

    # TODO: add hypernet stuff here. Will require touching some beam search stuff.
    def decode(
        self,
        encoded,
        encoder_input_tokens,  # only needed for masks
        decoder_input_tokens,
        decoder_target_tokens,
        encoder_segment_ids=None,
        decoder_segment_ids=None,
        decoder_positions=None,
        enable_dropout=True,
        decode=False,
        max_decode_length=None,
    ):
        """Applies Transformer decoder-branch on encoded-input and target."""
        cfg = self.config

        # Make padding attention masks.
        if decode:
            # Do not mask decoder attention based on targets padding at
            # decoding/inference time.
            decoder_mask = None
            encoder_decoder_mask = layers.make_attention_mask(
                jnp.ones_like(decoder_target_tokens), encoder_input_tokens > 0, dtype=cfg.dtype
            )
        else:
            decoder_mask = layers.make_decoder_mask(
                decoder_target_tokens=decoder_target_tokens,
                dtype=cfg.dtype,
                decoder_segment_ids=decoder_segment_ids,
            )
            encoder_decoder_mask = layers.make_attention_mask(
                decoder_target_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype
            )

        # Add segmentation block-diagonal attention masks if using segmented data.
        if encoder_segment_ids is not None:
            if decode:
                raise ValueError(
                    "During decoding, packing should not be used but "
                    "`encoder_segment_ids` was passed to `Transformer.decode`."
                )

            encoder_decoder_mask = layers.combine_masks(
                encoder_decoder_mask,
                layers.make_attention_mask(
                    decoder_segment_ids, encoder_segment_ids, jnp.equal, dtype=cfg.dtype
                ),
            )

        logits = self.decoder(
            encoded,
            decoder_input_tokens=decoder_input_tokens,
            decoder_positions=decoder_positions,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            deterministic=not enable_dropout,
            decode=decode,
            max_decode_length=max_decode_length,
        )
        return logits

    def __call__(
        self,
        encoder_input_tokens,
        hyper_encoder_input_tokens,
        decoder_input_tokens,
        decoder_target_tokens,
        encoder_segment_ids=None,
        hyper_encoder_segment_ids=None,
        decoder_segment_ids=None,
        encoder_positions=None,
        hyper_encoder_positions=None,
        decoder_positions=None,
        *,
        enable_dropout: bool = True,
        decode: bool = False,
    ):
        """Applies Transformer model on the inputs.

        This method requires both decoder_target_tokens and decoder_input_tokens,
        which is a shifted version of the former. For a packed dataset, it usually
        has additional processing applied. For example, the first element of each
        sequence has id 0 instead of the shifted EOS id from the previous sequence.

        Args:
          encoder_input_tokens: input data to the encoder.
          decoder_input_tokens: input token to the decoder.
          decoder_target_tokens: target token to the decoder.
          encoder_segment_ids: encoder segmentation info for packed examples.
          decoder_segment_ids: decoder segmentation info for packed examples.
          encoder_positions: encoder subsequence positions for packed examples.
          decoder_positions: decoder subsequence positions for packed examples.
          enable_dropout: Ensables dropout if set to True.
          decode: Whether to prepare and use an autoregressive cache.

        Returns:
          logits array from full transformer.
        """
        # generate adapters
        awd, awu, bd, bu = self.hyper(hyper_encoder_input_tokens, deterministic=not enable_dropout)
        encoded = self.encode(
            encoder_input_tokens,
            awd,
            awu,
            bd,
            bu,
            encoder_segment_ids=encoder_segment_ids,
            enable_dropout=enable_dropout,
        )

        return self.decode(
            encoded,
            encoder_input_tokens,  # only used for masks
            decoder_input_tokens,
            decoder_target_tokens,
            encoder_segment_ids=encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            enable_dropout=enable_dropout,
            decode=decode,
        )
