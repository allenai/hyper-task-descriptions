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

from flax import linen as nn
from flax import struct
import jax.numpy as jnp
from t5x.examples.t5 import layers

from t5x.examples.t5.network import T5Config, Decoder
from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaModel


@struct.dataclass
class HyperT5Config(T5Config):
    adapter_size: int = 64
    hbottleneck_size_size: int = 128


class Hypernet(nn.Module):
    config: HyperT5Config

    def setup(self):
        cfg = self.config
        self.roberta = FlaxRobertaModel.from_pretrained("roberta-base")
        self.embed = layers.Embed(
            num_embeddings=cfg.num_encoder_layers + cfg.num_decoder_layers,
            features=cfg.emb_dim,
            name="layer_embedder",
        )

    @nn.compact
    def __call__(self, encoder_input_tokens, encoder_mask=None, deterministic=False):
        cfg = self.config
        # TODO: convert t5 input ids to roberta ones. Or use t5 encoder instead.
        outputs = self.roberta(encoder_input_tokens, encoder_mask)
        pooled_output = outputs[1]
        # grab embeds, and
        embeddings = self.embed(jnp.arange(cfg.num_encoder_layers + cfg.num_decoder_layers))
        embeddings = jnp.repeat(embeddings, pooled_output.shape[0], axis=0)

        hyper_input = jnp.concatenate(
            [embeddings, jnp.repeat(pooled_output, embeddings.shape[1], axis=0)], axis=-1
        )

        intermediate_embeddings = layers.MlpBlock(
            intermediate_dim=cfg.hbottleneck_size,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="mlp",
        )(hyper_input, deterministic=deterministic)
        adapter_down = layers.MlpBlock(
            intermediate_dim=cfg.mlp_dim * cfg.adapter_size,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="mlp",
        )(intermediate_embeddings, deterministic=deterministic)
        adapter_up = layers.MlpBlock(
            intermediate_dim=cfg.mlp_dim * cfg.adapter_size,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="mlp",
        )(intermediate_embeddings, deterministic=deterministic)
        adapter_bias_down = layers.MlpBlock(
            intermediate_dim=cfg.adapter_size,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="mlp",
        )(intermediate_embeddings, deterministic=deterministic)
        adapter_bias_up = layers.MlpBlock(
            intermediate_dim=cfg.mlp_dim,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="mlp",
        )(intermediate_embeddings, deterministic=deterministic)
        return adapter_down, adapter_up, adapter_bias_down, adapter_bias_up


class HyperEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    config: HyperT5Config
    relative_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        inputs,
        adapter_wu=None,
        adapter_wd=None,
        adapter_bu=None,
        adapter_bd=None,
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
        adapter_y = adapter_wd @ y + adapter_bd
        adapter_y = nn.gelu(adapter_y)
        adapter_y = adapter_wu @ adapter_y + adapter_bu
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
        adapter_wu=None,
        adapter_wd=None,
        adapter_bu=None,
        adapter_bd=None,
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
                adapter_wu[:, lyr],
                adapter_wd[:, lyr],
                adapter_bu[:, lyr],
                adapter_bd[:, lyr],
                encoder_mask,
                deterministic,
            )

        x = layers.LayerNorm(dtype=cfg.dtype, name="encoder_norm")(x)
        return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)


class HyperTransformer(nn.Module):
    """An encoder-decoder Transformer model, with hypernets."""

    config: T5Config

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

        self.hyper = Hypernet(config=cfg)
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
        decoder_input_tokens,
        decoder_target_tokens,
        encoder_segment_ids=None,
        decoder_segment_ids=None,
        encoder_positions=None,
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
        awd, awu, bd, bu = self.hyper(encoder_input_tokens)
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
