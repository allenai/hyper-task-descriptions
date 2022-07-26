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
from typing import Callable, Iterable

import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import partitioning as nn_partitioning
from t5x.examples.t5 import layers
from t5x.examples.t5.network import T5Config
from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaModel
from typing_extensions import TypeAlias

from hyper_task_descriptions.modeling.layers import MlpBlock, SimpleLinear
from hyper_task_descriptions.modeling.lora import LoraMultiHeadDotProductAttention

# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint

# Type annotations
Array: TypeAlias = jnp.ndarray
DType: TypeAlias = jnp.dtype
PRNGKey: TypeAlias = jnp.ndarray
Shape = Iterable[int]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]


@struct.dataclass
class HyperLoraT5Config(T5Config):
    add_adapters: bool = True
    layer_embed_size: int = 10
    adapter_size: int = 64
    hbottleneck_size: int = 128
    num_prefix_tokens: int = 30
    roberta_model: str = "hamishivi/fixed-roberta-base"  # fixes some partitioning issues
    roberta_max_position_embeddings: int = 520
    roberta_type_vocab_size: int = 8
    roberta_vocab_size: int = 50272
    lora_hyper_gen: bool = False
    lora_rank: int = 2


class HyperLoraNet(nn.Module):
    config: HyperLoraT5Config
    shared_embedding: nn.Module

    # we setup here as loading huggingface weights
    def setup(self):
        cfg = self.config
        roberta = FlaxRobertaModel.from_pretrained(
            cfg.roberta_model,
            max_position_embeddings=cfg.roberta_max_position_embeddings,
            type_vocab_size=cfg.roberta_type_vocab_size,
            vocab_size=cfg.roberta_vocab_size,
        )

        # encodes the task description
        self.encoder = roberta.module  # module = the 'actual' flax module

        self.embedder = jnp.asarray(
            param_with_axes(
                "embedding",
                nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                (cfg.num_encoder_layers + cfg.num_decoder_layers, cfg.layer_embed_size),
                jnp.float32,
                axes=("vocab", "embed"),
            ),
            jnp.float32,
        )
        self.intermediate_embedder = SimpleLinear(
            output_dim=cfg.hbottleneck_size,
            act_fn="gelu",
            dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            kernel_axes=("embed", "mlp"),
            name="intermediate_hypernet",
        )
        # contrastive head is two-layer mlp following simCLR
        # they use a sigmoid activation tho but using gelu for consistency
        self.contrastive_head = MlpBlock(
            intermediate_dim=roberta.config.hidden_size,
            activations=("gelu",),
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="contrastive_head",
        )

        # TODO: add lora generators here.

        # self.lora_a_gen = SimpleLinear(
        #     output_dim=cfg.emb_dim * cfg.adapter_size,
        #     act_fn="linear",
        #     dropout_rate=cfg.dropout_rate,
        #     dtype=cfg.dtype,
        #     kernel_axes=("mlp", "embed"),
        #     kernel_init=nn.initializers.variance_scaling(1e-18, "fan_avg", "uniform"),
        #     name="adapter_down_mlp",
        # )
        # self.lora_b_gen = SimpleLinear(
        #     output_dim=cfg.emb_dim * cfg.adapter_size,
        #     act_fn="linear",
        #     dropout_rate=cfg.dropout_rate,
        #     dtype=cfg.dtype,
        #     kernel_axes=("mlp", "embed"),
        #     kernel_init=nn.initializers.variance_scaling(1e-18, "fan_avg", "uniform"),
        #     name="adapter_up_mlp",
        # )

        self.prefix_key_gen = SimpleLinear(
            output_dim=cfg.num_prefix_tokens * cfg.num_heads * cfg.head_dim,
            act_fn="linear",
            dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            kernel_axes=("mlp", "embed"),
            name="prefix_key_mlp",
        )
        self.prefix_value_gen = SimpleLinear(
            output_dim=cfg.num_prefix_tokens * cfg.num_heads * cfg.head_dim,
            act_fn="linear",
            dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            kernel_axes=("mlp", "embed"),
            name="prefix_value_mlp",
        )

    def __call__(self, encoder_input_tokens, deterministic=False):
        cfg = self.config
        # '1' is roberta pad token.
        attn_mask = encoder_input_tokens != 1
        output = self.encoder(encoder_input_tokens, attn_mask)
        # average representation for embeds
        sum_embeds = output[0].sum(axis=1) / attn_mask.sum(axis=1)[:, None]
        # save pooled output for later (eg contrastive training)
        contrastive_output = self.contrastive_head(sum_embeds, deterministic=deterministic)
        self.sow("intermediates", "features", contrastive_output)
        # add the layer embeddings, and pass through a single mlp layer
        total_layers = cfg.num_encoder_layers + cfg.num_decoder_layers
        embeds = jnp.arange(total_layers)
        embeds = self.embedder[embeds][
            None,
        ]
        embeddings = jnp.repeat(embeds, sum_embeds.shape[0], axis=0)

        hyper_input = jnp.concatenate(
            [embeddings, jnp.repeat(sum_embeds[:, None], embeddings.shape[1], axis=1)], axis=-1
        )

        intermediate_embeddings = self.intermediate_embedder(
            hyper_input, deterministic=deterministic
        )
        # # generate all our adapters, prefixes, etc.
        # adapter_down = self.adapter_down_gen(intermediate_embeddings, deterministic=deterministic)
        # adapter_down = jnp.reshape(adapter_down, (-1, total_layers, cfg.emb_dim, cfg.adapter_size))
        # adapter_up = self.adapter_up_gen(intermediate_embeddings, deterministic=deterministic)
        # adapter_up = jnp.reshape(adapter_up, (-1, total_layers, cfg.adapter_size, cfg.emb_dim))
        # adapter_bias_down = self.adapter_bias_down_gen(
        #     intermediate_embeddings, deterministic=deterministic
        # )
        # adapter_bias_up = self.adapter_bias_up_gen(
        #     intermediate_embeddings, deterministic=deterministic
        # )

        # TODO: generate lora A, B weights.

        prefix_key = self.prefix_key_gen(intermediate_embeddings, deterministic=deterministic)
        prefix_key = jnp.reshape(
            prefix_key, (-1, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim)
        )
        prefix_value = self.prefix_value_gen(intermediate_embeddings, deterministic=deterministic)
        prefix_value = jnp.reshape(
            prefix_value, (-1, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim)
        )

        lora_a = None
        lora_b = None
        return (
            lora_a,
            lora_b,
            prefix_key,
            prefix_value,
        )


class LoraEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    config: HyperLoraT5Config
    relative_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        inputs,
        lora_a=None,
        lora_b=None,
        prefix_key=None,
        prefix_value=None,
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
        x = LoraMultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="attention",
            hyper_gen=cfg.lora_hyper_gen,
        )(
            x,
            x,
            encoder_mask,
            encoder_bias,
            lora_a=lora_a,
            lora_b=lora_b,
            deterministic=deterministic,
        )
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
        # # adapter block
        # if cfg.add_adapters:
        #     adapter_y = (
        #         lax.batch_matmul(lx, adapter_wd)
        #         + adapter_bd[
        #             :,
        #             None,
        #         ]
        #     )
        #     adapter_y = nn.gelu(adapter_y)
        #     adapter_y = (
        #         lax.batch_matmul(adapter_y, adapter_wu)
        #         + adapter_bu[
        #             :,
        #             None,
        #         ]
        #     )
        #     y = y + adapter_y
        # final residual connection
        # TODO: scaled add?
        y = y + x
        return y


class LoraDecoderLayer(nn.Module):
    """Transformer decoder layer that attends to the encoder."""

    config: HyperLoraT5Config
    relative_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        inputs,
        encoded,
        lora_a=None,
        lora_b=None,
        prefix_key=None,
        prefix_value=None,
        decoder_mask=None,
        encoder_decoder_mask=None,
        deterministic=False,
        decode=False,
        max_decode_length=None,
    ):
        cfg = self.config

        # Relative position embedding as attention biases.
        l = max_decode_length if decode and max_decode_length else inputs.shape[-2]  # noqa: E741
        decoder_bias = self.relative_embedding(l, l, False)

        # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
        x = layers.LayerNorm(dtype=cfg.dtype, name="pre_self_attention_layer_norm")(inputs)

        # Self-attention block
        x = LoraMultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="self_attention",
            hyper_gen=cfg.lora_hyper_gen,
        )(
            x,
            x,
            decoder_mask,
            decoder_bias,
            lora_a=lora_a,
            lora_b=lora_b,
            deterministic=deterministic,
            decode=decode,
        )
        x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(x, deterministic=deterministic)
        x = x + inputs

        # Encoder-Decoder block.
        y = layers.LayerNorm(dtype=cfg.dtype, name="pre_cross_attention_layer_norm")(x)
        y = LoraMultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="encoder_decoder_attention",
            hyper_gen=cfg.lora_hyper_gen,
        )(
            y,
            encoded,
            encoder_decoder_mask,
            lora_a=lora_a,
            lora_b=lora_b,
            deterministic=deterministic,
        )
        y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)
        y = y + x

        # MLP block.
        lz = layers.LayerNorm(dtype=cfg.dtype, name="pre_mlp_layer_norm")(y)
        z = layers.MlpBlock(
            intermediate_dim=cfg.mlp_dim,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            name="mlp",
        )(lz, deterministic=deterministic)
        z = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(z, deterministic=deterministic)
        # adapter block
        # if cfg.add_adapters:
        #     adapter_z = (
        #         lax.batch_matmul(lz, adapter_wd)
        #         + adapter_bd[
        #             :,
        #             None,
        #         ]
        #     )
        #     adapter_z = nn.gelu(adapter_z)
        #     adapter_z = (
        #         lax.batch_matmul(adapter_z, adapter_wu)
        #         + adapter_bu[
        #             :,
        #             None,
        #         ]
        #     )
        #     # final residual connection
        #     # TODO: scaled add?
        #     z = z + adapter_z
        z = z + y
        return z


class LoraEncoder(nn.Module):
    """A stack of encoder layers."""

    config: HyperLoraT5Config
    shared_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        encoder_input_tokens,
        lora_a=None,
        lora_b=None,
        prefix_key=None,
        prefix_value=None,
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
            x = LoraEncoderLayer(config=cfg, relative_embedding=rel_emb, name=f"layers_{lyr}")(
                x,
                lora_a,
                lora_b,
                # adapter_wd[:, lyr],
                # adapter_wu[:, lyr],
                # adapter_bd[:, lyr],
                # adapter_bu[:, lyr],
                prefix_key[:, lyr],
                prefix_value[:, lyr],
                encoder_mask,
                deterministic,
            )

        x = layers.LayerNorm(dtype=cfg.dtype, name="encoder_norm")(x)
        return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)


class LoraDecoder(nn.Module):
    config: HyperLoraT5Config
    shared_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        encoded,
        decoder_input_tokens,
        lora_a=None,
        lora_b=None,
        prefix_key=None,
        prefix_value=None,
        decoder_positions=None,
        decoder_mask=None,
        encoder_decoder_mask=None,
        deterministic=False,
        decode=False,
        max_decode_length=None,
    ):
        cfg = self.config
        assert decoder_input_tokens.ndim == 2  # [batch, len]
        rel_emb = layers.RelativePositionBiases(
            num_buckets=32,
            max_distance=128,
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            embedding_init=nn.initializers.variance_scaling(1.0, "fan_avg", "uniform"),
            name="relpos_bias",
        )

        # [batch, length] -> [batch, length, emb_dim]
        y = self.shared_embedding(decoder_input_tokens.astype("int32"))
        y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)
        y = y.astype(cfg.dtype)

        for lyr in range(cfg.num_decoder_layers):
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            y = LoraDecoderLayer(config=cfg, relative_embedding=rel_emb, name=f"layers_{lyr}")(
                y,
                encoded,
                decoder_mask=decoder_mask,
                # adapter_wd=adapter_wd[:, cfg.num_encoder_layers + lyr],
                # adapter_wu=adapter_wu[:, cfg.num_encoder_layers + lyr],
                # adapter_bd=adapter_bd[:, cfg.num_encoder_layers + lyr],
                # adapter_bu=adapter_bu[:, cfg.num_encoder_layers + lyr],
                lora_a=lora_a,
                lora_b=lora_b,
                prefix_key=prefix_key[:, cfg.num_encoder_layers + lyr],
                prefix_value=prefix_value[:, cfg.num_encoder_layers + lyr],
                encoder_decoder_mask=encoder_decoder_mask,
                deterministic=deterministic,
                decode=decode,
                max_decode_length=max_decode_length,
            )

        y = layers.LayerNorm(dtype=cfg.dtype, name="decoder_norm")(y)
        y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)

        # [batch, length, emb_dim] -> [batch, length, vocab_size]
        if cfg.logits_via_embedding:
            # Use the transpose of embedding matrix for logit transform.
            logits = self.shared_embedding.attend(y)
            # Correctly normalize pre-softmax logits for this shared case.
            logits = logits / jnp.sqrt(y.shape[-1])
        else:
            logits = layers.DenseGeneral(
                cfg.vocab_size,
                dtype=jnp.float32,  # Use float32 for stabiliity.
                kernel_axes=("embed", "vocab"),
                name="logits_dense",
            )(y)
        return logits


class LoraTransformer(nn.Module):
    """An encoder-decoder Transformer model, with hypernets."""

    config: HyperLoraT5Config

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

        self.hyper = HyperLoraNet(config=cfg, shared_embedding=self.shared_embedding)
        self.encoder = LoraEncoder(config=cfg, shared_embedding=self.shared_embedding)
        self.decoder = LoraDecoder(config=cfg, shared_embedding=self.shared_embedding)

    def encode(
        self,
        encoder_input_tokens,
        # awd,
        # awu,
        # bd,
        # bu,
        lora_a,
        lora_b,
        pk,
        pv,
        encoder_segment_ids=None,
        enable_dropout=True,
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
            encoder_input_tokens,
            lora_a,
            lora_b,
            pk,
            pv,
            encoder_mask,
            deterministic=not enable_dropout,
        )

    def hyperencode(self, hyper_input_tokens, enable_dropout=True):
        return self.hyper(hyper_input_tokens, deterministic=not enable_dropout)

    # TODO: add hypernet stuff here. Will require touching some beam search stuff.
    def decode(
        self,
        encoded,
        encoder_input_tokens,  # only needed for masks
        decoder_input_tokens,
        decoder_target_tokens,
        # adapter_wd=None,
        # adapter_wu=None,
        # adapter_bd=None,
        # adapter_bu=None,
        lora_a=None,
        lora_b=None,
        prefix_key=None,
        prefix_value=None,
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
            # adapter_wd=adapter_wd,
            # adapter_wu=adapter_wu,
            # adapter_bd=adapter_bd,
            # adapter_bu=adapter_bu,
            lora_a=lora_a,
            lora_b=lora_b,
            prefix_key=prefix_key,
            prefix_value=prefix_value,
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
          enable_dropout: Enables dropout if set to True.
          decode: Whether to prepare and use an autoregressive cache.

        Returns:
          logits array from full transformer.
        """
        # generate adapters
        lora_a, lora_b, pk, pv = self.hyperencode(
            hyper_encoder_input_tokens, enable_dropout=enable_dropout
        )
        encoded = self.encode(
            encoder_input_tokens,
            lora_a,
            lora_b,
            pk,
            pv,
            encoder_segment_ids=encoder_segment_ids,
            enable_dropout=enable_dropout,
        )

        return self.decode(
            encoded,
            encoder_input_tokens,  # only used for masks
            decoder_input_tokens,
            decoder_target_tokens,
            lora_a=lora_a,
            lora_b=lora_b,
            # adapter_wd=awd,
            # adapter_wu=awu,
            # adapter_bd=bd,
            # adapter_bu=bu,
            prefix_key=pk,
            prefix_value=pv,
            encoder_segment_ids=encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            enable_dropout=enable_dropout,
            decode=decode,
        )
