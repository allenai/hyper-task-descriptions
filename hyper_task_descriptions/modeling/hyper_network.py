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
from jax import lax
from t5x.examples.t5 import layers
from t5x.examples.t5.network import T5Config
from transformers.models.roberta.modeling_flax_roberta import (
    FlaxRobertaModel,
    create_position_ids_from_input_ids,
)
from typing_extensions import TypeAlias

from hyper_task_descriptions.modeling.layers import (
    MlpBlock,
    MultiHeadDotProductAttentionWithPrefix,
    SimpleLinear,
)

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
class HyperT5Config(T5Config):
    add_adapters: bool = True
    layer_embed_size: int = 10
    adapter_size: int = 64
    hbottleneck_size: int = 128
    num_prefix_tokens: int = 30
    roberta_model: str = "hamishivi/fixed-roberta-base"  # fixes some partitioning issues
    roberta_max_position_embeddings: int = 520
    roberta_type_vocab_size: int = 8
    roberta_vocab_size: int = 50272


class Hypernet(nn.Module):
    config: HyperT5Config
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
        total_layers = cfg.num_encoder_layers + cfg.num_decoder_layers
        self.adapter_down_gen = [
            SimpleLinear(
                output_dim=cfg.emb_dim * cfg.adapter_size,
                act_fn="linear",
                dropout_rate=cfg.dropout_rate,
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                kernel_init=nn.initializers.normal(0.02),
                name=f"adapter_down_mlp_{i}",
            )
            for i in range(total_layers)
        ]
        self.adapter_up_gen = [
            SimpleLinear(
                output_dim=cfg.emb_dim * cfg.adapter_size,
                act_fn="linear",
                dropout_rate=cfg.dropout_rate,
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                kernel_init=nn.initializers.normal(0.02),
                name=f"adapter_up_mlp_{i}",
            )
            for i in range(total_layers)
        ]
        self.adapter_bias_down_gen = [
            SimpleLinear(
                output_dim=cfg.adapter_size,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                kernel_init=nn.initializers.normal(0.02),
                name=f"adapter_bias_down_mlp_{i}",
                dropout_rate=cfg.dropout_rate,
            )
            for i in range(total_layers)
        ]
        self.adapter_bias_up_gen = [
            SimpleLinear(
                output_dim=cfg.emb_dim,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                kernel_init=nn.initializers.normal(0.02),
                name=f"adapter_bias_up_mlp_{i}",
                dropout_rate=cfg.dropout_rate,
            )
            for i in range(total_layers)
        ]
        self.prefix_key_gen = [
            SimpleLinear(
                output_dim=cfg.num_prefix_tokens * cfg.num_heads * cfg.head_dim,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                kernel_init=nn.initializers.normal(0.02),
                name=f"prefix_key_mlp_{i}",
                dropout_rate=cfg.dropout_rate,
            )
            for i in range(total_layers)
        ]
        self.prefix_value_gen = [
            SimpleLinear(
                output_dim=cfg.num_prefix_tokens * cfg.num_heads * cfg.head_dim,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                kernel_init=nn.initializers.normal(0.02),
                name=f"prefix_value_mlp_{i}",
                dropout_rate=cfg.dropout_rate,
            )
            for i in range(total_layers)
        ]
        self.prefix_key_gen_cc = [
            SimpleLinear(
                output_dim=cfg.num_prefix_tokens * cfg.num_heads * cfg.head_dim,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                kernel_init=nn.initializers.normal(0.02),
                name=f"prefix_key_cc_mlp_{i}",
                dropout_rate=cfg.dropout_rate,
            )
            for i in range(total_layers)
        ]
        self.prefix_value_gen_cc = [
            SimpleLinear(
                output_dim=cfg.num_prefix_tokens * cfg.num_heads * cfg.head_dim,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                kernel_init=nn.initializers.normal(0.02),
                name=f"prefix_value_cc_mlp_{i}",
                dropout_rate=cfg.dropout_rate,
            )
            for i in range(total_layers)
        ]

    def __call__(self, encoder_input_tokens, deterministic=False):
        cfg = self.config
        # TODO: use config to determine padding token id
        # '1' is roberta pad token.
        attn_mask = encoder_input_tokens != 1
        # the position ids created by the FlaxRobertaModule are wrong, so use correct func.
        pos_ids = create_position_ids_from_input_ids(encoder_input_tokens, 1)
        output = self.encoder(encoder_input_tokens, attn_mask, position_ids=pos_ids)
        # average representation for embeds
        sum_embeds = (output[0] * attn_mask[:, :, None]).sum(axis=1) / attn_mask.sum(axis=1)[
            :, None
        ]
        # save pooled output for later (eg contrastive training)
        contrastive_output = self.contrastive_head(sum_embeds, deterministic=deterministic)
        self.sow("intermediates", "features", contrastive_output)
        # no layer embeddings for now.
        # # add the layer embeddings, and pass through a single mlp layer
        # total_layers = cfg.num_encoder_layers + cfg.num_decoder_layers
        # embeds = jnp.arange(total_layers)
        # embeds = self.embedder[embeds][
        #     None,
        # ]
        # embeddings = jnp.repeat(embeds, sum_embeds.shape[0], axis=0)

        # hyper_input = jnp.concatenate(
        #     [embeddings, jnp.repeat(sum_embeds[:, None], embeddings.shape[1], axis=1)], axis=-1
        # )

        intermediate_embeddings = self.intermediate_embedder(
            sum_embeds, deterministic=deterministic
        )
        # add the layer embeddings, and pass through a single mlp layer
        total_layers = cfg.num_encoder_layers + cfg.num_decoder_layers

        # generate all our adapters, prefixes, etc.
        adapter_downs, adapter_ups, adapter_bias_downs, adapter_bias_ups = [], [], [], []
        prefix_keys, prefix_values = [], []
        prefix_keys_cc, prefix_values_cc = [], []
        for i in range(total_layers):
            adapter_downs.append(
                self.adapter_down_gen[i](intermediate_embeddings, deterministic=deterministic)
            )
            adapter_downs[-1] = adapter_downs[-1].reshape(-1, cfg.emb_dim, cfg.adapter_size)
            adapter_ups.append(
                self.adapter_up_gen[i](intermediate_embeddings, deterministic=deterministic)
            )
            adapter_ups[-1] = adapter_ups[-1].reshape(-1, cfg.adapter_size, cfg.emb_dim)
            adapter_bias_downs.append(
                self.adapter_bias_down_gen[i](intermediate_embeddings, deterministic=deterministic)
            )
            adapter_bias_ups.append(
                self.adapter_bias_up_gen[i](intermediate_embeddings, deterministic=deterministic)
            )
            prefix_keys.append(
                self.prefix_key_gen[i](intermediate_embeddings, deterministic=deterministic)
            )
            prefix_keys[-1] = prefix_keys[-1].reshape(
                -1, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim
            )
            prefix_values.append(
                self.prefix_value_gen[i](intermediate_embeddings, deterministic=deterministic)
            )
            prefix_values[-1] = prefix_values[-1].reshape(
                -1, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim
            )
            prefix_keys_cc.append(
                self.prefix_key_gen_cc[i](intermediate_embeddings, deterministic=deterministic)
            )
            prefix_keys_cc[-1] = prefix_keys_cc[-1].reshape(
                -1, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim
            )
            prefix_values_cc.append(
                self.prefix_value_gen_cc[i](intermediate_embeddings, deterministic=deterministic)
            )
            prefix_values_cc[-1] = prefix_values_cc[-1].reshape(
                -1, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim
            )
        adapter_down = jnp.stack(adapter_downs, axis=1)
        adapter_up = jnp.stack(adapter_ups, axis=1)
        adapter_bias_down = jnp.stack(adapter_bias_downs, axis=1)
        adapter_bias_up = jnp.stack(adapter_bias_ups, axis=1)
        prefix_key = jnp.stack(prefix_keys, axis=1)
        prefix_value = jnp.stack(prefix_values, axis=1)
        prefix_key_cc = jnp.stack(prefix_keys_cc, axis=1)
        prefix_value_cc = jnp.stack(prefix_values_cc, axis=1)
        return (
            adapter_down,
            adapter_up,
            adapter_bias_down,
            adapter_bias_up,
            prefix_key,
            prefix_value,
            prefix_key_cc,
            prefix_value_cc,
        )
        return (
            adapter_down,
            adapter_up,
            adapter_bias_down,
            adapter_bias_up,
            prefix_key,
            prefix_value,
        )


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
        x = MultiHeadDotProductAttentionWithPrefix(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="attention",
        )(x, x, prefix_key, prefix_value, encoder_mask, encoder_bias, deterministic=deterministic)
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
        if cfg.add_adapters:
            adapter_y = (
                lax.batch_matmul(lx, adapter_wd)
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
            y = y + adapter_y
        # final residual connection
        # TODO: scaled add?
        y = y + x
        return y


class HyperDecoderLayer(nn.Module):
    """Transformer decoder layer that attends to the encoder."""

    config: HyperT5Config
    relative_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        inputs,
        encoded,
        adapter_wd=None,
        adapter_wu=None,
        adapter_bd=None,
        adapter_bu=None,
        prefix_key=None,
        prefix_value=None,
        prefix_key_cc=None,
        prefix_value_cc=None,
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
        x = MultiHeadDotProductAttentionWithPrefix(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="self_attention",
        )(
            x,
            x,
            prefix_key,
            prefix_value,
            decoder_mask,
            decoder_bias,
            deterministic=deterministic,
            decode=decode,
        )
        x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(x, deterministic=deterministic)
        x = x + inputs

        # Encoder-Decoder block.
        y = layers.LayerNorm(dtype=cfg.dtype, name="pre_cross_attention_layer_norm")(x)
        y = MultiHeadDotProductAttentionWithPrefix(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="encoder_decoder_attention",
        )(
            y,
            encoded,
            prefix_key_cc,
            prefix_value_cc,
            encoder_decoder_mask,
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
        if cfg.add_adapters:
            adapter_z = (
                lax.batch_matmul(lz, adapter_wd)
                + adapter_bd[
                    :,
                    None,
                ]
            )
            adapter_z = nn.gelu(adapter_z)
            adapter_z = (
                lax.batch_matmul(adapter_z, adapter_wu)
                + adapter_bu[
                    :,
                    None,
                ]
            )
            # final residual connection
            # TODO: scaled add?
            z = z + adapter_z
        z = z + y
        return z


class HyperEncoder(nn.Module):
    """A stack of encoder layers."""

    config: HyperT5Config
    shared_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        encoder_input_tokens,
        adapter_wd=None,
        adapter_wu=None,
        adapter_bd=None,
        adapter_bu=None,
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
            x = HyperEncoderLayer(config=cfg, relative_embedding=rel_emb, name=f"layers_{lyr}")(
                x,
                adapter_wd[:, lyr],
                adapter_wu[:, lyr],
                adapter_bd[:, lyr],
                adapter_bu[:, lyr],
                prefix_key[:, lyr],
                prefix_value[:, lyr],
                encoder_mask,
                deterministic,
            )

        x = layers.LayerNorm(dtype=cfg.dtype, name="encoder_norm")(x)
        return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)


class HyperDecoder(nn.Module):
    config: HyperT5Config
    shared_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        encoded,
        decoder_input_tokens,
        adapter_wd=None,
        adapter_wu=None,
        adapter_bd=None,
        adapter_bu=None,
        prefix_key=None,
        prefix_value=None,
        prefix_key_cc=None,
        prefix_value_cc=None,
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
            y = HyperDecoderLayer(config=cfg, relative_embedding=rel_emb, name=f"layers_{lyr}")(
                y,
                encoded,
                decoder_mask=decoder_mask,
                adapter_wd=adapter_wd[:, cfg.num_encoder_layers + lyr],
                adapter_wu=adapter_wu[:, cfg.num_encoder_layers + lyr],
                adapter_bd=adapter_bd[:, cfg.num_encoder_layers + lyr],
                adapter_bu=adapter_bu[:, cfg.num_encoder_layers + lyr],
                prefix_key=prefix_key[:, cfg.num_encoder_layers + lyr],
                prefix_value=prefix_value[:, cfg.num_encoder_layers + lyr],
                prefix_key_cc=prefix_key_cc[:, cfg.num_encoder_layers + lyr],
                prefix_value_cc=prefix_value_cc[:, cfg.num_encoder_layers + lyr],
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
        self.decoder = HyperDecoder(config=cfg, shared_embedding=self.shared_embedding)

    def encode(
        self,
        encoder_input_tokens,
        awd,
        awu,
        bd,
        bu,
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
            awd,
            awu,
            bd,
            bu,
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
        adapter_wd=None,
        adapter_wu=None,
        adapter_bd=None,
        adapter_bu=None,
        prefix_key=None,
        prefix_value=None,
        prefix_key_cc=None,
        prefix_value_cc=None,
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
            adapter_wd=adapter_wd,
            adapter_wu=adapter_wu,
            adapter_bd=adapter_bd,
            adapter_bu=adapter_bu,
            prefix_key=prefix_key,
            prefix_value=prefix_value,
            prefix_key_cc=prefix_key_cc,
            prefix_value_cc=prefix_value_cc,
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
        awd, awu, bd, bu, pk, pv, pkcc, pvcc = self.hyperencode(
            hyper_encoder_input_tokens, enable_dropout=enable_dropout
        )
        encoded = self.encode(
            encoder_input_tokens,
            awd,
            awu,
            bd,
            bu,
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
            adapter_wd=awd,
            adapter_wu=awu,
            adapter_bd=bd,
            adapter_bu=bu,
            prefix_key=pk,
            prefix_value=pv,
            prefix_key_cc=pkcc,
            prefix_value_cc=pvcc,
            encoder_segment_ids=encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            enable_dropout=enable_dropout,
            decode=decode,
        )
