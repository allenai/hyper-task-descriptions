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
from collections import defaultdict
from typing import Callable, Iterable

import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import partitioning as nn_partitioning
from jax import lax
from t5x.examples.t5 import layers
from t5x.examples.t5.network import T5Config
from transformers import FlaxT5EncoderModel
from typing_extensions import TypeAlias

from hyper_task_descriptions.modeling.layers import (
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
    use_adapter: bool = True
    use_prefix: bool = True
    layer_embed_size: int = 10
    adapter_size: int = 64
    hbottleneck_size: int = 128
    num_prefix_tokens: int = 30
    hyperencoder_model: str = "google/t5-large-lm-adapt"
    lora_hyper_gen: bool = False
    lora_ranks: tuple = (2, None, 2, None)
    layer_embedding_method: str = "component"  # concat, layer, component


class Hypernet(nn.Module):
    config: HyperT5Config
    shared_embedding: nn.Module

    # we setup here as loading huggingface weights
    def setup(self):
        cfg = self.config
        assert cfg.layer_embedding_method in [
            "concat",
            "layer",
            "component",
        ], "Invalid layer embedding method"
        encoder = FlaxT5EncoderModel.from_pretrained(cfg.hyperencoder_model, from_pt=True)

        # encodes the task description
        self.encoder = encoder.module  # module = the 'actual' flax module

        # setup embeddings
        layer_embed_components = cfg.num_encoder_layers + cfg.num_decoder_layers
        num_components = 0
        if cfg.use_adapter:
            num_components += 4  # adapter up, down, bias up, bias down
        if cfg.use_prefix:
            num_components += 4  # prefix key, value, prefix key cc, prefix value cc
        if cfg.layer_embedding_method == "component":
            layer_embed_components *= num_components

        self.embedder = jnp.asarray(
            param_with_axes(
                "embedding",
                nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                (layer_embed_components, encoder.config.d_model),
                jnp.float32,
                axes=("vocab", "embed"),
            ),
            jnp.float32,
        )
        if cfg.use_adapter:
            self.adapter_down_gen = SimpleLinear(
                output_dim=cfg.emb_dim * cfg.adapter_size,
                act_fn="linear",
                dropout_rate=cfg.dropout_rate,
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                name="adapter_down_mlp",
            )
            self.adapter_up_gen = SimpleLinear(
                output_dim=cfg.emb_dim * cfg.adapter_size,
                act_fn="linear",
                dropout_rate=cfg.dropout_rate,
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                name="adapter_up_mlp",
            )
            self.adapter_bias_down_gen = SimpleLinear(
                output_dim=cfg.adapter_size,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                name="adapter_bias_down_mlp",
                dropout_rate=cfg.dropout_rate,
            )
            self.adapter_bias_up_gen = SimpleLinear(
                output_dim=cfg.emb_dim,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                name="adapter_bias_up_mlp",
                dropout_rate=cfg.dropout_rate,
            )
        if cfg.use_prefix:
            self.prefix_key_gen = SimpleLinear(
                output_dim=cfg.num_prefix_tokens * cfg.num_heads * cfg.head_dim,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                name="prefix_key_mlp",
                dropout_rate=cfg.dropout_rate,
            )
            self.prefix_value_gen = SimpleLinear(
                output_dim=cfg.num_prefix_tokens * cfg.num_heads * cfg.head_dim,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                name="prefix_value_ml",
                dropout_rate=cfg.dropout_rate,
            )
            self.prefix_key_gen_cc = SimpleLinear(
                output_dim=cfg.num_prefix_tokens * cfg.num_heads * cfg.head_dim,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                name="prefix_key_cc_mlp",
                dropout_rate=cfg.dropout_rate,
            )
            self.prefix_value_gen_cc = SimpleLinear(
                output_dim=cfg.num_prefix_tokens * cfg.num_heads * cfg.head_dim,
                act_fn="linear",
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                name="prefix_value_cc_mlp",
                dropout_rate=cfg.dropout_rate,
            )

    def __call__(self, encoder_input_tokens, deterministic=False):
        cfg = self.config
        total_layers = cfg.num_encoder_layers + cfg.num_decoder_layers
        # 0 is t5 padding id.
        attn_mask = encoder_input_tokens != 0
        # get type issues otherwuse so make sure tokens are ints.
        encoder_input_tokens = encoder_input_tokens.astype("i4")
        output = self.encoder(encoder_input_tokens, attn_mask)
        # save pooled output for later (eg contrastive training)
        mean_seq = (output[0] * attn_mask[:, :, None]).sum(axis=1) / attn_mask.sum(axis=1)[:, None]
        self.sow("intermediates", "features", mean_seq)
        # layer embedding setup
        if cfg.layer_embedding_method == "layer":
            seq_output = (output[0] * attn_mask[:, :, None])[
                :, :, None
            ]  # to prevent padding annoying us.
            layer_embeds = self.embedder[None, :, None, :].repeat(
                encoder_input_tokens.shape[0], axis=0
            )
            sum_embeds = layers.dot_product_attention(
                layer_embeds,
                seq_output,
                seq_output,
                (1 - attn_mask[:, None, None, :]) * -1e9,
                deterministic=deterministic,
            )[:, :, 0, :]
        elif cfg.layer_embedding_method == "component":
            seq_output = (output[0] * attn_mask[:, :, None])[
                :, :, None
            ]  # to prevent padding annoying us.
            layer_embeds = self.embedder[None, :, None, :].repeat(
                encoder_input_tokens.shape[0], axis=0
            )
            # layer embeds = [batch size, num_layers, 1, hidden size]
            # seq output = [batch size, instr. length, 1, hidden size]
            sum_embeds = layers.dot_product_attention(
                layer_embeds,
                seq_output,
                seq_output,
                (1 - attn_mask[:, None, None, :]) * -1e9,
                deterministic=deterministic,
            )[:, :, 0, :]
            # reshape to [batch, layers, num_comp, feats]
            sum_embeds = sum_embeds.reshape(seq_output.shape[0], total_layers, 8, -1)
        else:  # else = use concat
            # layer embeds - repeat in batch, length dim
            sum_embeds = sum_embeds[:, None].repeat(total_layers, axis=1)
            layer_embs = self.embedder[
                None,
                :,
            ].repeat(sum_embeds.shape[0], axis=0)
            sum_embeds = jnp.concatenate([mean_seq, layer_embs], axis=-1)

        # add the layer embeddings, and pass through a single mlp layer
        total_layers = cfg.num_encoder_layers + cfg.num_decoder_layers

        # generate all our adapters, prefixes, etc.

        generated_parameter_dict = defaultdict(list)
        # choose our specific input to the hypernet. feel free to customize.

        def choose_hypernet_input(intermediate_embed, component_id):
            if not cfg.use_adapter and cfg.use_prefix:
                component_id -= 4  # prefix only, we only have 4 comps.
            if cfg.layer_embedding_method == "component":
                return intermediate_embed[:, :, component_id]
            return intermediate_embed

        if cfg.use_adapter:
            # adapter weight down
            adapter_wd = self.adapter_down_gen(
                choose_hypernet_input(sum_embeds, 0), deterministic=deterministic
            )
            adapter_wd = adapter_wd.reshape(-1, total_layers, cfg.emb_dim, cfg.adapter_size)
            adapter_wd = adapter_wd / jnp.sqrt(sum_embeds.shape[-1])
            generated_parameter_dict["adapter_wd"] = adapter_wd
            # adapter weight up
            adapter_wu = self.adapter_up_gen(
                choose_hypernet_input(sum_embeds, 1), deterministic=deterministic
            )
            adapter_wu = adapter_wu.reshape(-1, total_layers, cfg.adapter_size, cfg.emb_dim)
            adapter_wu = adapter_wu / jnp.sqrt(sum_embeds.shape[-1])
            generated_parameter_dict["adapter_wu"] = adapter_wu
            # adapter bias down
            adapter_bd = self.adapter_bias_down_gen(
                choose_hypernet_input(sum_embeds, 2), deterministic=deterministic
            )
            adapter_bd = adapter_bd / jnp.sqrt(sum_embeds.shape[-1])
            generated_parameter_dict["adapter_bd"] = adapter_bd
            # adapter bias up
            adapter_bu = self.adapter_bias_up_gen(
                choose_hypernet_input(sum_embeds, 3), deterministic=deterministic
            )
            adapter_bu = adapter_bu / jnp.sqrt(sum_embeds.shape[-1])
            generated_parameter_dict["adapter_bu"] = adapter_bu
        if cfg.use_prefix:
            # prefix key
            prefix_key = self.prefix_key_gen(
                choose_hypernet_input(sum_embeds, 4), deterministic=deterministic
            )
            prefix_key = prefix_key.reshape(
                -1, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim
            )
            prefix_key = prefix_key / jnp.sqrt(sum_embeds.shape[-1])
            generated_parameter_dict["prefix_key"] = prefix_key
            # prefix value
            prefix_value = self.prefix_value_gen(
                choose_hypernet_input(sum_embeds, 5), deterministic=deterministic
            )
            prefix_value = prefix_value.reshape(
                -1, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim
            )
            prefix_value = prefix_value / jnp.sqrt(sum_embeds.shape[-1])
            generated_parameter_dict["prefix_value"] = prefix_value
            # prefix key cc
            prefix_key_cc = self.prefix_key_gen_cc(
                choose_hypernet_input(sum_embeds, 6), deterministic=deterministic
            )
            prefix_key_cc = prefix_key_cc.reshape(
                -1, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim
            )
            prefix_key_cc = prefix_key_cc / jnp.sqrt(sum_embeds.shape[-1])
            generated_parameter_dict["prefix_key_cc"] = prefix_key_cc
            # prefix value cc
            prefix_value_cc = self.prefix_value_gen_cc(
                choose_hypernet_input(sum_embeds, 7), deterministic=deterministic
            )
            prefix_value_cc = prefix_value_cc.reshape(
                -1, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim
            )
            prefix_value_cc = prefix_value_cc / jnp.sqrt(sum_embeds.shape[-1])
            generated_parameter_dict["prefix_value_cc"] = prefix_value_cc

        return generated_parameter_dict


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
            use_prefix=cfg.use_prefix,
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
        if cfg.use_adapter:
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
            use_prefix=cfg.use_prefix,
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
            use_prefix=cfg.use_prefix,
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
        if cfg.use_adapter:
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
                adapter_wd=adapter_wd[:, lyr] if adapter_wd is not None else None,
                adapter_wu=adapter_wu[:, lyr] if adapter_wu is not None else None,
                adapter_bd=adapter_bd[:, lyr] if adapter_bd is not None else None,
                adapter_bu=adapter_bu[:, lyr] if adapter_bu is not None else None,
                prefix_key=prefix_key[:, lyr] if prefix_key is not None else None,
                prefix_value=prefix_value[:, lyr] if prefix_value is not None else None,
                encoder_mask=encoder_mask,
                deterministic=deterministic,
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
                adapter_wd=adapter_wd[:, cfg.num_encoder_layers + lyr]
                if adapter_wd is not None
                else None,
                adapter_wu=adapter_wu[:, cfg.num_encoder_layers + lyr]
                if adapter_wu is not None
                else None,
                adapter_bd=adapter_bd[:, cfg.num_encoder_layers + lyr]
                if adapter_bd is not None
                else None,
                adapter_bu=adapter_bu[:, cfg.num_encoder_layers + lyr]
                if adapter_bu is not None
                else None,
                prefix_key=prefix_key[:, cfg.num_encoder_layers + lyr]
                if prefix_key is not None
                else None,
                prefix_value=prefix_value[:, cfg.num_encoder_layers + lyr]
                if prefix_value is not None
                else None,
                prefix_key_cc=prefix_key_cc[:, cfg.num_encoder_layers + lyr]
                if prefix_key_cc is not None
                else None,
                prefix_value_cc=prefix_value_cc[:, cfg.num_encoder_layers + lyr]
                if prefix_value_cc is not None
                else None,
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
        adapters,
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

        adapters_ = {key: val for key, val in adapters.items() if not key.endswith("_cc")}
        return self.encoder(
            encoder_input_tokens,
            **adapters_,
            encoder_mask=encoder_mask,
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
        adapters=None,
        encoder_segment_ids=None,
        decoder_segment_ids=None,
        decoder_positions=None,
        enable_dropout=True,
        decode=False,
        max_decode_length=None,
    ):
        """Applies Transformer decoder-branch on encoded-input and target."""
        cfg = self.config

        adapters = adapters or {}

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
            **adapters,
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
        adapters = self.hyperencode(hyper_encoder_input_tokens, enable_dropout=enable_dropout)
        encoded = self.encode(
            encoder_input_tokens,
            adapters=adapters,
            encoder_segment_ids=encoder_segment_ids,
            enable_dropout=enable_dropout,
        )

        return self.decode(
            encoded,
            encoder_input_tokens,  # only used for masks
            decoder_input_tokens,
            decoder_target_tokens,
            adapters=adapters,
            encoder_segment_ids=encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            enable_dropout=enable_dropout,
            decode=decode,
        )
