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
from flax.linen import partitioning as nn_partitioning
from t5x.examples.t5 import layers
from t5x.examples.t5.network import Transformer
from transformers.models.roberta.modeling_flax_roberta import FlaxRobertaModel
from typing_extensions import TypeAlias

from hyper_task_descriptions.modeling.hyper_network import (
    HyperT5Config,
    HyperTransformer,
)
from hyper_task_descriptions.modeling.layers import MlpBlock, SimpleLinear
from hyper_task_descriptions.modeling.lora import (
    LoraMultiHeadDotProductAttentionWithPrefix,
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


class HyperLoraNet(nn.Module):
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
                (cfg.num_encoder_layers + 2 * cfg.num_decoder_layers, cfg.layer_embed_size),
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

        self.q_rank, self.k_rank, self.v_rank, self.o_rank = cfg.lora_ranks

        if cfg.lora_hyper_gen:
            if self.q_rank:
                self.lora_qa_gen = SimpleLinear(
                    output_dim=cfg.emb_dim * self.q_rank,
                    act_fn="linear",
                    dropout_rate=cfg.dropout_rate,
                    dtype=cfg.dtype,
                    kernel_axes=("embed", "joined_kv"),
                    kernel_init=nn.initializers.normal(0.01),
                    name="lora_qa_gen",
                )

                self.lora_qb_gen = SimpleLinear(
                    output_dim=self.q_rank * cfg.num_heads * cfg.head_dim,
                    act_fn="linear",
                    dropout_rate=cfg.dropout_rate,
                    dtype=cfg.dtype,
                    kernel_axes=("embed", "joined_kv"),
                    kernel_init=nn.initializers.zeros,
                    name="lora_qb_gen",
                )

            if self.k_rank:
                self.lora_ka_gen = SimpleLinear(
                    output_dim=cfg.emb_dim * self.k_rank,
                    act_fn="linear",
                    dropout_rate=cfg.dropout_rate,
                    dtype=cfg.dtype,
                    kernel_axes=("embed", "joined_kv"),
                    kernel_init=nn.initializers.normal(0.01),
                    name="lora_ka_gen",
                )

                self.lora_kb_gen = SimpleLinear(
                    output_dim=self.k_rank * cfg.num_heads * cfg.head_dim,
                    act_fn="linear",
                    dropout_rate=cfg.dropout_rate,
                    dtype=cfg.dtype,
                    kernel_axes=("embed", "joined_kv"),
                    kernel_init=nn.initializers.zeros,
                    name="lora_kb_gen",
                )

            if self.v_rank:
                self.lora_va_gen = SimpleLinear(
                    output_dim=cfg.emb_dim * self.v_rank,
                    act_fn="linear",
                    dropout_rate=cfg.dropout_rate,
                    dtype=cfg.dtype,
                    kernel_axes=("embed", "joined_kv"),
                    kernel_init=nn.initializers.normal(0.01),
                    name="lora_va_gen",
                )

                self.lora_vb_gen = SimpleLinear(
                    output_dim=self.v_rank * cfg.num_heads * cfg.head_dim,
                    act_fn="linear",
                    dropout_rate=cfg.dropout_rate,
                    dtype=cfg.dtype,
                    kernel_axes=("embed", "joined_kv"),
                    kernel_init=nn.initializers.zeros,
                    name="lora_vb_gen",
                )

            if self.o_rank:
                self.lora_oa_gen = SimpleLinear(
                    output_dim=cfg.num_heads * cfg.head_dim * self.o_rank,
                    act_fn="linear",
                    dropout_rate=cfg.dropout_rate,
                    dtype=cfg.dtype,
                    kernel_axes=("joined_kv", "embed"),
                    kernel_init=nn.initializers.normal(0.01),
                    name="lora_oa_gen",
                )

                self.lora_ob_gen = SimpleLinear(
                    output_dim=self.o_rank * cfg.emb_dim,
                    act_fn="linear",
                    dropout_rate=cfg.dropout_rate,
                    dtype=cfg.dtype,
                    kernel_axes=("embed", "joined_kv"),
                    kernel_init=nn.initializers.zeros,
                    name="lora_ob_gen",
                )

        self.use_prefix = cfg.use_prefix

        if self.use_prefix:
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

        # decoder has self and cross attention both, hence enc + 2 x dec lora layers.
        total_layers = cfg.num_encoder_layers + 2 * cfg.num_decoder_layers
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

        if cfg.lora_hyper_gen:
            if self.q_rank:
                lora_qa = self.lora_qa_gen(intermediate_embeddings, deterministic=deterministic)
                lora_qa = jnp.reshape(lora_qa, (-1, total_layers, cfg.emb_dim, self.q_rank))
                lora_qb = self.lora_qb_gen(intermediate_embeddings, deterministic=deterministic)
                lora_qb = jnp.reshape(
                    lora_qb, (-1, total_layers, self.q_rank, cfg.num_heads, cfg.head_dim)
                )
            else:
                lora_qa, lora_qb = None, None

            if self.k_rank:
                lora_ka = self.lora_ka_gen(intermediate_embeddings, deterministic=deterministic)
                lora_ka = jnp.reshape(lora_ka, (-1, total_layers, cfg.emb_dim, self.k_rank))
                lora_kb = self.lora_kb_gen(intermediate_embeddings, deterministic=deterministic)
                lora_kb = jnp.reshape(
                    lora_kb, (-1, total_layers, self.k_rank, cfg.num_heads, cfg.head_dim)
                )
            else:
                lora_ka, lora_kb = None, None

            if self.v_rank:
                lora_va = self.lora_va_gen(intermediate_embeddings, deterministic=deterministic)
                lora_va = jnp.reshape(lora_va, (-1, total_layers, cfg.emb_dim, self.v_rank))
                lora_vb = self.lora_vb_gen(intermediate_embeddings, deterministic=deterministic)
                lora_vb = jnp.reshape(
                    lora_vb, (-1, total_layers, self.v_rank, cfg.num_heads, cfg.head_dim)
                )
            else:
                lora_va, lora_vb = None, None

            if self.o_rank:
                lora_oa = self.lora_oa_gen(intermediate_embeddings, deterministic=deterministic)
                lora_oa = jnp.reshape(
                    lora_oa, (-1, total_layers, cfg.num_heads, cfg.head_dim, self.o_rank)
                )
                lora_ob = self.lora_ob_gen(intermediate_embeddings, deterministic=deterministic)
                lora_ob = jnp.reshape(lora_ob, (-1, total_layers, self.o_rank, cfg.emb_dim))
            else:
                lora_oa, lora_ob = None, None
        else:
            lora_qa, lora_qb, lora_ka, lora_kb, lora_va, lora_vb, lora_oa, lora_ob = (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        if self.use_prefix:
            # TODO: do we need 2 prefix keys and prefix values for self and cross attn respectively?
            #   If not, need to update embedder.
            #   Looks like we aren't using it yet?
            prefix_key = self.prefix_key_gen(intermediate_embeddings, deterministic=deterministic)
            prefix_key = jnp.reshape(
                prefix_key, (-1, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim)
            )
            prefix_value = self.prefix_value_gen(
                intermediate_embeddings, deterministic=deterministic
            )
            prefix_value = jnp.reshape(
                prefix_value, (-1, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim)
            )
        else:
            prefix_key = None
            prefix_value = None

        return {
            "lora_qa": lora_qa,
            "lora_qb": lora_qb,
            "lora_ka": lora_ka,
            "lora_kb": lora_kb,
            "lora_va": lora_va,
            "lora_vb": lora_vb,
            "lora_oa": lora_oa,
            "lora_ob": lora_ob,
            "prefix_key": prefix_key,
            "prefix_value": prefix_value,
            "prefix_key_cc": None,
            "prefix_value_cc": None,
        }


class LoraEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    config: HyperT5Config
    relative_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        inputs,
        lora_qa=None,
        lora_qb=None,
        lora_ka=None,
        lora_kb=None,
        lora_va=None,
        lora_vb=None,
        lora_oa=None,
        lora_ob=None,
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
        x = LoraMultiHeadDotProductAttentionWithPrefix(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="attention",
            hyper_gen=cfg.lora_hyper_gen,
            lora_ranks=cfg.lora_ranks,
        )(
            x,
            x,
            encoder_mask,
            encoder_bias,
            lora_qa=lora_qa,
            lora_qb=lora_qb,
            lora_ka=lora_ka,
            lora_kb=lora_kb,
            lora_va=lora_va,
            lora_vb=lora_vb,
            lora_oa=lora_oa,
            lora_ob=lora_ob,
            prefix_key=prefix_key,
            prefix_value=prefix_value,
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

        y = y + x
        return y


class LoraDecoderLayer(nn.Module):
    """Transformer decoder layer that attends to the encoder."""

    config: HyperT5Config
    relative_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        inputs,
        encoded,
        lora_qa=None,
        lora_qb=None,
        lora_ka=None,
        lora_kb=None,
        lora_va=None,
        lora_vb=None,
        lora_oa=None,
        lora_ob=None,
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
        q_rank, k_rank, v_rank, o_rank = (x and cfg.lora_hyper_gen for x in cfg.lora_ranks)

        # Relative position embedding as attention biases.
        l = max_decode_length if decode and max_decode_length else inputs.shape[-2]  # noqa: E741
        decoder_bias = self.relative_embedding(l, l, False)

        # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
        x = layers.LayerNorm(dtype=cfg.dtype, name="pre_self_attention_layer_norm")(inputs)

        # Self-attention block
        x = LoraMultiHeadDotProductAttentionWithPrefix(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="self_attention",
            hyper_gen=cfg.lora_hyper_gen,
            lora_ranks=cfg.lora_ranks,
        )(
            x,
            x,
            decoder_mask,
            decoder_bias,
            lora_qa=lora_qa[:, 0] if q_rank else None,
            lora_qb=lora_qb[:, 0] if q_rank else None,
            lora_ka=lora_ka[:, 0] if k_rank else None,
            lora_kb=lora_kb[:, 0] if k_rank else None,
            lora_va=lora_va[:, 0] if v_rank else None,
            lora_vb=lora_vb[:, 0] if v_rank else None,
            lora_oa=lora_oa[:, 0] if o_rank else None,
            lora_ob=lora_ob[:, 0] if o_rank else None,
            prefix_key=prefix_key if cfg.use_prefix else None,
            prefix_value=prefix_value if cfg.use_prefix else None,
            deterministic=deterministic,
            decode=decode,
        )
        x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(x, deterministic=deterministic)
        x = x + inputs

        # Encoder-Decoder block.
        y = layers.LayerNorm(dtype=cfg.dtype, name="pre_cross_attention_layer_norm")(x)
        y = LoraMultiHeadDotProductAttentionWithPrefix(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="encoder_decoder_attention",
            hyper_gen=cfg.lora_hyper_gen,
            lora_ranks=cfg.lora_ranks,
        )(
            y,
            encoded,
            encoder_decoder_mask,
            lora_qa=lora_qa[:, 1] if q_rank else None,
            lora_qb=lora_qb[:, 1] if q_rank else None,
            lora_ka=lora_ka[:, 1] if k_rank else None,
            lora_kb=lora_kb[:, 1] if k_rank else None,
            lora_va=lora_va[:, 1] if v_rank else None,
            lora_vb=lora_vb[:, 1] if v_rank else None,
            lora_oa=lora_oa[:, 1] if o_rank else None,
            lora_ob=lora_ob[:, 1] if o_rank else None,
            prefix_key=prefix_key_cc if cfg.use_prefix else None,
            prefix_value=prefix_value_cc if cfg.use_prefix else None,
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

        z = z + y
        return z


class LoraEncoder(nn.Module):
    """A stack of encoder layers."""

    config: HyperT5Config
    shared_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        encoder_input_tokens,
        lora_qa=None,
        lora_qb=None,
        lora_ka=None,
        lora_kb=None,
        lora_va=None,
        lora_vb=None,
        lora_oa=None,
        lora_ob=None,
        prefix_key=None,
        prefix_value=None,
        encoder_mask=None,
        deterministic=False,
    ):
        cfg = self.config
        q_rank, k_rank, v_rank, o_rank = (x and cfg.lora_hyper_gen for x in cfg.lora_ranks)

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
                lora_qa=lora_qa[:, lyr] if q_rank else None,
                lora_qb=lora_qb[:, lyr] if q_rank else None,
                lora_ka=lora_ka[:, lyr] if k_rank else None,
                lora_kb=lora_kb[:, lyr] if k_rank else None,
                lora_va=lora_va[:, lyr] if v_rank else None,
                lora_vb=lora_vb[:, lyr] if v_rank else None,
                lora_oa=lora_oa[:, lyr] if o_rank else None,
                lora_ob=lora_ob[:, lyr] if o_rank else None,
                prefix_key=prefix_key[:, lyr] if cfg.use_prefix else None,
                prefix_value=prefix_value[:, lyr] if cfg.use_prefix else None,
                encoder_mask=encoder_mask,
                deterministic=deterministic,
            )

        x = layers.LayerNorm(dtype=cfg.dtype, name="encoder_norm")(x)
        return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)


class LoraDecoder(nn.Module):
    config: HyperT5Config
    shared_embedding: nn.Module

    @nn.compact
    def __call__(
        self,
        encoded,
        decoder_input_tokens,
        lora_qa=None,
        lora_qb=None,
        lora_ka=None,
        lora_kb=None,
        lora_va=None,
        lora_vb=None,
        lora_oa=None,
        lora_ob=None,
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
        q_rank, k_rank, v_rank, o_rank = (x and cfg.lora_hyper_gen for x in cfg.lora_ranks)
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

        lyr_name = 0
        for lyr in range(
            cfg.num_encoder_layers, cfg.num_encoder_layers + 2 * cfg.num_decoder_layers, 2
        ):
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            y = LoraDecoderLayer(config=cfg, relative_embedding=rel_emb, name=f"layers_{lyr_name}")(
                y,
                encoded,
                decoder_mask=decoder_mask,
                lora_qa=lora_qa[:, lyr : lyr + 2] if q_rank else None,
                lora_qb=lora_qb[:, lyr : lyr + 2] if q_rank else None,
                lora_ka=lora_ka[:, lyr : lyr + 2] if k_rank else None,
                lora_kb=lora_kb[:, lyr : lyr + 2] if k_rank else None,
                lora_va=lora_va[:, lyr : lyr + 2] if v_rank else None,
                lora_vb=lora_vb[:, lyr : lyr + 2] if v_rank else None,
                lora_oa=lora_oa[:, lyr : lyr + 2] if o_rank else None,
                lora_ob=lora_ob[:, lyr : lyr + 2] if o_rank else None,
                prefix_key=prefix_key[:, lyr] if cfg.use_prefix else None,
                prefix_value=prefix_value[:, lyr] if cfg.use_prefix else None,
                prefix_key_cc=prefix_key_cc[:, lyr] if cfg.use_prefix else None,
                prefix_value_cc=prefix_value_cc[:, lyr] if cfg.use_prefix else None,
                encoder_decoder_mask=encoder_decoder_mask,
                deterministic=deterministic,
                decode=decode,
                max_decode_length=max_decode_length,
            )

            lyr_name += 1

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


class HyperLoraTransformer(HyperTransformer):
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

        self.hyper = HyperLoraNet(config=cfg, shared_embedding=self.shared_embedding)
        self.encoder = LoraEncoder(config=cfg, shared_embedding=self.shared_embedding)
        self.decoder = LoraDecoder(config=cfg, shared_embedding=self.shared_embedding)


class LoraTransformer(Transformer):
    """An encoder-decoder Transformer model with LoRA"""

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

        self.encoder = LoraEncoder(config=cfg, shared_embedding=self.shared_embedding)
        self.decoder = LoraDecoder(config=cfg, shared_embedding=self.shared_embedding)
