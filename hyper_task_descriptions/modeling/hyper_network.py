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
from transformers import FlaxT5EncoderModel
from typing_extensions import TypeAlias

from hyper_task_descriptions.modeling.layers import SimpleLinear
from hyper_task_descriptions.modeling.lora import (
    LoraMultiHeadDotProductAttentionWithPrefix,
)
from hyper_task_descriptions.modeling.hf_t5_enc import FlaxT5EncoderModuleSharedEmbedding
from t5x.examples.t5 import layers
from t5x.examples.t5.network import T5Config

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
    hyperencoder_model: str = "google/t5-large-lm-adapt"
    layer_embedding_method: str = "component"  # concat, layer, component
    use_instructions: bool = True  # if false, we use a single learnt embedding as input to hnet
    use_adapter: bool = True
    adapter_size: int = 64
    use_prefix: bool = True
    num_prefix_tokens: int = 30
    use_lora: bool = False
    lora_ranks: tuple = (None, None, None, None)
    use_instruction_embedding: bool = False  # for debugging. Use prompt-style embed for instruction.
    use_linear: bool = False
    use_soft_prompt: bool = False
    use_segment_embeds: bool = False


# create our component id dict
# since we create component-specific embeddings, we need to
# be able to keep track of which embedding for which component.
def create_component_id_dict(cfg: HyperT5Config):
    num_components = 0
    component_2_id = {}
    if cfg.use_adapter:
        num_components += 4
        component_2_id["adapter_wd"] = 0
        component_2_id["adapter_wu"] = 1
        component_2_id["adapter_bd"] = 2
        component_2_id["adapter_bu"] = 3
    if cfg.use_prefix:
        num_components += 2  # prefix key, value
        component_2_id["prefix_key"] = num_components - 2
        component_2_id["prefix_value"] = num_components - 1
    if cfg.use_lora:
        q_rank, k_rank, v_rank, o_rank = cfg.lora_ranks
        if q_rank is not None:
            num_components += 2  # q, k, v
            component_2_id["lora_qa"] = num_components - 2
            component_2_id["lora_qb"] = num_components - 1
        if k_rank is not None:
            num_components += 2
            component_2_id["lora_ka"] = num_components - 2
            component_2_id["lora_kb"] = num_components - 1
        if v_rank is not None:
            num_components += 2
            component_2_id["lora_va"] = num_components - 2
            component_2_id["lora_vb"] = num_components - 1
        if o_rank is not None:
            num_components += 2
            component_2_id["lora_oa"] = num_components - 2
            component_2_id["lora_ob"] = num_components - 1
    if num_components == 0:
        num_components += 1  # avoid div by zero error in init
    return num_components, component_2_id


class Hypernet(nn.Module):
    encoder: nn.Module
    config: HyperT5Config
    shared_embedding: nn.Module

    # we setup here as loading huggingface weights
    def setup(self):
        cfg = self.config
        # setup embeddings - enc attn, dec attn, cross attn
        self.num_components, self.component_2_id = create_component_id_dict(cfg)
        layer_embed_components = cfg.num_encoder_layers + (cfg.num_decoder_layers * 2)
        #encoder = FlaxT5EncoderModel.from_pretrained(cfg.hyperencoder_model, from_pt=True)
        self.henc_segment = jnp.asarray(
            param_with_axes(
                "henc_segment",
                nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                (16, cfg.emb_dim),
                jnp.float32,
                axes=("vocab", "embed"),
            ),
            jnp.float32,
        )

        self.attn = layers.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="hyperattn",
        )

        if cfg.layer_embedding_method == "component":
            layer_embed_components *= self.num_components
        self.embedder = jnp.asarray(
            param_with_axes(
                "component_embedding",
                nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                (layer_embed_components, cfg.emb_dim),
                jnp.float32,
                axes=("vocab", "embed"),
            ),
            jnp.float32,
        )
        if cfg.use_instructions:
            assert cfg.layer_embedding_method in [
                "concat",
                "layer",
                "component",
            ], "Invalid layer embedding method"
            # encodes the task description
            # self.encoder = FlaxT5EncoderModuleSharedEmbedding(
            #     config=encoder.config,
            #     shared_embedding=self.shared_embedding,
            #     dtype=cfg.dtype,
            # )

        if cfg.use_instruction_embedding:
            self.instruction_linear = SimpleLinear(
                cfg.emb_dim,
                act_fn="linear",
                dropout_rate=cfg.dropout_rate,
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                name="instruction_embed",
                # kernel_init=lambda _, shape, dtype: jnp.eye(shape[0], dtype=dtype),
            )
            self.inst_ln = layers.LayerNorm(dtype=cfg.dtype, name="instruction_embed_layernorm")

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
                name="prefix_value_mlp",
                dropout_rate=cfg.dropout_rate,
            )
        self.q_rank, self.k_rank, self.v_rank, self.o_rank = cfg.lora_ranks
        if cfg.use_lora:
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

    def __call__(self, encoder_input_tokens, deterministic=False):
        cfg = self.config
        bsz = encoder_input_tokens.shape[0]
        total_layers = cfg.num_encoder_layers + cfg.num_decoder_layers * 2
        if cfg.use_instructions:
            # 0 is t5 padding id.
            attn_mask = encoder_input_tokens != 0
            # get type issues otherwise so make sure tokens are ints.
            encoder_input_tokens = encoder_input_tokens.astype("i4")
            layer_out = self.encoder(encoder_input_tokens, deterministic=deterministic, hyper=True)
            output = layer_out
            #layer_out = layer_out.hidden_states
            # save pooled output for later (eg contrastive training)
            mean_seq = (output * attn_mask[:, :, None]).sum(axis=1) / attn_mask.sum(axis=1)[
                :, None
            ]
            self.sow("intermediates", "features", mean_seq)
            # we have encoder self attn, decoder self attn, decoder cross attn
            total_layers = cfg.num_encoder_layers + (cfg.num_decoder_layers * 2)
            # layer embedding setup
            if cfg.layer_embedding_method == "layer":
                seq_output = (output * attn_mask[:, :, None])  # to prevent padding annoying us.
                layer_embeds = self.embedder[None, :, :].repeat(
                    encoder_input_tokens.shape[0], axis=0
                )
                mask = layers.make_attention_mask(
                    jnp.ones((layer_embeds.shape[0], layer_embeds.shape[1])), encoder_input_tokens, dtype=cfg.dtype
                )
                sum_embeds = self.attn(layer_embeds, seq_output, mask=mask, deterministic=deterministic)

            elif cfg.layer_embedding_method == "component":
                seq_output = (output * attn_mask[:, :, None])  # to prevent padding annoying us.
                layer_embeds = self.embedder[None, :, :].repeat(
                    encoder_input_tokens.shape[0], axis=0
                )
                mask = layers.make_attention_mask(
                    jnp.ones((layer_embeds.shape[0], layer_embeds.shape[1])), encoder_input_tokens, dtype=cfg.dtype
                )
                sum_embeds = self.attn(layer_embeds, seq_output, mask=mask, deterministic=deterministic)
            else:  # else = use concat
                # layer embeds - repeat in batch, length dim
                sum_embeds = sum_embeds[:, None].repeat(total_layers, axis=1)
                layer_embs = self.embedder[
                    None,
                    :,
                ].repeat(sum_embeds.shape[0], axis=0)
                sum_embeds = jnp.concatenate([mean_seq, layer_embs], axis=-1)
        else:
            sum_embeds = self.embedder[None, :].repeat(encoder_input_tokens.shape[0], axis=0)
        # at this point, sum embeds should be [batch, layers, num_comp, feats]
        # (or at least reshape-able to it). Note num_comp = 1 for concat or layer methods.
        sum_embeds = sum_embeds.reshape(
            encoder_input_tokens.shape[0], total_layers, self.num_components, -1
        )

        generated_parameter_dict = {}

        # choose our specific input to the hypernet. feel free to customize.
        def generate_parameter(param_gen, inputs, component_id, shape):
            assert component_id in self.component_2_id, "component name not found"
            if cfg.layer_embedding_method == "component":
                inputs = inputs[:, :, self.component_2_id[component_id]]
            parameters = param_gen(inputs, deterministic=deterministic)
            parameters = parameters.reshape(shape) / jnp.sqrt(inputs.shape[-1])
            return parameters

        if cfg.use_instruction_embedding:
            layer_embeds = [o * attn_mask[:, :, None] for o in layer_out]
            instruction_embed = (output * attn_mask[:, :, None])
            if cfg.use_segment_embeds:
                instruction_embed = self.henc_segment[None,None,0] + instruction_embed
            if cfg.use_linear:
                instruction_embed = self.instruction_linear(instruction_embed, deterministic=deterministic)
                instruction_embed = instruction_embed / jnp.sqrt(instruction_embed.shape[-1])
                #instruction_embed = self.inst_ln(instruction_embed)
            
            generated_parameter_dict["instruction_embedding"] = instruction_embed
            generated_parameter_dict["instruction_embedding_layers"] = instruction_embed

        if cfg.use_adapter:
            # adapter weight down
            generated_parameter_dict["adapter_wd"] = generate_parameter(
                self.adapter_down_gen,
                sum_embeds,
                "adapter_wd",
                (bsz, total_layers, cfg.emb_dim, cfg.adapter_size),
            )
            # adapter weight up
            generated_parameter_dict["adapter_wu"] = generate_parameter(
                self.adapter_up_gen,
                sum_embeds,
                "adapter_wu",
                (bsz, total_layers, cfg.adapter_size, cfg.emb_dim),
            )
            # adapter bias down
            generated_parameter_dict["adapter_bd"] = generate_parameter(
                self.adapter_bias_down_gen,
                sum_embeds,
                "adapter_bd",
                (bsz, total_layers, cfg.adapter_size),
            )
            # adapter bias up
            generated_parameter_dict["adapter_bu"] = generate_parameter(
                self.adapter_bias_up_gen, sum_embeds, "adapter_bu", (-1, total_layers, cfg.emb_dim)
            )
        if cfg.use_prefix:
            # prefix key
            generated_parameter_dict["prefix_key"] = generate_parameter(
                self.prefix_key_gen,
                sum_embeds,
                "prefix_key",
                (bsz, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim),
            )
            # prefix value
            generated_parameter_dict["prefix_value"] = generate_parameter(
                self.prefix_value_gen,
                sum_embeds,
                "prefix_value",
                (bsz, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim),
            )
        if cfg.use_lora:
            if self.q_rank:
                generated_parameter_dict["lora_qa"] = generate_parameter(
                    self.lora_qa_gen,
                    sum_embeds,
                    "lora_qa",
                    (bsz, total_layers, cfg.emb_dim, self.q_rank),
                )
                generated_parameter_dict["lora_qb"] = generate_parameter(
                    self.lora_qb_gen,
                    sum_embeds,
                    "lora_qb",
                    (bsz, total_layers, self.q_rank, cfg.num_heads, cfg.head_dim),
                )
            if self.k_rank:
                generated_parameter_dict["lora_ka"] = generate_parameter(
                    self.lora_ka_gen,
                    sum_embeds,
                    "lora_ka",
                    (bsz, total_layers, cfg.emb_dim, self.k_rank),
                )
                generated_parameter_dict["lora_kb"] = generate_parameter(
                    self.lora_kb_gen,
                    sum_embeds,
                    "lora_kb",
                    (bsz, total_layers, self.k_rank, cfg.num_heads, cfg.head_dim),
                )
            if self.v_rank:
                generated_parameter_dict["lora_va"] = generate_parameter(
                    self.lora_va_gen,
                    sum_embeds,
                    "lora_va",
                    (bsz, total_layers, cfg.emb_dim, self.v_rank),
                )
                generated_parameter_dict["lora_vb"] = generate_parameter(
                    self.lora_vb_gen,
                    sum_embeds,
                    "lora_vb",
                    (bsz, total_layers, self.v_rank, cfg.num_heads, cfg.head_dim),
                )
            if self.o_rank:
                generated_parameter_dict["lora_oa"] = generate_parameter(
                    self.lora_oa_gen,
                    sum_embeds,
                    "lora_oa",
                    (bsz, total_layers, cfg.num_heads, cfg.head_dim, self.o_rank),
                )
                generated_parameter_dict["lora_ob"] = generate_parameter(
                    self.lora_ob_gen,
                    sum_embeds,
                    "lora_ob",
                    (bsz, total_layers, self.o_rank, cfg.emb_dim),
                )

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
        lora_qa=None,
        lora_qb=None,
        lora_ka=None,
        lora_kb=None,
        lora_va=None,
        lora_vb=None,
        lora_oa=None,
        lora_ob=None,
        encoder_mask=None,
        deterministic=False,
        hyper=False,
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
            use_prefix=cfg.use_prefix,
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
            use_prefix=not hyper,
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
        # adapter block
        if cfg.use_adapter and not hyper:
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
        lora_qa=None,
        lora_qb=None,
        lora_ka=None,
        lora_kb=None,
        lora_va=None,
        lora_vb=None,
        lora_oa=None,
        lora_ob=None,
        decoder_mask=None,
        encoder_decoder_mask=None,
        deterministic=False,
        decode=False,
        max_decode_length=None,
    ):
        cfg = self.config
        q_rank, k_rank, v_rank, o_rank = (x and cfg.use_lora for x in cfg.lora_ranks)

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
            use_prefix=cfg.use_prefix,
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
            prefix_key=prefix_key[:, 0] if cfg.use_prefix else None,
            prefix_value=prefix_value[:, 0] if cfg.use_prefix else None,
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
            use_prefix=cfg.use_prefix,
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
            prefix_key=prefix_key[:, 1] if cfg.use_prefix else None,
            prefix_value=prefix_value[:, 1] if cfg.use_prefix else None,
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
        adaptations={},
        encoder_mask=None,
        deterministic=False,
        hyper=False,
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

        # hyper_input_tokens = adaptations['hyper_encoder_input_tokens']
        # encoder_input_tokens = jnp.concatenate([hyper_input_tokens, encoder_input_tokens], axis=1)

        # [batch, length] -> [batch, length, emb_dim]
        x = self.shared_embedding(encoder_input_tokens.astype("int32"))
        x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(x, deterministic=deterministic)
        x = x.astype(cfg.dtype)

        # concat. currently not using mask cor. but thats ok
        if cfg.use_instruction_embedding and not hyper:
            embed = adaptations.pop('instruction_embedding')
            encoder_tokens = jnp.concatenate(
                [adaptations.pop('hyper_encoder_input_tokens'), encoder_input_tokens],
                axis=1)
            lyr_encoder_mask = layers.make_attention_mask(
                encoder_tokens > 0, encoder_tokens > 0, dtype=cfg.dtype
            )
            instruction_embeds = adaptations.pop('instruction_embedding_layers')

        for lyr in range(cfg.num_encoder_layers):
            layer_adaptations = {k: v[:, lyr] for k, v in adaptations.items()}
            # if cfg.use_instruction_embedding and not hyper:
            #     x = jnp.concatenate([embed, x], axis=1)
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            x = HyperEncoderLayer(config=cfg, relative_embedding=rel_emb, name=f"layers_{lyr}")(
                x,
                **layer_adaptations,
                encoder_mask=encoder_mask,
                deterministic=deterministic,
                hyper=hyper,
            )
            # if cfg.use_instruction_embedding and not hyper:
            #     x = x[:, embed.shape[1]:]

        enc_segment = jnp.asarray(
            param_with_axes(
                "enc_segment",
                nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                (16, cfg.emb_dim),
                jnp.float32,
                axes=("vocab", "embed"),
            ),
            jnp.float32,
        )

        if cfg.use_segment_embeds and not hyper:
            x = x + enc_segment[None,None,0]

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
        adaptations={},
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

        for lyr in range(
            cfg.num_encoder_layers, cfg.num_encoder_layers + cfg.num_decoder_layers * 2, 2
        ):
            # [batch, length, emb_dim] -> [batch, length, emb_dim]

            # grab adaptations - note that adapters only need one as no c-a to worry abt
            layer_adaptations = {
                k: v[:, lyr : lyr + 2] for k, v in adaptations.items() if "adapter" not in k
            }
            layer_adaptations_ada = {k: v[:, lyr] for k, v in adaptations.items() if "adapter" in k}
            # I would use |=, but maintaining compat with older python
            layer_adaptations = {**layer_adaptations, **layer_adaptations_ada}
            lyr_name = (
                lyr - cfg.num_encoder_layers
            ) // 2  # to maintain rng equivalence with original code
            y = HyperDecoderLayer(
                config=cfg, relative_embedding=rel_emb, name=f"layers_{lyr_name}"
            )(
                y,
                encoded,
                decoder_mask=decoder_mask,
                **layer_adaptations,
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
        # set some things correctly
        if not cfg.use_lora:
            assert cfg.lora_ranks == (
                None,
                None,
                None,
                None,
            ), "lora_ranks must be None if not using lora"

        self.encoder = HyperEncoder(config=cfg, shared_embedding=self.shared_embedding)
        self.decoder = HyperDecoder(config=cfg, shared_embedding=self.shared_embedding)
        self.hyper = Hypernet(encoder=self.encoder, config=cfg, shared_embedding=self.shared_embedding)

    def encode(
        self,
        encoder_input_tokens,
        adaptations={},
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
            adaptations=adaptations,
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
        adaptations={},
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
            adaptations=adaptations,
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
        adaptations = self.hyperencode(hyper_encoder_input_tokens, enable_dropout=enable_dropout)
        if self.config.use_instruction_embedding:
            instruction_embedding = adaptations["instruction_embedding"]
            adaptations['hyper_encoder_input_tokens'] = hyper_encoder_input_tokens
        encoded = self.encode(
            encoder_input_tokens,
            adaptations=adaptations,
            encoder_segment_ids=encoder_segment_ids,
            enable_dropout=enable_dropout,
        )
        # we re-insert instruction embedding here
        if self.config.use_instruction_embedding:
            encoded = jnp.concatenate(
                [instruction_embedding, encoded], axis=1
            )
            encoder_input_tokens = jnp.concatenate(
                [hyper_encoder_input_tokens, encoder_input_tokens], axis=1
            )
        return self.decode(
            encoded,
            encoder_input_tokens,  # only used for masks
            decoder_input_tokens,
            decoder_target_tokens,
            adaptations=adaptations,
            encoder_segment_ids=encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            enable_dropout=enable_dropout,
            decode=decode,
        )
