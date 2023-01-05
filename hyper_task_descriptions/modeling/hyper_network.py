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
from typing import Callable, Iterable, Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen import partitioning as nn_partitioning
from jax import lax
from t5x.examples.t5 import layers
from t5x.examples.t5.network import T5Config
from typing_extensions import TypeAlias

from hyper_task_descriptions.modeling.layers import MlpBlock, SimpleLinear
from hyper_task_descriptions.modeling.lora import (
    LoraMultiHeadDotProductAttentionWithPrefix,
)

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
    hypernet_activations: Tuple[
        str,
    ] = ("gelu",)
    use_instructions: bool = True  # if false, we use a single learnt embedding as input to hnet
    use_adapter: bool = True
    adapter_size: int = 64
    use_prompt: bool = False
    num_prompt_tokens: int = 100
    use_prefix: bool = True
    num_prefix_tokens: int = 30
    use_lora: bool = False
    lora_ranks: tuple = (None, None, None, None)
    use_fusion_in_decoder: bool = False  # enables fid
    use_linear: bool = False  # linear transform on top of fid. required for mismatched models.
    hnet_layernorm: bool = False
    per_layer_hnet: bool = False
    share_hnet_encoder: bool = True
    use_tanh_prefix: bool = False


# create our component id dict
# since we create component-specific embeddings, we need to
# be able to keep track of which embedding for which component.
# 'component id' = the range of component ids for a given component
# this is because we used to have a single embedding for each component
def create_component_id_dict(cfg: HyperT5Config):
    num_components = 0
    component_2_id = {}
    if cfg.use_adapter:
        num_components += cfg.adapter_size * 2 + 2
        component_2_id["adapter_wd"] = (0, cfg.adapter_size)
        component_2_id["adapter_wu"] = (cfg.adapter_size, cfg.adapter_size * 2)
        component_2_id["adapter_bd"] = (cfg.adapter_size * 2, cfg.adapter_size * 2 + 1)
        component_2_id["adapter_bu"] = (cfg.adapter_size * 2 + 1, cfg.adapter_size * 2 + 2)
    if cfg.use_prefix:
        component_2_id["prefix_key"] = (num_components, num_components + cfg.num_prefix_tokens)
        component_2_id["prefix_value"] = (
            num_components + cfg.num_prefix_tokens,
            num_components + cfg.num_prefix_tokens * 2,
        )
        num_components += 2 * cfg.num_prefix_tokens  # prefix key, value
    if cfg.use_prompt:
        component_2_id["prompt"] = (num_components, num_components + cfg.num_prompt_tokens)
        num_components += cfg.num_prompt_tokens
    if cfg.use_lora:  # TODO: fix lora.
        q_rank, k_rank, v_rank, o_rank = cfg.lora_ranks
        if q_rank is not None:
            component_2_id["lora_qa"] = (num_components, num_components + q_rank)
            component_2_id["lora_qb"] = (num_components + q_rank, num_components + q_rank * 2)
            num_components += q_rank * 2
        if k_rank is not None:
            component_2_id["lora_ka"] = (num_components, num_components + k_rank)
            component_2_id["lora_kb"] = (num_components + k_rank, num_components + k_rank * 2)
            num_components += k_rank * 2
        if v_rank is not None:
            component_2_id["lora_va"] = (num_components, num_components + v_rank)
            component_2_id["lora_vb"] = (num_components + v_rank, num_components + v_rank * 2)
            num_components += v_rank * 2
        if o_rank is not None:
            component_2_id["lora_oa"] = (num_components, num_components + o_rank)
            component_2_id["lora_ob"] = (num_components + o_rank, num_components + o_rank * 2)
            num_components += o_rank * 2
    if num_components == 0:
        num_components += 1  # avoid div by zero error in init
    return num_components, component_2_id


class Hypernet(nn.Module):
    underlying_encoder: nn.Module
    underlying_decoder: nn.Module
    config: HyperT5Config
    shared_embedding: nn.Module

    # we setup here as loading huggingface weights
    def setup(self):
        cfg = self.config
        # setup embeddings - enc attn, dec attn, cross attn
        self.num_components, self.component_2_id = create_component_id_dict(cfg)
        layer_embed_components = cfg.num_encoder_layers + (cfg.num_decoder_layers * 2)
        # here, we use the hypernet for generate per-layer values
        if cfg.per_layer_hnet:
            layer_embed_components = 1

        self.attn = layers.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate,
            float32_logits=cfg.float32_attention_logits,
            name="hyperattn",
        )
        if not cfg.use_instructions:
            self.num_components = 16

        if cfg.share_hnet_encoder:
            self.encoder = self.underlying_encoder
        else:
            self.encoder = HyperEncoder(cfg, self.shared_embedding, name="hyper_encoder")

        if cfg.layer_embedding_method == "decoder":
            self.decoder = HyperDecoder(cfg, self.shared_embedding, name="hyper_decoder")
        else:
            self.decoder = self.underlying_decoder

        if cfg.layer_embedding_method == "component" or cfg.layer_embedding_method == "decoder":
            layer_embed_components *= self.num_components
        self.true_layer_embed = layer_embed_components
        if cfg.layer_embedding_method == "decoder":
            while layer_embed_components % 16 != 0:
                layer_embed_components += 1
        
        # to make sure compat with partitioning.
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
                "decoder",
            ], "Invalid layer embedding method"

        if cfg.use_fusion_in_decoder:
            self.instruction_linear = SimpleLinear(
                cfg.emb_dim,
                act_fn="linear",
                dropout_rate=cfg.dropout_rate,
                dtype=cfg.dtype,
                kernel_axes=("mlp", "embed"),
                name="instruction_embed",
            )

        def hypernetwork(output, name, activations=cfg.hypernet_activations):
            if cfg.per_layer_hnet:
                output *= cfg.num_encoder_layers + 2 * cfg.num_decoder_layers
            return MlpBlock(
                intermediate_dim=cfg.emb_dim,  # same size as model
                output_dim=output,
                activations=cfg.hypernet_activations,
                intermediate_dropout_rate=0.0,
                dtype=cfg.dtype,
                name=name,
            )

        if cfg.use_adapter:
            output_dim = cfg.emb_dim
            self.adapter_down_norm = layers.LayerNorm(name="adapter_down_norm")
            self.adapter_down_gen = hypernetwork(output_dim, "adapter_down")
            self.adapter_up_norm = layers.LayerNorm(name="adapter_up_norm")
            self.adapter_up_gen = hypernetwork(output_dim, "adapter_up")
            self.adapter_bias_down_norm = layers.LayerNorm(name="adapter_bias_down_norm")
            self.adapter_bias_down_gen = hypernetwork(cfg.adapter_size, "adapter_bias_down")
            self.adapter_bias_up_norm = layers.LayerNorm(name="adapter_bias_up_norm")
            self.adapter_bias_up_gen = hypernetwork(cfg.emb_dim, "adapter_bias_up")
        if cfg.use_prefix:
            if cfg.use_tanh_prefix:
                activations = ("tanh",)
            else:
                activations = cfg.hypernet_activations
            output_dim = cfg.num_heads * cfg.head_dim
            self.prefix_key_norm = layers.LayerNorm(name="prefix_key_norm")
            self.prefix_key_gen = hypernetwork(output_dim, "prefix_key", activations=activations)
            self.prefix_value_norm = layers.LayerNorm(name="prefix_value_norm")
            self.prefix_value_gen = hypernetwork(output_dim, "prefix_value", activations=activations)
        if cfg.use_prompt:
            self.prompt_norm = layers.LayerNorm(name="prompt_norm")
            self.prompt_gen = hypernetwork(cfg.emb_dim, "prompt")
        self.q_rank, self.k_rank, self.v_rank, self.o_rank = cfg.lora_ranks
        if cfg.use_lora:
            if self.q_rank:
                self.lora_qa_norm = layers.LayerNorm(name="lora_qa_norm")
                self.lora_qa_gen = hypernetwork(cfg.emb_dim, "lora_qa")
                self.lora_qb_norm = layers.LayerNorm(name="lora_qb_norm")
                self.lora_qb_gen = hypernetwork(cfg.num_heads * cfg.head_dim, "lora_qb")
            if self.k_rank:
                self.lora_ka_norm = layers.LayerNorm(name="lora_ka_norm")
                self.lora_ka_gen = hypernetwork(cfg.emb_dim, "lora_ka")
                self.lora_kb_norm = layers.LayerNorm(name="lora_kb_norm")
                self.lora_kb_gen = hypernetwork(cfg.num_heads * cfg.head_dim, "lora_kb")
            if self.v_rank:
                self.lora_va_norm = layers.LayerNorm(name="lora_va_norm")
                self.lora_va_gen = hypernetwork(cfg.emb_dim, "lora_va")
                self.lora_vb_norm = layers.LayerNorm(name="lora_vb_norm")
                self.lora_vb_gen = hypernetwork(cfg.num_heads * cfg.head_dim, "lora_vb")
            if self.o_rank:
                self.lora_oa_norm = layers.LayerNorm(name="lora_oa_norm")
                self.lora_oa_gen = hypernetwork(cfg.emb_dim, "lora_oa")
                self.lora_ob_norm = layers.LayerNorm(name="lora_ob_norm")
                self.lora_ob_gen = hypernetwork(cfg.head_dim, "lora_ob")

    def __call__(self, encoder_input_tokens, deterministic=False):
        cfg = self.config
        bsz = encoder_input_tokens.shape[0]
        total_layers = cfg.num_encoder_layers + cfg.num_decoder_layers * 2
        if cfg.use_instructions:
            # 0 is t5 padding id.
            attn_mask = encoder_input_tokens != 0
            # get type issues otherwise so make sure tokens are ints.
            encoder_input_tokens = encoder_input_tokens.astype("i4")
            output = self.encoder(encoder_input_tokens, deterministic=deterministic, hyper=True)
            # save pooled output for later (eg contrastive training)
            mean_seq = (output * attn_mask[:, :, None]).sum(axis=1) / attn_mask.sum(axis=1)[:, None]
            self.sow("intermediates", "features", mean_seq)
            # we have encoder self attn, decoder self attn, decoder cross attn
            total_layers = cfg.num_encoder_layers + (cfg.num_decoder_layers * 2)
            # layer embedding setup
            if cfg.layer_embedding_method == "layer":
                seq_output = output * attn_mask[:, :, None]  # to prevent padding annoying us.
                layer_embeds = self.embedder[None, :, :].repeat(
                    encoder_input_tokens.shape[0], axis=0
                )
                mask = layers.make_attention_mask(
                    jnp.ones((layer_embeds.shape[0], layer_embeds.shape[1])),
                    encoder_input_tokens,
                    dtype=cfg.dtype,
                )
                sum_embeds = self.attn(
                    layer_embeds, seq_output, mask=mask, deterministic=deterministic
                )

            elif cfg.layer_embedding_method == "component":
                seq_output = output * attn_mask[:, :, None]  # to prevent padding annoying us.
                layer_embeds = self.embedder[None, :, :].repeat(
                    encoder_input_tokens.shape[0], axis=0
                )
                mask = layers.make_attention_mask(
                    jnp.ones((layer_embeds.shape[0], layer_embeds.shape[1])),
                    encoder_input_tokens,
                    dtype=cfg.dtype,
                )
                sum_embeds = self.attn(
                    layer_embeds, seq_output, mask=mask, deterministic=deterministic
                )
            elif cfg.layer_embedding_method == "decoder":
                seq_output = output * attn_mask[:, :, None]
                layer_embeds = self.embedder[None, :self.true_layer_embed, :].repeat(
                    encoder_input_tokens.shape[0], axis=0
                )
                enc_dec_mask = layers.make_attention_mask(
                    jnp.ones((layer_embeds.shape[0], layer_embeds.shape[1])),
                    encoder_input_tokens,
                    dtype=cfg.dtype,
                )
                sum_embeds = self.decoder(
                    seq_output,
                    hyper=True,
                    hyper_embeds=layer_embeds,
                    decoder_input_tokens=None,
                    decoder_mask=None,  # we allow the decoder to see the whole sequence.
                    encoder_decoder_mask=enc_dec_mask,
                    deterministic=deterministic,
                    decode=False,
                )
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
        if cfg.per_layer_hnet:
            sum_embeds_layers = 1
        else:
            sum_embeds_layers = total_layers
        sum_embeds = sum_embeds.reshape(
            encoder_input_tokens.shape[0], sum_embeds_layers, self.num_components, -1
        )

        generated_parameter_dict = {}

        # choose our specific input to the hypernet. feel free to customize.
        def generate_parameter(param_gen, layer_norm, inputs, component_id, shape):
            assert component_id in self.component_2_id, "component name not found"
            if cfg.layer_embedding_method == "component" or cfg.layer_embedding_method == "decoder":
                start, end = self.component_2_id[component_id]
                # reshape to collapse the components into one blob
                inputs = inputs[:, :, start:end]
            # layernorm for hypertune
            if cfg.layer_embedding_method == "decoder" or cfg.hnet_layernorm:
                inputs = layer_norm(inputs)
            parameters = param_gen(inputs, deterministic=deterministic)
            parameters = parameters.reshape(shape) / jnp.sqrt(inputs.shape[-1])
            return parameters

        if cfg.use_fusion_in_decoder:
            instruction_embed = output * attn_mask[:, :, None]
            if cfg.use_linear:
                instruction_embed = self.instruction_linear(
                    instruction_embed, deterministic=deterministic
                )
                instruction_embed = instruction_embed / jnp.sqrt(instruction_embed.shape[-1])

            generated_parameter_dict["instruction_embedding"] = instruction_embed

        if cfg.use_adapter:
            # adapter weight down
            generated_parameter_dict["adapter_wd"] = generate_parameter(
                self.adapter_down_gen,
                self.adapter_down_norm,
                sum_embeds,
                "adapter_wd",
                (bsz, total_layers, cfg.emb_dim, cfg.adapter_size),
            )
            # adapter weight up
            generated_parameter_dict["adapter_wu"] = generate_parameter(
                self.adapter_up_gen,
                self.adapter_up_norm,
                sum_embeds,
                "adapter_wu",
                (bsz, total_layers, cfg.adapter_size, cfg.emb_dim),
            )
            # adapter bias down
            generated_parameter_dict["adapter_bd"] = generate_parameter(
                self.adapter_bias_down_gen,
                self.adapter_bias_down_norm,
                sum_embeds,
                "adapter_bd",
                (bsz, total_layers, cfg.adapter_size),
            )
            # adapter bias up
            generated_parameter_dict["adapter_bu"] = generate_parameter(
                self.adapter_bias_up_gen,
                self.adapter_bias_up_norm,
                sum_embeds,
                "adapter_bu",
                (-1, total_layers, cfg.emb_dim),
            )
        if cfg.use_prefix:
            # prefix key
            generated_parameter_dict["prefix_key"] = generate_parameter(
                self.prefix_key_gen,
                self.prefix_key_norm,
                sum_embeds,
                "prefix_key",
                (bsz, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim),
            )
            # prefix value
            generated_parameter_dict["prefix_value"] = generate_parameter(
                self.prefix_value_gen,
                self.prefix_value_norm,
                sum_embeds,
                "prefix_value",
                (bsz, total_layers, cfg.num_prefix_tokens, cfg.num_heads, cfg.head_dim),
            )
        if cfg.use_prompt:
            generated_parameter_dict["prompt"] = generate_parameter(
                self.prompt_gen,
                self.prompt_norm,
                sum_embeds,
                "prompt",
                (bsz, total_layers, cfg.num_prompt_tokens, cfg.emb_dim),
            )[
                :, 0
            ]  # no layers since prompt is l1 only
        if cfg.use_lora:
            if self.q_rank:
                generated_parameter_dict["lora_qa"] = generate_parameter(
                    self.lora_qa_gen,
                    self.lora_qa_norm,
                    sum_embeds,
                    "lora_qa",
                    (bsz, total_layers, cfg.emb_dim, self.q_rank),
                )
                generated_parameter_dict["lora_qb"] = generate_parameter(
                    self.lora_qb_gen,
                    self.lora_qb_norm,
                    sum_embeds,
                    "lora_qb",
                    (bsz, total_layers, self.q_rank, cfg.num_heads, cfg.head_dim),
                )
            if self.k_rank:
                generated_parameter_dict["lora_ka"] = generate_parameter(
                    self.lora_ka_gen,
                    self.lora_ka_norm,
                    sum_embeds,
                    "lora_ka",
                    (bsz, total_layers, cfg.emb_dim, self.k_rank),
                )
                generated_parameter_dict["lora_kb"] = generate_parameter(
                    self.lora_kb_gen,
                    self.lora_kb_norm,
                    sum_embeds,
                    "lora_kb",
                    (bsz, total_layers, self.k_rank, cfg.num_heads, cfg.head_dim),
                )
            if self.v_rank:
                generated_parameter_dict["lora_va"] = generate_parameter(
                    self.lora_va_gen,
                    self.lora_va_norm,
                    sum_embeds,
                    "lora_va",
                    (bsz, total_layers, cfg.emb_dim, self.v_rank),
                )
                generated_parameter_dict["lora_vb"] = generate_parameter(
                    self.lora_vb_gen,
                    self.lora_vb_norm,
                    sum_embeds,
                    "lora_vb",
                    (bsz, total_layers, self.v_rank, cfg.num_heads, cfg.head_dim),
                )
            if self.o_rank:
                generated_parameter_dict["lora_oa"] = generate_parameter(
                    self.lora_oa_gen,
                    self.lora_oa_norm,
                    sum_embeds,
                    "lora_oa",
                    (bsz, total_layers, cfg.num_heads, cfg.head_dim, self.o_rank),
                )
                generated_parameter_dict["lora_ob"] = generate_parameter(
                    self.lora_ob_gen,
                    self.lora_ob_norm,
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
            use_prefix=cfg.use_prefix,
            use_gen=not hyper,
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
        hyper=False,
    ):
        cfg = self.config
        q_rank, k_rank, v_rank, o_rank = (x and cfg.use_lora for x in cfg.lora_ranks)

        # Relative position embedding as attention biases.
        l = max_decode_length if decode and max_decode_length else inputs.shape[-2]  # noqa: E741
        decoder_bias = self.relative_embedding(l, l, False)

        # no self-attention in hyperdecoder (costly)
        if not hyper:
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
                prefix_key=prefix_key[:, 0] if cfg.use_prefix and prefix_key is not None else None,
                prefix_value=prefix_value[:, 0]
                if cfg.use_prefix and prefix_value is not None
                else None,
                deterministic=deterministic,
                decode=decode,
                use_prefix=cfg.use_prefix,
                use_gen=not hyper,
            )
            x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
                x, deterministic=deterministic
            )
            x = x + inputs
        else:
            x = inputs

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
            prefix_key=prefix_key[:, 1] if cfg.use_prefix and prefix_key is not None else None,
            prefix_value=prefix_value[:, 1]
            if cfg.use_prefix and prefix_value is not None
            else None,
            deterministic=deterministic,
            use_prefix=cfg.use_prefix,
            use_gen=not hyper,
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
        if cfg.use_adapter and not hyper:
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

        # [batch, length] -> [batch, length, emb_dim]
        x = self.shared_embedding(encoder_input_tokens.astype("int32"))
        x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(x, deterministic=deterministic)
        x = x.astype(cfg.dtype)

        # append prompt
        if cfg.use_prompt and not hyper:
            prompt = adaptations["prompt"]
            x = jnp.concatenate([prompt, x], axis=1)
            bsz = x.shape[0]
            encoder_input_tokens = jnp.concatenate(
                [jnp.ones((bsz, prompt.shape[1]), dtype=cfg.dtype), encoder_input_tokens], axis=1
            )
            encoder_mask = layers.make_attention_mask(
                encoder_input_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype
            )

        for lyr in range(cfg.num_encoder_layers):
            layer_adaptations = {k: v[:, lyr] for k, v in adaptations.items() if "prompt" not in k}
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            x = HyperEncoderLayer(config=cfg, relative_embedding=rel_emb, name=f"layers_{lyr}")(
                x,
                **layer_adaptations,
                encoder_mask=encoder_mask,
                deterministic=deterministic,
                hyper=hyper,
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
        adaptations={},
        decoder_positions=None,
        decoder_mask=None,
        encoder_decoder_mask=None,
        deterministic=False,
        decode=False,
        max_decode_length=None,
        hyper=False,
        hyper_embeds=None,
    ):
        cfg = self.config
        if not hyper:
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
        if not hyper:
            y = self.shared_embedding(decoder_input_tokens.astype("int32"))
            y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
                y, deterministic=deterministic
            )
            y = y.astype(cfg.dtype)
        else:
            y = hyper_embeds

        for lyr in range(
            cfg.num_encoder_layers, cfg.num_encoder_layers + cfg.num_decoder_layers * 2, 2
        ):
            # [batch, length, emb_dim] -> [batch, length, emb_dim]

            # grab adaptations - note that adapters only need one as no c-a to worry abt
            layer_adaptations = {
                k: v[:, lyr : lyr + 2]
                for k, v in adaptations.items()
                if "adapter" not in k and "prompt" not in k
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
                hyper=hyper,
            )

        yd = layers.LayerNorm(dtype=cfg.dtype, name="decoder_norm")(y)
        y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(yd, deterministic=deterministic)
        if hyper:
            return y

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
        self.hyper = Hypernet(
            underlying_encoder=self.encoder,
            underlying_decoder=self.decoder,
            config=cfg,
            shared_embedding=self.shared_embedding,
        )

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

        if cfg.use_prompt:
            prompt = adaptations["prompt"]
            bsz = encoded.shape[0]
            encoder_input_tokens = jnp.concatenate(
                [jnp.ones((bsz, prompt.shape[1]), dtype=cfg.dtype), encoder_input_tokens], axis=1
            )

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
        if self.config.use_fusion_in_decoder:
            instruction_embedding = adaptations.pop("instruction_embedding")
            # adaptations["hyper_encoder_input_tokens"] = hyper_encoder_input_tokens
        encoded = self.encode(
            encoder_input_tokens,
            adaptations=adaptations,
            encoder_segment_ids=encoder_segment_ids,
            enable_dropout=enable_dropout,
        )
        # we re-insert instruction embedding here
        if self.config.use_fusion_in_decoder:
            encoded = jnp.concatenate([instruction_embedding, encoded], axis=1)
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
