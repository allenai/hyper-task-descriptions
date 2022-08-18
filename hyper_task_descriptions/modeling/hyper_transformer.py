"""
Define wrapper class for three-input model.
Required so we can have different underlying encoder and hypernet inputs.
This is adapted from the EncoderDecoderModel class in t5x.
"""
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import clu.metrics as clu_metrics
import jax
import jax.numpy as jnp
import numpy as np
import seqio
import tensorflow as tf
from flax import linen as nn
from flax.core import scope as flax_scope
from flax.core.frozen_dict import freeze, unfreeze
from seqio import FeatureConverter, non_padding_position, utils
from t5x import decoding, losses
from t5x import metrics as metrics_lib
from t5x import optimizers
from t5x.models import DecodeFnCallable, EncoderDecoderModel, compute_base_metrics
from t5x.utils import override_params_axes_names
from transformers import FlaxRobertaModel
from typing_extensions import TypeAlias

from hyper_task_descriptions.modeling.lora_partitioning import lora_axes_names_override
from hyper_task_descriptions.modeling.losses import cosine_similarity_loss
from hyper_task_descriptions.modeling.roberta_partitioning import (
    roberta_axes_names_override,
)

Array: TypeAlias = Union[np.ndarray, jnp.ndarray, jax.pxla.ShardedDeviceArray, tf.Tensor]
if TYPE_CHECKING:
    PyTreeDef = Any
else:
    PyTreeDef = type(jax.tree_structure(None))


def trim_and_pad(
    k: str,
    t: tf.Tensor,
    task_feature_lengths: Mapping[str, int],
    task_feature_paddings: Mapping[str, int],
) -> tf.Tensor:
    """
    Trim/pad to the first axis of `t` to be of size `length`.
    fixed version from seqio that allows changing pad value.
    """
    if k not in task_feature_lengths:
        return t
    length_k = task_feature_lengths[k]
    t = t[:length_k]
    pad_amt = length_k - tf.shape(t)[0]
    padded_t = tf.pad(
        t, [(0, pad_amt)] + [(0, 0)] * (len(t.shape) - 1), constant_values=task_feature_paddings[k]
    )
    padded_t.set_shape([length_k] + t.shape.as_list()[1:])
    return padded_t


class HyperEncDecFeatureConverter(FeatureConverter):
    """Feature converter for an encoder-decoder with hypernet architecture.
    Really this is just providing a second encoder input to the model.
    """

    TASK_FEATURES = {
        "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "hyper_inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
    }
    MODEL_FEATURES = {
        "encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "hyper_encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
    }
    # t5x by default pads everything with the token id 0, but for roberta it is the token id 1
    TASK_PADDING = {
        "inputs": 0,
        "hyper_inputs": 1,
        "targets": 0,
    }
    PACKING_FEATURE_DTYPES = {
        "encoder_segment_ids": tf.int32,
        "hyper_encoder_segment_ids": tf.int32,
        "decoder_segment_ids": tf.int32,
        "encoder_positions": tf.int32,
        "decoder_positions": tf.int32,
    }

    def _convert_features(
        self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
    ) -> tf.data.Dataset:
        """Convert the dataset to be fed to the encoder-decoder model.

        The conversion process involves two steps

        1. Each feature in the `task_feature_lengths` is trimmed/padded and
           optionally packed depending on the value of self.pack.
        2. "inputs" fields are mapped to the encoder input and "targets" are mapped
           to decoder input (after being shifted) and target.

        All the keys in the `task_feature_lengths` should be present in the input
        dataset, which may contain some extra features that are not in the
        `task_feature_lengths`. They will not be included in the output dataset.
        One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
        fields.

        Args:
          ds: an input tf.data.Dataset to be converted.
          task_feature_lengths: a mapping from feature to its length.

        Returns:
          ds: the converted dataset.
        """

        def convert_example(features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
            # targets_segment_id is present only for a packed dataset.
            decoder_input_tokens = utils.make_autoregressive_inputs(
                features["targets"], sequence_id=features.get("targets_segment_ids", None)
            )

            d = {
                "encoder_input_tokens": features["inputs"],
                "hyper_encoder_input_tokens": features["hyper_inputs"],
                "decoder_target_tokens": features["targets"],
                "decoder_input_tokens": decoder_input_tokens,
                # Loss is computed for all but the padding positions.
                "decoder_loss_weights": non_padding_position(features["targets"]),
            }

            if self.pack:
                d["encoder_segment_ids"] = features["inputs_segment_ids"]
                d["hyper_encoder_segment_ids"] = features["hyper_inputs_segment_ids"]
                d["decoder_segment_ids"] = features["targets_segment_ids"]
                d["encoder_positions"] = features["inputs_positions"]
                d["decoder_positions"] = features["targets_positions"]

            return d

        if self.pack:
            """
            Packing is non-trivial to get working with (a) the huggingface roberta model (possibly not supported?),
            and (b) the adapter generation side of things, since you would need to swap what adapters you're using
            mid-sequence (since there multiple examples in the same sequence). Not worth the effort. May revisit.
            """
            raise NotImplementedError("We do not use packing.")

        # padding only, no packing.
        ds = ds.map(
            lambda x: {
                k: trim_and_pad(k, t, task_feature_lengths, self.TASK_PADDING) for k, t in x.items()
            },
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        return ds.map(convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def get_model_feature_lengths(
        self, task_feature_lengths: Mapping[str, int]
    ) -> Mapping[str, int]:
        """Define the length relationship between input and output features."""
        encoder_length = task_feature_lengths["inputs"]
        hyper_encoder_length = task_feature_lengths["hyper_inputs"]
        decoder_length = task_feature_lengths["targets"]

        model_feature_lengths = {
            "encoder_input_tokens": encoder_length,
            "hyper_encoder_input_tokens": hyper_encoder_length,
            "decoder_target_tokens": decoder_length,
            "decoder_input_tokens": decoder_length,
            "decoder_loss_weights": decoder_length,
        }
        if self.pack:
            model_feature_lengths["encoder_segment_ids"] = encoder_length
            model_feature_lengths["hyper_encoder_segment_ids"] = hyper_encoder_length
            model_feature_lengths["decoder_segment_ids"] = decoder_length
            model_feature_lengths["encoder_positions"] = encoder_length
            model_feature_lengths["decoder_positions"] = decoder_length

        return model_feature_lengths


class HyperEncoderDecoderModel(EncoderDecoderModel):
    """Wrapper class for the models.Transformer nn.module."""

    FEATURE_CONVERTER_CLS = HyperEncDecFeatureConverter

    def __init__(
        self,
        module: nn.Module,
        input_vocabulary: seqio.Vocabulary,
        output_vocabulary: seqio.Vocabulary,
        optimizer_def: optimizers.OptimizerDefType,
        decode_fn: DecodeFnCallable = decoding.beam_search,
        feature_converter_cls: Optional[Callable[..., seqio.FeatureConverter]] = None,
        label_smoothing: float = 0.0,
        z_loss: float = 0.0,
        loss_normalizing_factor: Optional[float] = None,
    ):
        if feature_converter_cls is not None:
            self.FEATURE_CONVERTER_CLS = feature_converter_cls  # type: ignore # pylint: disable=invalid-name
        super().__init__(
            module=module,
            input_vocabulary=input_vocabulary,
            output_vocabulary=output_vocabulary,
            optimizer_def=optimizer_def,
            decode_fn=decode_fn,
            label_smoothing=label_smoothing,
            z_loss=z_loss,
            loss_normalizing_factor=loss_normalizing_factor,
        )

    def get_initial_variables(
        self,
        rng: jax.random.KeyArray,
        input_shapes: Mapping[str, Array],
        input_types: Optional[Mapping[str, jnp.dtype]] = None,
    ) -> flax_scope.FrozenVariableDict:
        """Get the initial variables for an encoder-decoder model."""
        input_types = {} if input_types is None else input_types
        encoder_shape = input_shapes["encoder_input_tokens"]
        encoder_type = input_types.get("encoder_input_tokens", jnp.float32)
        hyper_encoder_shape = input_shapes["hyper_encoder_input_tokens"]
        hyper_encoder_type = input_types.get("hyper_encoder_input_tokens", jnp.float32)
        decoder_shape = input_shapes["decoder_input_tokens"]
        decoder_type = input_types.get("decoder_input_tokens", jnp.float32)
        if "encoder_positions" in input_shapes:
            encoder_positions = jnp.ones(
                input_shapes["encoder_positions"], input_types.get("encoder_positions", jnp.int32)
            )
        else:
            encoder_positions = None
        if "hyper_encoder_positions" in input_shapes:
            hyper_encoder_positions = jnp.ones(
                input_shapes["hyper_encoder_positions"],
                input_types.get("hyper_encoder_positions", jnp.int32),
            )
        else:
            hyper_encoder_positions = None
        if "decoder_positions" in input_shapes:
            decoder_positions = jnp.ones(
                input_shapes["decoder_positions"], input_types.get("decoder_positions", jnp.int32)
            )
        else:
            decoder_positions = None
        if "encoder_segment_ids" in input_shapes:
            encoder_segment_ids = jnp.ones(
                input_shapes["encoder_segment_ids"],
                input_types.get("encoder_segment_ids", jnp.int32),
            )
        else:
            encoder_segment_ids = None
        if "hyper_encoder_segment_ids" in input_shapes:
            hyper_encoder_segment_ids = jnp.ones(
                input_shapes["hyper_encoder_segment_ids"],
                input_types.get("hyper_encoder_segment_ids", jnp.int32),
            )
        else:
            hyper_encoder_segment_ids = None
        if "decoder_segment_ids" in input_shapes:
            decoder_segment_ids = jnp.ones(
                input_shapes["decoder_segment_ids"],
                input_types.get("decoder_segment_ids", jnp.int32),
            )
        else:
            decoder_segment_ids = None
        initial_variables = self.module.init(
            rng,
            jnp.ones(encoder_shape, encoder_type),
            jnp.ones(hyper_encoder_shape, hyper_encoder_type),
            jnp.ones(decoder_shape, decoder_type),
            jnp.ones(decoder_shape, decoder_type),
            encoder_positions=encoder_positions,
            hyper_encoder_positions=hyper_encoder_positions,
            decoder_positions=decoder_positions,
            encoder_segment_ids=encoder_segment_ids,
            hyper_encoder_segment_ids=hyper_encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decode=False,
            enable_dropout=False,
        )

        override_param_axes = roberta_axes_names_override
        # TODO: override this in lora_network
        # TODO: don't do this for every case.

        override_param_axes += lora_axes_names_override
        #  roberta has no partitions, so we add that here.
        initial_variables = override_params_axes_names(initial_variables, override_param_axes)
        # add pretrained model
        initial_variables = unfreeze(initial_variables)
        roberta_params = FlaxRobertaModel.from_pretrained(
            self.module.config.roberta_model,
            max_position_embeddings=520,
            type_vocab_size=8,
            vocab_size=50272,
        ).params
        initial_variables["params"]["hyper"]["encoder"] = roberta_params
        initial_variables = freeze(initial_variables)
        return initial_variables

    def _compute_logits(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        dropout_rng: Optional[jax.random.KeyArray] = None,
        mutable: flax_scope.CollectionFilter = False,
        other_variables: Optional[PyTreeDef] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, flax_scope.FrozenVariableDict]]:
        """Computes logits via a forward pass of `self.module_cls`."""
        # Dropout is provided only for the training mode.
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else None
        if other_variables is None:
            other_variables = {}
        return self.module.apply(
            {"params": params, **other_variables},
            batch["encoder_input_tokens"],
            batch["hyper_encoder_input_tokens"],
            batch["decoder_input_tokens"],
            batch["decoder_target_tokens"],
            encoder_segment_ids=batch.get("encoder_segment_ids", None),
            hyper_encoder_segment_ids=batch.get("hyper_encoder_segment_ids", None),
            decoder_segment_ids=batch.get("decoder_segment_ids", None),
            encoder_positions=batch.get("encoder_positions", None),
            hyper_encoder_positions=batch.get("hyper_encoder_positions", None),
            decoder_positions=batch.get("decoder_positions", None),
            decode=False,
            enable_dropout=rngs is not None,
            rngs=rngs,
            mutable=mutable,
        )

    def _compute_logits_from_slice(
        self,
        flat_ids: jnp.ndarray,
        flat_cache: Mapping[str, jnp.ndarray],
        params: PyTreeDef,
        encoded_inputs: jnp.ndarray,
        adaptations: Dict[str, jnp.ndarray],  # Tuple[jnp.ndarray, ...],
        raw_inputs: jnp.ndarray,
        max_decode_length: int,
    ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Token slice to logits from decoder model."""
        # flat_ids: [batch * beam, seq_len=1]
        # cache is expanded inside beam_search to become flat_cache
        # flat_cache: [batch * beam, num_heads, depth_per_head, max_decode_len]
        # flat_logits: [batch * beam, seq_len=1, vocab]
        flat_logits, new_vars = self.module.apply(
            {"params": params, "cache": flat_cache},
            encoded_inputs,
            raw_inputs,  # only needed for encoder padding mask
            flat_ids,
            flat_ids,
            adapters=adaptations,
            enable_dropout=False,
            decode=True,
            max_decode_length=max_decode_length,
            mutable=["cache"],
            method=self.module.decode,
        )
        # Remove sequence length dimension since it's always 1 during decoding.
        flat_logits = jnp.squeeze(flat_logits, axis=1)
        new_flat_cache = new_vars["cache"]
        return flat_logits, new_flat_cache

    # for now, not heavily editing the decoder side.
    # will require more edits when I add hypernet stuff to decoder.
    def predict_batch_with_aux(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        rng: Optional[jax.random.KeyArray] = None,
        decoder_params: Optional[MutableMapping[str, Any]] = None,
        return_all_decodes: bool = False,
        num_decodes: int = 1,
        prompt_with_targets: bool = False,
    ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Predict with fast decoding beam search on a batch.

        Here we refer to "parameters" for values that can be compiled into the
        model dynamically, as opposed to static configuration settings that require
        a recompile. For example, the model weights and the decoder brevity-penalty
        are parameters and can be modified without requiring a recompile. The number
        of layers, the batch size and the decoder beam size are configuration
        options that require recompilation if changed.

        This method can be used with a customizable decoding function as long as it
        follows the signature of `DecodeFnCallable`. In order to provide a unified
        interface for the decoding functions, we use a generic names. For example, a
        beam size is a concept unique to beam search. Conceptually, it corresponds
        to the number of sequences returned by the beam search.  Therefore, the
        generic argument `num_decodes` corresponds to the beam size if
        `self._decode_fn` is a beam search. For temperature sampling, `num_decodes`
        corresponds to the number of independent sequences to be sampled. Typically
        `num_decodes = 1` is used for temperature sampling.

        If `return_all_decodes = True`, the return tuple contains the predictions
        with a shape [batch, num_decodes, max_decode_len] and the scores (i.e., log
        probability of the generated sequence) with a shape [batch, num_decodes].

        If `return_all_decodes = False`, the return tuple contains the predictions
        with a shape [batch, max_decode_len] and the scores with a shape [batch].

        `decoder_params` can be used to pass dynamic configurations to
        `self.decode_fn`. An example usage is to pass different random seed (i.e.,
        `jax.random.PRNGKey(seed)` with different `seed` value). This can be done by
        setting `decoder_params['decode_rng'] = jax.random.PRNGKey(seed)`.

        If `prompt_with_targets = True`, then `decoder_prompt_inputs` is initialized
        from the batch's `decoder_input_tokens`. The EOS is stripped to avoid
        decoding to stop after the prompt by matching to `output_vocabulary.eos_id`.

        Args:
          params: model parameters.
          batch: a batch of inputs.
          rng: an optional RNG key to use during prediction, which is passed as
            'decode_rng' to the decoding function.
          decoder_params: additional (model-independent) parameters for the decoder.
          return_all_decodes: whether to return the entire beam or just the top-1.
          num_decodes: the number of beams to use in beam search.
          prompt_with_targets: Whether the force decode decoder_inputs.

        Returns:
          A tuple containing:
            the batch of predictions, with the entire beam if requested
            an auxiliary dictionary of decoder scores
        """
        # Prepare zeroed-out autoregressive cache.
        # [batch, input_len]
        inputs = batch["encoder_input_tokens"]
        hyper_inputs = batch["hyper_encoder_input_tokens"]
        # [batch, target_len]
        target_shape = batch["decoder_input_tokens"].shape
        target_type = batch["decoder_input_tokens"].dtype
        _, variables_with_cache = self.module.apply(
            {"params": params},
            jnp.ones(inputs.shape, inputs.dtype),
            jnp.ones(hyper_inputs.shape, hyper_inputs.dtype),
            jnp.ones(target_shape, target_type),
            jnp.ones(target_shape, target_type),
            decode=True,
            enable_dropout=False,
            mutable=["cache"],
        )

        cache = variables_with_cache["cache"]

        # Prepare hypertransformer by first calling hypernet and storing the results. We will
        # pass these to both the following encoder and decoder calls.
        adaptations = self.module.apply(
            {"params": params}, hyper_inputs, enable_dropout=False, method=self.module.hyperencode
        )

        batch_adaptions = {
            a_name: decoding.flat_batch_beam_expand(a, num_decodes) if a is not None else None
            for a_name, a in adaptations.items()
        }

        # Prepare transformer fast-decoder call for beam search: for beam search, we
        # need to set up our decoder model to handle a batch size equal to
        # batch_size * num_decodes, where each batch item's data is expanded
        # in-place rather than tiled.
        # i.e. if we denote each batch element subtensor as el[n]:
        # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
        # [batch * num_decodes, input_len, emb_dim]
        # encoded inputs is the encoded states, ready for decoding.
        encoded_inputs = decoding.flat_batch_beam_expand(
            self.module.apply(
                {"params": params},
                inputs,
                adapters=adaptations,
                enable_dropout=False,
                method=self.module.encode,
            ),
            num_decodes,
        )

        # [batch * num_decodes, input_len]
        raw_inputs = decoding.flat_batch_beam_expand(inputs, num_decodes)

        tokens_ids_to_logits = functools.partial(
            self._compute_logits_from_slice,
            params=params,
            encoded_inputs=encoded_inputs,
            raw_inputs=raw_inputs,
            adaptations=batch_adaptions,
            max_decode_length=target_shape[1],
        )

        if decoder_params is None:
            decoder_params = {}
        if rng is not None:
            if decoder_params.get("decode_rng") is not None:
                raise ValueError(
                    f"Got RNG both from the `rng` argument ({rng}) and "
                    f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
                    "Please specify one or the other."
                )
            decoder_params["decode_rng"] = rng

        # `decoder_prompt_inputs` is initialized from the batch's
        # `decoder_input_tokens`. The EOS is stripped to avoid decoding to stop
        # after the prompt by matching to `output_vocabulary.eos_id`.
        # These inputs are ignored by the beam search decode fn.
        if prompt_with_targets:
            decoder_prompt_inputs = batch["decoder_input_tokens"]
            decoder_prompt_inputs = decoder_prompt_inputs * (
                decoder_prompt_inputs != self.output_vocabulary.eos_id
            )
        else:
            decoder_prompt_inputs = jnp.zeros_like(batch["decoder_input_tokens"])

        # TODO(hwchung): rename the returned value names to more generic ones.
        # Using the above-defined single-step decoder function, run a
        # beam search over possible sequences given input encoding.
        # decodes: [batch, num_decodes, max_decode_len + 1]
        # scores: [batch, num_decodes]
        scanned = hasattr(self.module, "scan_layers") and self.module.scan_layers
        decodes, scores = self._decode_fn(
            inputs=decoder_prompt_inputs,
            cache=cache,
            tokens_to_logits=tokens_ids_to_logits,
            eos_id=self.output_vocabulary.eos_id,
            num_decodes=num_decodes,
            cache_offset=1 if scanned else 0,
            **decoder_params,
        )

        # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
        # in increasing order of log-probability.
        # Return the highest scoring beam sequence.
        if return_all_decodes:
            return decodes, {"scores": scores}
        else:
            return decodes[:, -1, :], {"scores": scores[:, -1]}


class HyperEncDecContFeatureConverter(HyperEncDecFeatureConverter):
    """Feature converter for an encoder-decoder with hypernet architecture.
    Really this is just providing a second encoder input to the model.
    """

    TASK_FEATURES = {
        "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "hyper_inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "task_names": FeatureConverter.FeatureSpec(dtype=tf.int32),
    }
    # t5x by default pads everything with the token id 0, but for roberta it is the token id 1
    TASK_PADDING = {"inputs": 0, "hyper_inputs": 1, "targets": 0, "task_names": 0}
    MODEL_FEATURES = {
        "encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "hyper_encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "task_names": FeatureConverter.FeatureSpec(dtype=tf.int32),
    }
    PACKING_FEATURE_DTYPES = {
        "encoder_segment_ids": tf.int32,
        "hyper_encoder_segment_ids": tf.int32,
        "decoder_segment_ids": tf.int32,
        "encoder_positions": tf.int32,
        "decoder_positions": tf.int32,
        "task_names": tf.int32,
    }

    def _convert_features(
        self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
    ) -> tf.data.Dataset:
        """Convert the dataset to be fed to the encoder-decoder model.

        The conversion process involves two steps

        1. Each feature in the `task_feature_lengths` is trimmed/padded and
           optionally packed depending on the value of self.pack.
        2. "inputs" fields are mapped to the encoder input and "targets" are mapped
           to decoder input (after being shifted) and target.

        All the keys in the `task_feature_lengths` should be present in the input
        dataset, which may contain some extra features that are not in the
        `task_feature_lengths`. They will not be included in the output dataset.
        One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
        fields.

        Args:
          ds: an input tf.data.Dataset to be converted.
          task_feature_lengths: a mapping from feature to its length.

        Returns:
          ds: the converted dataset.
        """

        def convert_example(features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
            # targets_segment_id is present only for a packed dataset.
            decoder_input_tokens = utils.make_autoregressive_inputs(
                features["targets"], sequence_id=features.get("targets_segment_ids", None)
            )

            d = {
                "encoder_input_tokens": features["inputs"],
                "hyper_encoder_input_tokens": features["hyper_inputs"],
                "decoder_target_tokens": features["targets"],
                "decoder_input_tokens": decoder_input_tokens,
                # Loss is computed for all but the padding positions.
                "decoder_loss_weights": non_padding_position(features["targets"]),
                "task_names": features["task_names"],
            }

            if self.pack:
                d["encoder_segment_ids"] = features["inputs_segment_ids"]
                d["hyper_encoder_segment_ids"] = features["hyper_inputs_segment_ids"]
                d["decoder_segment_ids"] = features["targets_segment_ids"]
                d["encoder_positions"] = features["inputs_positions"]
                d["decoder_positions"] = features["targets_positions"]
                d["task_names"] = features["task_names"]

            return d

        if self.pack:
            """
            Packing is non-trivial to get working with (a) the huggingface roberta model (possibly not supported?),
            and (b) the adapter generation side of things, since you would need to swap what adapters you're using
            mid-sequence (since there multiple examples in the same sequence). Not worth the effort. May revisit.
            """
            raise NotImplementedError("We do not use packing.")

        # padding only, no packing.
        ds = ds.map(
            lambda x: {
                k: trim_and_pad(k, t, task_feature_lengths, self.TASK_PADDING) for k, t in x.items()
            },
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        return ds.map(convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def get_model_feature_lengths(
        self, task_feature_lengths: Mapping[str, int]
    ) -> Mapping[str, int]:
        """Define the length relationship between input and output features."""
        encoder_length = task_feature_lengths["inputs"]
        hyper_encoder_length = task_feature_lengths["hyper_inputs"]
        decoder_length = task_feature_lengths["targets"]
        task_name_length = task_feature_lengths["task_names"]

        model_feature_lengths = {
            "encoder_input_tokens": encoder_length,
            "hyper_encoder_input_tokens": hyper_encoder_length,
            "decoder_target_tokens": decoder_length,
            "decoder_input_tokens": decoder_length,
            "decoder_loss_weights": decoder_length,
            "task_names": task_name_length,
        }
        if self.pack:
            model_feature_lengths["encoder_segment_ids"] = encoder_length
            model_feature_lengths["hyper_encoder_segment_ids"] = hyper_encoder_length
            model_feature_lengths["decoder_segment_ids"] = decoder_length
            model_feature_lengths["encoder_positions"] = encoder_length
            model_feature_lengths["decoder_positions"] = decoder_length
            model_feature_lengths["task_names"] = task_name_length

        return model_feature_lengths


class HyperEncoderDecoderContrastiveModel(HyperEncoderDecoderModel):
    """
    Basically our hypernet-based model, but with a contrastive loss.
    """

    FEATURE_CONVERTER_CLS = HyperEncDecContFeatureConverter

    def loss_fn(
        self,
        params: PyTreeDef,
        batch: Mapping[str, jnp.ndarray],
        dropout_rng: Optional[jax.random.KeyArray],
        cosine_loss_multiplier: int = 6000,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, metrics_lib.MetricsMap]]:
        """"""
        logits, mod_vars = self._compute_logits(params, batch, dropout_rng, mutable="intermediates")
        # note we should only have one hypernet feature (hypernet called once)
        hypernet_feats = mod_vars["intermediates"]["hyper"]["features"][0]
        # construct the contrastive loss truth
        cosine_truth = (batch["task_names"][None, :, 0] == batch["task_names"]).astype(jnp.int32)

        # cosine loss - for truth we want 0 for neg (not same task), 1 for pos (same task)
        cos_loss = cosine_similarity_loss(hypernet_feats, hypernet_feats, cosine_truth)

        loss_normalizing_factor: Optional[
            Union[float, int, str, losses.SpecialLossNormalizingFactor]
        ]
        (loss_normalizing_factor, weights) = losses.get_loss_normalizing_factor_and_weights(
            self._loss_normalizing_factor, batch
        )

        loss, z_loss, _ = losses.compute_weighted_cross_entropy(
            logits,
            targets=batch["decoder_target_tokens"],
            weights=weights,
            label_smoothing=self._label_smoothing,
            z_loss=self._z_loss,
            loss_normalizing_factor=loss_normalizing_factor,
        )
        # loss += cos_loss * cosine_loss_multiplier  # upweight since otherwise ce loss dominates
        metrics = self._compute_metrics(
            logits=logits,
            targets=batch["decoder_target_tokens"],
            mask=weights,
            loss=loss,
            z_loss=z_loss,
            cosine_loss=cos_loss,
            cosine_truth=cosine_truth,
        )
        return loss, metrics

    def _compute_metrics(
        self,
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
        loss: jnp.ndarray,
        z_loss: Optional[jnp.ndarray] = None,
        cosine_loss: Optional[jnp.ndarray] = None,
        cosine_truth: Optional[jnp.ndarray] = None,
    ) -> metrics_lib.MetricsMap:
        metrics = compute_base_metrics(
            logits=logits, targets=targets, mask=mask, loss=loss, z_loss=z_loss
        )
        if cosine_loss is not None and cosine_truth is not None:
            metrics.update(
                {
                    "cosine_loss": metrics_lib.AveragePerStep(total=cosine_loss),
                    "positive_cosine_samples": clu_metrics.Average(
                        total=cosine_truth.sum(), count=jnp.ones_like(cosine_truth).sum()
                    ),
                    "total_positive_samples_per_step": metrics_lib.AveragePerStep(
                        total=cosine_truth.sum()
                    ),
                }
            )
        return metrics


class LoraEncoderDecoderModel(EncoderDecoderModel):
    FEATURE_CONVERTER_CLS = HyperEncDecContFeatureConverter

    def get_initial_variables(
        self,
        rng: jax.random.KeyArray,
        input_shapes: Mapping[str, Array],
        input_types: Optional[Mapping[str, jnp.dtype]] = None,
    ) -> flax_scope.FrozenVariableDict:
        initial_variables = super().get_initial_variables(rng, input_shapes, input_types)

        # Add lora partitions
        override_param_axes = lora_axes_names_override
        initial_variables = override_params_axes_names(initial_variables, override_param_axes)
        return initial_variables
