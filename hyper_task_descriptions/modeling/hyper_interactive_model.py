# type: ignore
"""
Some changes to the t5x interactive model for dual-input setup.
"""
import functools
import inspect
import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Callable, Tuple, Union

import clu.data.dataset_iterator
import jax
import seqio
import tensorflow as tf
from t5x import trainer as trainer_lib
from t5x import utils
from t5x.infer import _Inferences
from t5x.interactive_model import (
    InferenceType,
    InteractiveModel,
    _extract_tokens_and_aux_values,
)

from hyper_task_descriptions.hf_vocab import HuggingfaceVocabulary


class HyperInteractiveModel(InteractiveModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        t5_vocab = HuggingfaceVocabulary("t5-base")
        output_features = {
            "inputs": seqio.Feature(t5_vocab, add_eos=True, dtype=tf.int32),
            "hyper_inputs": seqio.Feature(t5_vocab, add_eos=True, dtype=tf.int32),
            "targets": seqio.Feature(t5_vocab, add_eos=True, dtype=tf.int32),
            "task_names": seqio.Feature(
                seqio.PassThroughVocabulary(1), add_eos=False, dtype=tf.int32
            ),
        }
        self._features = dict(sorted(output_features.items()))

    def train_step_with_preprocessors(
        self,
        examples: Sequence[Union[str, dict[str, str]]],
        preprocessors: Sequence[Callable[..., tf.data.Dataset]],
    ):
        # --------------------------------------------------------------------------
        # Initialize dataset and dataset iterator
        # --------------------------------------------------------------------------
        if len(examples) < self._batch_size:
            raise ValueError(
                "At least one batch of data must be provided. Please decrease the "
                "batch_size or provide more examples."
            )

        train_dataset = get_dataset_from_natural_text_examples(
            examples,
            preprocessors=preprocessors,
            task_feature_lengths=self._task_feature_lengths,
            features=self._features,
        )
        train_dataset = self._feature_converter(
            train_dataset, task_feature_lengths=self._task_feature_lengths
        )
        train_dataset = train_dataset.padded_batch(self._batch_size, drop_remainder=True)
        train_iter = clu.data.dataset_iterator.TfDatasetIterator(train_dataset, checkpoint=True)

        # --------------------------------------------------------------------------
        # Take 1 train step.
        # --------------------------------------------------------------------------
        # `stop_training` is requested, break out the main loop immediately.
        if self._trainer.stop_training:
            logging.info("Stopping training early since `stop_training` is requested.")
            return

        if isinstance(self._train_state, Sequence):
            raise ValueError("Expected a single train state, but instead received a Sequence.")
        try:
            first_step = int(utils.get_local_data(self._train_state.step))
            self._train_summary = self._trainer.train(train_iter, 1, start_step=first_step)
        except trainer_lib.PreemptionError as e:
            logging.info("Saving emergency checkpoint.")
            self._checkpoint_manager.save(
                self._trainer.train_state, self._save_checkpoint_cfg.state_transformation_fns
            )
            logging.info("Saving emergency checkpoint done.")
            raise e

        # Save a checkpoint.
        logging.info("Saving checkpoint.")
        self._checkpoint_manager.save(
            self._trainer.train_state, self._save_checkpoint_cfg.state_transformation_fns
        )

        # Wait until computations are done before exiting
        utils.sync_global_devices("complete")
        self._train_state = self._trainer.train_state

    def infer_with_preprocessors(
        self,
        mode: InferenceType,
        examples: Sequence[Union[str, dict[str, str]]],
        preprocessors: Sequence[Callable[..., tf.data.Dataset]],
    ) -> _Inferences:
        # --------------------------------------------------------------------------
        # Parse Mode
        # --------------------------------------------------------------------------
        if mode == InferenceType.PREDICT_WITH_AUX:
            infer_step = self._model.predict_batch_with_aux
        elif mode == InferenceType.SCORE:
            infer_step = self._model.score_batch
        else:
            raise ValueError(
                "Mode must be `predict_with_aux`, or `score`," f" but instead was {mode}."
            )
        infer_fn = functools.partial(
            utils.get_infer_fn(
                infer_step=infer_step,
                batch_size=self._batch_size,
                train_state_axes=self._train_state_initializer.train_state_axes,
                partitioner=self._partitioner,
            ),
            train_state=self._train_state,
        )

        # --------------------------------------------------------------------------
        # Construct a dataset and dataset iterator.
        # --------------------------------------------------------------------------
        dataset = get_dataset_from_natural_text_examples(
            examples,
            preprocessors=preprocessors,
            task_feature_lengths=self._task_feature_lengths,
            features=self._features,
        )
        model_dataset = self._feature_converter(
            dataset, task_feature_lengths=self._task_feature_lengths
        )
        # Zip task and model features.
        infer_dataset = tf.data.Dataset.zip((dataset, model_dataset))
        # Create batches and index them.
        infer_dataset = infer_dataset.padded_batch(
            self._batch_size, drop_remainder=False
        ).enumerate()
        infer_dataset_iter: Iterator[Tuple[int, Any]] = iter(
            infer_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        )

        # --------------------------------------------------------------------------
        # Run inference
        # --------------------------------------------------------------------------
        # Main Loop over "batches".
        all_inferences = []
        all_aux_values = {}
        for chunk, chunk_batch in infer_dataset_iter:
            # Load the dataset for the next chunk. We can't use `infer_dataset_iter`
            # directly since `infer_fn` needs to know the exact size of each chunk,
            # which may be smaller for the final one.
            chunk_dataset = tf.data.Dataset.from_tensor_slices(chunk_batch)
            chunk_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)

            # Unzip chunk dataset in to pretokenized and model datasets.
            task_dataset = chunk_dataset.map(
                lambda p, m: p, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            model_dataset = chunk_dataset.map(
                lambda p, m: m, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

            # Get a chunk-specific RNG key.
            chunk_rng = jax.random.fold_in(jax.random.PRNGKey(0), chunk)

            inferences = _extract_tokens_and_aux_values(
                infer_fn(model_dataset.enumerate(), rng=chunk_rng)
            )

            predictions, aux_values = inferences
            accumulated_inferences = []
            for idx, inputs in task_dataset.enumerate().as_numpy_iterator():
                prediction = predictions[idx]
                # Decode predictions if applicable.
                if mode == InferenceType.PREDICT_WITH_AUX:
                    prediction = (
                        self._features["targets"]
                        .vocabulary.decode_tf(tf.constant(prediction))
                        .numpy()
                    )
                accumulated_inferences.append((inputs, prediction))
            all_inferences += accumulated_inferences
            # Accumulate aux values over batches.
            if not all_aux_values:
                all_aux_values = aux_values
            else:
                for key, values in aux_values.items():
                    all_aux_values[key] += values

        return all_inferences, all_aux_values


def get_dataset_from_natural_text_examples(
    examples: Sequence[Union[str, dict[str, str]]],
    preprocessors: Sequence[Callable[..., tf.data.Dataset]],
    task_feature_lengths: Mapping[str, int],
    features: Mapping[str, Any],
) -> tf.data.Dataset:
    """Returns a tf.data.Dataset from a list of examples.
    Args:
      examples: a single batch of examples that should be transformed into a
        tf.data.Dataset. The examples can either take the form of a string (ex: a
        single input for inference), or a dictionary mapping "input"/"target" to a
        string containing that element.
      preprocessors: an optional list of functions that receive a tf.data.Dataset
        and return a tf.data.Dataset. These will be executed sequentially and the
        final dataset must include features matching `self._features`.
      task_feature_lengths: dictionary mapping feature key to maximum length (int)
        for that feature. If feature is longer than this length after
        preprocessing, the feature will be truncated. May be set to None to avoid
        truncation.
      features: dictionary defining what features should be present in all
        examples.
    Returns:
      A tf.data.Dataset.
    """
    # ------------------------------------------------------------------------
    # Construct a `tf.data.Dataset` from the provided examples
    # ------------------------------------------------------------------------
    merged_examples = {
        "inputs": [],
        "hyper_inputs": [],
        "targets": [],
        "answers": [],
        "task_names": [],
    }
    for example in examples:
        # If the provided example is just a string, add an empty target string
        if isinstance(example, dict):
            example_dict = example
        else:
            raise ValueError("Example must be a dict")
        merged_examples["inputs"].append(example_dict["input"])
        merged_examples["targets"].append(example_dict["target"])
        merged_examples["hyper_inputs"].append(example_dict["hyper_input"])
        merged_examples["task_names"].append(example_dict["task_name"])
        # Answers is a list of possible targets (in this case a single target).
        merged_examples["answers"].append([example_dict["target"]])
    dataset = tf.data.Dataset.from_tensor_slices(merged_examples)

    # Define `ShardInfo` that doesn't shard the data pipeline.
    shard_info = seqio.ShardInfo(0, 1)
    dataset = dataset.shard(shard_info.num_shards, shard_info.index)

    # ------------------------------------------------------------------------
    # Preprocess data
    # ------------------------------------------------------------------------
    for prep_fn in preprocessors:
        # prep_fn must not rely on variable length keyword args such as **kwargs.
        fn_args = set(inspect.signature(prep_fn).parameters.keys())
        kwargs = {}
        if "sequence_length" in fn_args:
            kwargs["sequence_length"] = task_feature_lengths
        if "output_features" in fn_args:
            kwargs["output_features"] = features
        dataset = prep_fn(dataset, **kwargs)

    def _validate_preprocessing(dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Validates preprocessed dataset, raising Exceptions if needed.
        Args:
          dataset: a tf.data.Dataset to validate.
        Returns:
          a validated tf.data.Dataset.
        Raises:
          ValueError: dataset has missing feature or the incorrect type/rank for a
            feature.
        """
        actual_specs = dataset.element_spec
        for feat, feat_spec in features.items():
            if feat not in actual_specs:
                if feat_spec.required:
                    raise ValueError(
                        "Task dataset is missing expected output feature after "
                        f"preprocessing: {feat}"
                    )
                else:
                    # It's ok that this feature does not exist.
                    continue
            actual_spec = actual_specs[feat]
            if feat_spec.dtype != actual_spec.dtype:
                raise ValueError(
                    f"Task dataset has incorrect type for feature '{feat}' after "
                    f"preprocessing: Got {actual_spec.dtype.name}, expected "
                    f"{feat_spec.dtype.name}"
                )
            if feat_spec.rank != actual_spec.shape.rank:
                raise ValueError(
                    f"Task dataset has incorrect rank for feature '{feat}' after "
                    f"preprocessing: Got {actual_spec.shape.rank}, expected "
                    f"{feat_spec.rank}"
                )

        return dataset

    dataset = _validate_preprocessing(dataset)
    dataset = seqio.utils.trim_dataset(dataset, task_feature_lengths, features)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)
