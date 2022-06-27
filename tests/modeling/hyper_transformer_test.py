import pytest
import tensorflow as tf
from seqio.test_utils import assert_dataset, create_default_dataset

from hyper_task_descriptions.modeling.hyper_transformer import (
    HyperEncDecFeatureConverter,
)


def test_encoder_decoder_unpacked():
    x = [{"inputs": [9, 4, 3, 8, 1], "hyper_inputs": [5, 7], "targets": [3, 9, 4, 1]}]
    ds = create_default_dataset(x, feature_names=["inputs", "hyper_inputs", "targets"])
    task_feature_lengths = {"inputs": 7, "hyper_inputs": 3, "targets": 5}

    converter = HyperEncDecFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
        "decoder_target_tokens": [3, 9, 4, 1, 0],
        "hyper_encoder_input_tokens": [5, 7, 1],  # hyper_inputs padding is 1.
        # mtf.transformer.autoregressive_inputs does not zero out the last eos
        # when the data is not packed. This test mimic the behavior.
        "decoder_input_tokens": [0, 3, 9, 4, 1],
        "decoder_loss_weights": [1, 1, 1, 1, 0],
    }
    assert_dataset(converted_ds, expected)


def test_encoder_decoder_targets_max_length():
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5, 1], "hyper_inputs": [5, 7, 3, 1, 2]}]
    ds = create_default_dataset(x, feature_names=["inputs", "hyper_inputs", "targets"])
    task_feature_lengths = {"inputs": 5, "targets": 5, "hyper_inputs": 5}

    converter = HyperEncDecFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 1],
        "decoder_target_tokens": [3, 9, 4, 5, 1],
        "hyper_encoder_input_tokens": [5, 7, 3, 1, 2],
        "decoder_input_tokens": [0, 3, 9, 4, 5],
        "decoder_loss_weights": [1, 1, 1, 1, 1],
    }
    assert_dataset(converted_ds, expected)


def test_encoder_decoder_extra_long_inputs():
    x = [
        {
            "inputs": [9, 4, 3, 8, 4, 5, 1],
            "targets": [3, 9, 4, 7, 8, 1],
            "hyper_inputs": [5, 7, 3, 1, 2],
        }
    ]
    ds = create_default_dataset(x, feature_names=["inputs", "hyper_inputs", "targets"])
    task_feature_lengths = {"inputs": 5, "targets": 8, "hyper_inputs": 3}
    expected_msg = (
        r".*Feature \\'inputs\\' has length not less than or equal to the "
        r"expected length of 5 during input_validation.*"
    )
    with pytest.raises(tf.errors.InvalidArgumentError, match=expected_msg):
        converter = HyperEncDecFeatureConverter(pack=False)
        converted_ds = converter(ds, task_feature_lengths)
        list(converted_ds.as_numpy_iterator())


def test_encoder_decoder_extra_long_hyper_inputs():
    x = [
        {"inputs": [9, 4, 3, 8, 4], "targets": [3, 9, 4, 7, 8, 1], "hyper_inputs": [5, 7, 3, 1, 2]}
    ]
    ds = create_default_dataset(x, feature_names=["inputs", "hyper_inputs", "targets"])
    task_feature_lengths = {"inputs": 5, "targets": 8, "hyper_inputs": 3}
    expected_msg = (
        r".*Feature \\'hyper_inputs\\' has length not less than or equal to the "
        r"expected length of 3 during input_validation.*"
    )
    with pytest.raises(tf.errors.InvalidArgumentError, match=expected_msg):
        converter = HyperEncDecFeatureConverter(pack=False)
        converted_ds = converter(ds, task_feature_lengths)
        list(converted_ds.as_numpy_iterator())


def test_encoder_decoder_packed():
    x = [
        {"inputs": [7, 8, 5, 1], "targets": [3, 9, 1], "hyper_inputs": [5, 7]},
        {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1], "hyper_inputs": [6, 3, 2]},
    ]
    ds = create_default_dataset(x, feature_names=["inputs", "hyper_inputs", "targets"])
    task_feature_lengths = {"inputs": 10, "targets": 7, "hyper_inputs": 8}

    with pytest.raises(NotImplementedError, match="We do not use packing."):
        converter = HyperEncDecFeatureConverter(pack=True)
        converted_ds = converter(ds, task_feature_lengths)
        list(converted_ds.as_numpy_iterator())
        # expected = {
        #     "encoder_input_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        #     "encoder_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        #     "encoder_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        #     "hyper_encoder_input_tokens": [5, 7, 6, 3, 2, 1, 1, 1],
        #     "hyper_encoder_segment_ids": [1, 1, 2, 2, 2, 0, 0, 0],
        #     "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
        #     "decoder_input_tokens": [0, 3, 9, 0, 4, 0, 0],
        #     "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
        #     "decoder_segment_ids": [1, 1, 1, 2, 2, 0, 0],
        #     "decoder_positions": [0, 1, 2, 0, 1, 0, 0],
        # }
        # assert_dataset(converted_ds, expected)


def test_encoder_decoder_pretokenized_field():
    x = [
        {
            "inputs": [7, 8, 5, 1],
            "targets": [3, 9, 1],
            "hyper_inputs": [5, 7],
            "targets_pretokenized": "abc",
        },
        {
            "inputs": [8, 4, 9, 3, 1],
            "targets": [4, 1],
            "hyper_inputs": [6, 3, 1],
            "targets_pretokenized": "def",
        },
    ]
    types = {
        "inputs": tf.int32,
        "targets": tf.int32,
        "hyper_inputs": tf.int32,
        "targets_pretokenized": tf.string,
    }
    shapes = {
        "inputs": [None],
        "targets": [None],
        "targets_pretokenized": [],
        "hyper_inputs": [None],
    }
    ds = tf.data.Dataset.from_generator(lambda: x, output_types=types, output_shapes=shapes)

    task_feature_lengths = {"inputs": 10, "targets": 7, "hyper_inputs": 8}
    converter = HyperEncDecFeatureConverter(pack=False)
    # Check whether convert_features raise error because targets_pretokenized is
    # present in the ds but not in the task_feature_lengths
    converter(ds, task_feature_lengths)
