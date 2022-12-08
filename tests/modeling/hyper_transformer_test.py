import functools
from unittest import mock

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized
from flax import traverse_util
from seqio.test_utils import assert_dataset, create_default_dataset
from t5x import decoding

from hyper_task_descriptions.common.testing import (
    HyperTaskDescriptionsTestCase,
    get_test_model,
)
from hyper_task_descriptions.modeling.hyper_network import HyperT5Config
from hyper_task_descriptions.modeling.hyper_transformer import (
    HyperEncDecContFeatureConverter,
    HyperEncDecFeatureConverter,
    HyperEncoderDecoderModel,
)

BATCH_SIZE, ENCODER_LEN, MAX_DECODE_LEN, EMBED_DIM, HYPER_ENCODER_LEN = 2, 3, 4, 5, 6


class TestHyperEncDecFeatureConverter(HyperTaskDescriptionsTestCase):
    def test_encoder_decoder_unpacked(self):
        x = [{"inputs": [9, 4, 3, 8, 1], "hyper_inputs": [5, 7], "targets": [3, 9, 4, 1]}]
        ds = create_default_dataset(x, feature_names=["inputs", "hyper_inputs", "targets"])
        task_feature_lengths = {"inputs": 7, "hyper_inputs": 3, "targets": 5}

        converter = HyperEncDecFeatureConverter(pack=False)
        converted_ds = converter(ds, task_feature_lengths)

        expected = {
            "encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
            "decoder_target_tokens": [3, 9, 4, 1, 0],
            "hyper_encoder_input_tokens": [5, 7, 0],  # hyper_inputs padding is 0.
            # mtf.transformer.autoregressive_inputs does not zero out the last eos
            # when the data is not packed. This test mimic the behavior.
            "decoder_input_tokens": [0, 3, 9, 4, 1],
            "decoder_loss_weights": [1, 1, 1, 1, 0],
        }
        assert_dataset(converted_ds, expected)

    def test_encoder_decoder_targets_max_length(self):
        x = [
            {"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 5, 1], "hyper_inputs": [5, 7, 3, 1, 2]}
        ]
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

    def test_encoder_decoder_extra_long_inputs(self):
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

    def test_encoder_decoder_extra_long_hyper_inputs(self):
        x = [
            {
                "inputs": [9, 4, 3, 8, 4],
                "targets": [3, 9, 4, 7, 8, 1],
                "hyper_inputs": [5, 7, 3, 1, 2],
            }
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

    def test_encoder_decoder_packed(self):
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

    def test_encoder_decoder_pretokenized_field(self):
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


class TestHyperEncDecContFeatureConverter(HyperTaskDescriptionsTestCase):
    def test_encoder_decoder_unpacked(self):
        x = [
            {
                "inputs": [9, 4, 3, 8, 1],
                "hyper_inputs": [5, 7],
                "targets": [3, 9, 4, 1],
                "task_names": [9, 5, 2],
            }
        ]
        ds = create_default_dataset(
            x, feature_names=["inputs", "hyper_inputs", "targets", "task_names"]
        )
        task_feature_lengths = {"inputs": 7, "hyper_inputs": 3, "targets": 5, "task_names": 4}

        converter = HyperEncDecContFeatureConverter(pack=False)
        converted_ds = converter(ds, task_feature_lengths)

        expected = {
            "encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
            "decoder_target_tokens": [3, 9, 4, 1, 0],
            "hyper_encoder_input_tokens": [5, 7, 0],  # hyper_inputs padding is 0.
            # mtf.transformer.autoregressive_inputs does not zero out the last eos
            # when the data is not packed. This test mimic the behavior.
            "decoder_input_tokens": [0, 3, 9, 4, 1],
            "decoder_loss_weights": [1, 1, 1, 1, 0],
            "task_names": [9, 5, 2, 0],
        }
        assert_dataset(converted_ds, expected)

    def test_encoder_decoder_targets_max_length(self):
        x = [
            {
                "inputs": [9, 4, 3, 8, 1],
                "targets": [3, 9, 4, 5, 1],
                "hyper_inputs": [5, 7, 3, 1, 2],
                "task_names": [9, 5, 2, 3, 1],
            }
        ]
        ds = create_default_dataset(
            x, feature_names=["inputs", "hyper_inputs", "targets", "task_names"]
        )
        task_feature_lengths = {"inputs": 5, "targets": 5, "hyper_inputs": 5, "task_names": 5}

        converter = HyperEncDecContFeatureConverter(pack=False)
        converted_ds = converter(ds, task_feature_lengths)

        expected = {
            "encoder_input_tokens": [9, 4, 3, 8, 1],
            "decoder_target_tokens": [3, 9, 4, 5, 1],
            "hyper_encoder_input_tokens": [5, 7, 3, 1, 2],
            "decoder_input_tokens": [0, 3, 9, 4, 5],
            "decoder_loss_weights": [1, 1, 1, 1, 1],
            "task_names": [9, 5, 2, 3, 1],
        }
        assert_dataset(converted_ds, expected)

    def test_encoder_decoder_extra_long_inputs(self):
        x = [
            {
                "inputs": [9, 4, 3, 8, 4, 5, 1],
                "targets": [3, 9, 4, 7, 8, 1],
                "hyper_inputs": [5, 7, 3, 1, 2],
                "task_names": [9, 5, 2],
            }
        ]
        ds = create_default_dataset(
            x, feature_names=["inputs", "hyper_inputs", "targets", "task_names"]
        )
        task_feature_lengths = {"inputs": 5, "targets": 8, "hyper_inputs": 3, "task_names": 4}
        expected_msg = (
            r".*Feature \\'inputs\\' has length not less than or equal to the "
            r"expected length of 5 during input_validation.*"
        )
        with pytest.raises(tf.errors.InvalidArgumentError, match=expected_msg):
            converter = HyperEncDecContFeatureConverter(pack=False)
            converted_ds = converter(ds, task_feature_lengths)
            list(converted_ds.as_numpy_iterator())

    def test_encoder_decoder_extra_long_task_names(self):
        x = [
            {
                "inputs": [9, 4, 3, 8, 4],
                "targets": [3, 9, 4, 7, 8, 1],
                "hyper_inputs": [5, 7, 3],
                "task_names": [9, 5, 2, 7, 8, 4],
            }
        ]
        ds = create_default_dataset(
            x, feature_names=["inputs", "hyper_inputs", "targets", "task_names"]
        )
        task_feature_lengths = {"inputs": 5, "targets": 8, "hyper_inputs": 3, "task_names": 5}
        expected_msg = (
            r".*Feature \\'task_names\\' has length not less than or equal to the "
            r"expected length of 5 during input_validation.*"
        )
        with pytest.raises(tf.errors.InvalidArgumentError, match=expected_msg):
            converter = HyperEncDecContFeatureConverter(pack=False)
            converted_ds = converter(ds, task_feature_lengths)
            list(converted_ds.as_numpy_iterator())

    def test_encoder_decoder_packed(self):
        x = [
            {
                "inputs": [7, 8, 5, 1],
                "targets": [3, 9, 1],
                "hyper_inputs": [5, 7],
                "task_names": [5, 8, 1],
            },
            {
                "inputs": [8, 4, 9, 3, 1],
                "targets": [4, 1],
                "hyper_inputs": [6, 3, 2],
                "task_names": [7, 2],
            },
        ]
        ds = create_default_dataset(
            x, feature_names=["inputs", "hyper_inputs", "targets", "task_names"]
        )
        task_feature_lengths = {"inputs": 10, "targets": 7, "hyper_inputs": 8, "task_names": 9}

        with pytest.raises(NotImplementedError, match="We do not use packing."):
            converter = HyperEncDecContFeatureConverter(pack=True)
            converted_ds = converter(ds, task_feature_lengths)
            list(converted_ds.as_numpy_iterator())

    def test_encoder_decoder_pretokenized_field(self):
        x = [
            {
                "inputs": [7, 8, 5, 1],
                "targets": [3, 9, 1],
                "hyper_inputs": [5, 7],
                "targets_pretokenized": "abc",
                "task_names": [5, 2, 1],
            },
            {
                "inputs": [8, 4, 9, 3, 1],
                "targets": [4, 1],
                "hyper_inputs": [6, 3, 1],
                "targets_pretokenized": "def",
                "task_names": [6, 2],
            },
        ]
        types = {
            "inputs": tf.int32,
            "targets": tf.int32,
            "hyper_inputs": tf.int32,
            "targets_pretokenized": tf.string,
            "task_names": tf.int32,
        }
        shapes = {
            "inputs": [None],
            "targets": [None],
            "targets_pretokenized": [],
            "hyper_inputs": [None],
            "task_names": [None],
        }
        ds = tf.data.Dataset.from_generator(lambda: x, output_types=types, output_shapes=shapes)

        task_feature_lengths = {"inputs": 10, "targets": 7, "hyper_inputs": 8, "task_names": 9}
        converter = HyperEncDecContFeatureConverter(pack=False)
        # Check whether convert_features raise error because targets_pretokenized is
        # present in the ds but not in the task_feature_lengths
        converter(ds, task_feature_lengths)


class TestHyperEncoderDecoderModel(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(
            testcase_name="no_types",
            shapes={
                "encoder_input_tokens": [1, 512],
                "decoder_input_tokens": [1, 62],
                "hyper_encoder_input_tokens": [1, 8],
            },
            types=None,
        ),
        dict(
            testcase_name="int32",
            shapes={
                "encoder_input_tokens": [1, 512],
                "decoder_input_tokens": [1, 62],
                "hyper_encoder_input_tokens": [1, 8],
            },
            types={
                "encoder_input_tokens": jnp.int32,
                "decoder_input_tokens": jnp.int32,
                "hyper_encoder_input_tokens": jnp.int32,
            },
        ),
        dict(
            testcase_name="float32",
            shapes={
                "encoder_input_tokens": [1, 512],
                "decoder_input_tokens": [1, 62],
                "hyper_encoder_input_tokens": [1, 8],
                "encoder_positions": [1, 512],
                "decoder_positions": [1, 62],
                "hyper_encoder_positions": [1, 8],
            },
            types={
                "encoder_input_tokens": jnp.int32,
                "decoder_input_tokens": jnp.int32,
                "hyper_encoder_input_tokens": jnp.int32,
                "encoder_positions": jnp.int32,
                "decoder_positions": jnp.int32,
                "hyper_encoder_positions": jnp.int32,
            },
        ),
        #   dict(
        #       testcase_name='float32_segment_ids',
        #       shapes={
        #           'encoder_input_tokens': [1, 512],
        #           'decoder_input_tokens': [1, 62],
        #           'encoder_segment_ids': [1, 512],
        #           'decoder_segment_ids': [1, 62],
        #       },
        #       types={
        #           'encoder_input_tokens': jnp.int32,
        #           'decoder_input_tokens': jnp.int32,
        #           'encoder_segment_ids': jnp.int32,
        #           'decoder_segment_ids': jnp.int32
        #       }),
    )
    def test_get_initial_variables_shapes_and_types(self, shapes, types):

        from flax.core.frozen_dict import FrozenDict

        mock_transformer = mock.Mock()
        mock_transformer.config = mock.Mock()
        mock_transformer.config.hyperencoder_model = "google/t5-small-lm-adapt"
        # TODO: confirm that this mock test is reasonable
        mock_transformer.init.return_value = FrozenDict(
            {"params": {"hyper": {}}, "params_axes": {}}
        )
        mock_optimizer_def = mock.Mock()
        rng = mock.Mock()

        def mock_init(self):
            self.module = mock_transformer
            self.optimizer_def = mock_optimizer_def

        with mock.patch.object(HyperEncoderDecoderModel, "__init__", new=mock_init):
            model = HyperEncoderDecoderModel()
            model.get_initial_variables(rng, shapes, types)

        if types is None:
            encoder_input = jnp.ones(shapes["encoder_input_tokens"], dtype=jnp.float32)
            hyper_encoder_input = jnp.ones(shapes["hyper_encoder_input_tokens"], dtype=jnp.float32)
            decoder_input = jnp.ones(shapes["decoder_input_tokens"], dtype=jnp.float32)
        else:
            encoder_input = jnp.ones(
                shapes["encoder_input_tokens"], dtype=types["encoder_input_tokens"]
            )
            hyper_encoder_input = jnp.ones(
                shapes["hyper_encoder_input_tokens"], dtype=types["hyper_encoder_input_tokens"]
            )
            decoder_input = jnp.ones(
                shapes["decoder_input_tokens"], dtype=types["decoder_input_tokens"]
            )

        # Using `.assert_called_once_with` doesn't work because the simple
        # comparison it does for the array arguments fail (truth value of an array
        # is ambiguous).
        called_with = mock_transformer.init.call_args
        self.assertEqual(called_with[0][0], rng)
        np.testing.assert_allclose(called_with[0][1], encoder_input)
        np.testing.assert_allclose(called_with[0][2], hyper_encoder_input)
        np.testing.assert_allclose(called_with[0][3], decoder_input)
        np.testing.assert_allclose(called_with[0][4], decoder_input)
        # assert False

        if "encoder_positions" in shapes:
            encoder_positions = jnp.ones(
                shapes["encoder_positions"], dtype=types["encoder_positions"]
            )
            np.testing.assert_allclose(called_with[1]["encoder_positions"], encoder_positions)
        else:
            self.assertIsNone(called_with[1]["encoder_positions"])

        if "hyper_encoder_positions" in shapes:
            hyper_encoder_positions = jnp.ones(
                shapes["hyper_encoder_positions"], dtype=types["hyper_encoder_positions"]
            )
            np.testing.assert_allclose(
                called_with[1]["hyper_encoder_positions"], hyper_encoder_positions
            )
        else:
            self.assertIsNone(called_with[1]["hyper_encoder_positions"])

        if "decoder_positions" in shapes:
            decoder_positions = jnp.ones(
                shapes["decoder_positions"], dtype=types["decoder_positions"]
            )
            np.testing.assert_allclose(called_with[1]["decoder_positions"], decoder_positions)
        else:
            self.assertIsNone(called_with[1]["decoder_positions"])

        if "encoder_segment_ids" in shapes:
            encoder_positions = jnp.ones(
                shapes["encoder_segment_ids"], dtype=types["encoder_segment_ids"]
            )
            np.testing.assert_allclose(called_with[1]["encoder_segment_ids"], encoder_positions)
        else:
            self.assertIsNone(called_with[1]["encoder_segment_ids"])
        if "hyper_encoder_segment_ids" in shapes:
            hyper_encoder_positions = jnp.ones(
                shapes["hyper_encoder_segment_ids"], dtype=types["hyper_encoder_segment_ids"]
            )
            np.testing.assert_allclose(
                called_with[1]["hyper_encoder_segment_ids"], hyper_encoder_positions
            )
        else:
            self.assertIsNone(called_with[1]["hyper_encoder_segment_ids"])
        if "decoder_segment_ids" in shapes:
            decoder_segment_ids = jnp.ones(
                shapes["decoder_segment_ids"], dtype=types["decoder_segment_ids"]
            )
            np.testing.assert_allclose(called_with[1]["decoder_segment_ids"], decoder_segment_ids)
        else:
            self.assertIsNone(called_with[1]["decoder_segment_ids"])

        self.assertFalse(called_with[1]["decode"])
        self.assertFalse(called_with[1]["enable_dropout"])

    @parameterized.named_parameters(
        dict(testcase_name="no_force_decoding", prompt_with_targets=False),
        dict(testcase_name="force_decoding", prompt_with_targets=True),
    )
    def test_prompt_with_targets(self, prompt_with_targets):
        batch_size, encoder_len, max_decode_len, emb_dim, hyper_encoder_len = 2, 3, 4, 5, 2
        batch = {
            "encoder_input_tokens": np.zeros((batch_size, encoder_len), dtype=np.int32),
            "hyper_encoder_input_tokens": np.zeros((batch_size, hyper_encoder_len), dtype=np.int32),
            "decoder_input_tokens": np.full([batch_size, max_decode_len], 2, dtype=np.int32),
        }

        # These dummy logits represent the probability distribution where all the
        # probability mass is in one item (i.e., degenerate distribution). For
        # batch element 0, it is vocabulary index 3.
        # We test `_predict_step` to avoid having to define a task and its
        # vocabulary.
        dummy_logits = jnp.expand_dims(
            jnp.array([[-1e7, -1e7, -1e7, 0, -1e7], [-1e7, -1e7, -1e7, -1e7, 0]]), axis=1
        )

        mock_decode_fn = mock.Mock()
        mock_decode_fn.return_value = (
            np.full([batch_size, max_decode_len, 1], 3, dtype=np.int32),
            np.full([batch_size, 1], 1.0, dtype=np.float32),
        )

        class MockModule:
            def __init__(self):
                self.dtype = jnp.float32
                self.config = HyperT5Config(vocab_size=100, dtype=jnp.float32)

            def apply(self, *args, method=None, **kwargs):
                del args, kwargs
                if method is None:  # use for module.`__call__`
                    return (dummy_logits, {"cache": {}})
                else:
                    return method()

            def encode(self):
                # TODO: confirm
                return jnp.zeros((batch_size, encoder_len, emb_dim))

            def decode(self):
                return (dummy_logits, {"cache": {}})

            def hyperencode(self):
                # TODO: confirm
                return {
                    "adapter_wd": jnp.zeros((batch_size, hyper_encoder_len)),
                    "adapter_wu": jnp.zeros((batch_size, hyper_encoder_len)),
                    "adapter_bd": jnp.zeros((batch_size, hyper_encoder_len)),
                    "adapter_bu": jnp.zeros((batch_size, hyper_encoder_len)),
                    "prefix_key": jnp.zeros((batch_size, hyper_encoder_len)),
                    "prefix_value": jnp.zeros((batch_size, hyper_encoder_len)),
                    "prefix_key_cc": jnp.zeros((batch_size, hyper_encoder_len)),
                    "prefix_value_cc": jnp.zeros((batch_size, hyper_encoder_len)),
                }

        def mock_init(self):
            self.module = MockModule()
            self.module.scan_layers = False
            self._input_vocabulary = mock.Mock(eos_id=1)
            self._output_vocabulary = mock.Mock(eos_id=1)
            self._decode_fn = mock_decode_fn

        with mock.patch.object(HyperEncoderDecoderModel, "__init__", new=mock_init):
            model = HyperEncoderDecoderModel()

        model.predict_batch_with_aux({}, batch, prompt_with_targets=prompt_with_targets)

        if prompt_with_targets:
            expected_inputs = batch["decoder_input_tokens"]
        else:
            expected_inputs = np.zeros([batch_size, max_decode_len], dtype=np.int32)

        assert mock_decode_fn.call_count == 1
        # Look at the kwargs call list for inputs, assert_called_with doesn't
        # work well with np.array comparison.
        np.testing.assert_array_equal(mock_decode_fn.mock_calls[0][2]["inputs"], expected_inputs)

    def test_predict_batch_loop_and_caches_are_equal(self):
        vocab_size = 50
        lengths = np.array([[2], [3]])
        batch_size, beam_size, encoder_len, max_decode_len, hyper_encoder_len = 2, 2, 3, 7, 3
        batch = {
            "encoder_input_tokens": np.zeros((batch_size, encoder_len), dtype=np.int32),
            "hyper_encoder_input_tokens": np.zeros((batch_size, hyper_encoder_len), dtype=np.int32),
            "decoder_target_tokens": np.zeros((batch_size, encoder_len), dtype=np.int32),
            "decoder_input_tokens": np.concatenate(
                [
                    np.expand_dims(
                        np.concatenate(
                            [
                                [0],
                                np.arange(9, 9 + lengths[0][0], dtype=np.int32),
                                np.zeros((max_decode_len - lengths[0][0] - 1), dtype=np.int32),
                            ]
                        ),
                        axis=0,
                    ),  # First element
                    np.expand_dims(
                        np.concatenate(
                            [
                                [0],
                                np.arange(3, 3 + lengths[1][0], dtype=np.int32),
                                np.zeros((max_decode_len - lengths[1][0] - 1), dtype=np.int32),
                            ]
                        ),
                        axis=0,
                    ),  # Second element
                ],
                axis=0,
            ),
        }

        model = get_test_model(
            emb_dim=8, head_dim=3, num_heads=4, mlp_dim=16, vocab_size=50
        )  # test_utils.get_t5_test_model(vocab_size=50)
        module = model.module
        params = module.init(
            jax.random.PRNGKey(0),
            jnp.ones((batch_size, encoder_len)),
            jnp.ones((batch_size, hyper_encoder_len)),
            jnp.ones((batch_size, max_decode_len)),
            jnp.ones((batch_size, max_decode_len)),
            enable_dropout=False,
        )["params"]

        def mock_init(self):
            self.module = module
            # Set the EOS token to be larger then the vocabulary size. This forces the
            # model to decode all the way to `max_decode_length`, allowing us to test
            # behavior when one element reaches the end before the others.
            self._output_vocabulary = mock.Mock(eos_id=vocab_size + 12)
            self._decode_fn = decoding.beam_search

        with mock.patch.object(HyperEncoderDecoderModel, "__init__", new=mock_init):
            model = HyperEncoderDecoderModel()

        with mock.patch.object(
            model, "_compute_logits_from_slice", autospec=True
        ) as tokens_to_logits_mock:
            # Make the side effect of the mock, call the method on the class, with the
            # instance partialed in as `self`. This lets us call the actual code,
            # while recording the inputs, without an infinite loop you would get
            # calling `instance.method`
            tokens_to_logits_mock.side_effect = functools.partial(
                HyperEncoderDecoderModel._compute_logits_from_slice, model
            )
            # Disable jit, so that the `lax.while_loop` isn't traced, as the
            # collection of tracers in the mock call_args would generally trigger a
            # tracer leak error.
            with jax.disable_jit():
                _ = model.predict_batch_with_aux(
                    params, batch, prompt_with_targets=True, num_decodes=2
                )

        # Collect all the input tokens to our tokens_to_logits function
        all_inputs = []
        all_cache_keys = []  # Collect all the cache keys
        all_cache_values = []  # Collect all the cache values
        # Currently force decoding generates logits at every step. We should have
        # `max_decode_length` calls to our tokens -> logits func.
        self.assertLen(tokens_to_logits_mock.call_args_list, max_decode_len)
        for tokens_call in tokens_to_logits_mock.call_args_list:
            # Inputs: [B * Be, 1]
            decoding_state = tokens_call[0][0]
            inputs = decoding_state.cur_token
            cache = decoding_state.cache
            cache = flax.core.unfreeze(cache)
            # Cache: [B * Be, 1] * #Layers
            cache_keys = [
                v for k, v in traverse_util.flatten_dict(cache).items() if k[-1] == "cached_key"
            ]
            cache_values = [
                v for k, v in traverse_util.flatten_dict(cache).items() if k[-1] == "cached_value"
            ]
            all_inputs.append(inputs)
            all_cache_keys.append(cache_keys)
            all_cache_values.append(cache_values)
        # Convert inputs to a single block [B, DL, Be]
        all_inputs = np.concatenate(all_inputs, axis=1)
        # Convert caches into a single block per layer [B * Be, DL] * L
        all_cache_keys = [np.stack(c, axis=1) for c in zip(*all_cache_keys)]
        all_cache_values = [np.stack(c, axis=1) for c in zip(*all_cache_values)]

        # Make sure that for each batch, the cache for each beam is identical when
        # prompt is being forced.
        for b in range(batch_size):
            for i, input_token in enumerate(all_inputs[b * beam_size]):
                if i < lengths[b]:
                    self.assertEqual(input_token, batch["decoder_input_tokens"][b][i])
                    # For all layers.
                    for cache_keys in all_cache_keys:
                        np.testing.assert_array_equal(
                            cache_keys[b * beam_size][i], cache_keys[b * beam_size + 1][i]
                        )
                    for cache_values in all_cache_values:
                        np.testing.assert_array_equal(
                            cache_values[b * beam_size][i], cache_values[b * beam_size + 1][i]
                        )

    def test_score_batch(self):
        encoder_input_tokens = jnp.ones((2, 3))
        hyper_encoder_input_tokens = jnp.ones((2, 3))
        # For this test, decoder input and target tokens are dummy values.
        decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
        decoder_target_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
        decoder_loss_weights = jnp.array([[1, 1, 1, 0], [0, 1, 0, 1]])
        logits = jnp.arange(0, 24).reshape((2, 4, 3))
        params = {"foo": jnp.zeros(3)}

        mock_transformer = mock.Mock()
        mock_transformer.apply.return_value = logits
        mock_transformer.dtype = jnp.float32

        batch = {
            "encoder_input_tokens": encoder_input_tokens,
            "hyper_encoder_input_tokens": hyper_encoder_input_tokens,
            "decoder_input_tokens": decoder_input_tokens,
            "decoder_target_tokens": decoder_target_tokens,
            "decoder_loss_weights": decoder_loss_weights,
        }

        def mock_init(self):
            self.module = mock_transformer

        with mock.patch.object(HyperEncoderDecoderModel, "__init__", new=mock_init):
            model = HyperEncoderDecoderModel()
            res = model.score_batch(params, batch)

        mock_transformer.apply.assert_called_with(
            {"params": params},
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
            decode=False,
            enable_dropout=False,
            rngs=None,
            mutable=False,
        )
        np.testing.assert_allclose(res, [-3.222973, -1.815315], rtol=1e-4)

    def test_score_batch_can_return_intermediates(self):
        encoder_input_tokens = jnp.ones((2, 3))
        hyper_encoder_input_tokens = jnp.ones((2, 3))
        # For this test, decoder input and target tokens are dummy values.
        decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
        decoder_target_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
        decoder_loss_weights = jnp.array([[1, 1, 1, 0], [0, 1, 0, 1]])
        logits = jnp.arange(0, 24).reshape((2, 4, 3))
        modified_variables = {"intermediates": {"bar": jnp.ones(5)}}
        params = {"foo": jnp.zeros(3)}

        mock_transformer = mock.Mock()
        mock_transformer.apply.return_value = (logits, modified_variables)
        mock_transformer.dtype = jnp.float32

        batch = {
            "encoder_input_tokens": encoder_input_tokens,
            "hyper_encoder_input_tokens": hyper_encoder_input_tokens,
            "decoder_input_tokens": decoder_input_tokens,
            "decoder_target_tokens": decoder_target_tokens,
            "decoder_loss_weights": decoder_loss_weights,
        }

        def mock_init(self):
            self.module = mock_transformer

        with mock.patch.object(HyperEncoderDecoderModel, "__init__", new=mock_init):
            model = HyperEncoderDecoderModel()
            scores, intermediates = model.score_batch(params, batch, return_intermediates=True)

        mock_transformer.apply.assert_called_with(
            {"params": params},
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
            decode=False,
            enable_dropout=False,
            rngs=None,
            mutable=["intermediates"],
        )
        np.testing.assert_allclose(scores, [-3.222973, -1.815315], rtol=1e-4)
        # Incumbent intermediates are passed out unchanged.
        np.testing.assert_allclose(intermediates["bar"], jnp.ones(5))
        # A new collection of decoder intermediates are inserted by score_batch()
        np.testing.assert_allclose(
            intermediates["decoder"]["loss_weights"][0], decoder_loss_weights
        )
        np.testing.assert_allclose(
            intermediates["decoder"]["target_tokens"][0], decoder_target_tokens
        )

    # TODO: fix mock test and rerun.
    # @parameterized.named_parameters(
    #     dict(
    #         testcase_name="int32",
    #         batch={
    #             "encoder_input_tokens": np.zeros((BATCH_SIZE, ENCODER_LEN), dtype=np.int32),
    #             "hyper_encoder_input_tokens": np.zeros((BATCH_SIZE, HYPER_ENCODER_LEN), dtype=np.int32),
    #             "decoder_input_tokens": np.zeros((BATCH_SIZE, MAX_DECODE_LEN), dtype=np.int32),
    #         },
    #     ),
    #     dict(
    #         testcase_name="float32",
    #         batch={
    #             "encoder_input_tokens": np.zeros((BATCH_SIZE, ENCODER_LEN), dtype=np.float32),
    #             "hyper_encoder_input_tokens": np.zeros((BATCH_SIZE, HYPER_ENCODER_LEN), dtype=np.int32),
    #             "decoder_input_tokens": np.zeros((BATCH_SIZE, MAX_DECODE_LEN), dtype=np.float32),
    #         },
    #     ),
    # )
    # def test_predict_batch_fake_input_shapes_and_types(self, batch):

    #     # These dummy logits represent the probability distribution where all the
    #     # probability mass is in one item (i.e., degenerate distribution). For
    #     # batch element 0, it is vocabulary index 2.
    #     # We test `_predict_step` to avoid having to define a task and its
    #     # vocabulary.
    #     dummy_logits = jnp.ones((2, 1, 4), jnp.float32)

    #     class MockModule:
    #         def __init__(self):
    #             self.dtype = jnp.float32
    #             self.call_args_list = []

    #         def apply(self, *args, method=None, **kwargs):
    #             # Not sure why this isn't a real Mock so just record the args/kwargs
    #             self.call_args_list.append({"args": args, "kwargs": kwargs})
    #             del args, kwargs
    #             if method is None:  # use for module.`__call__`
    #                 return (dummy_logits, {"cache": {}})
    #             else:
    #                 return method()

    #         def encode(self):
    #             return jnp.zeros((BATCH_SIZE, ENCODER_LEN, EMBED_DIM))

    #         def decode(self):
    #             return (dummy_logits, {"cache": {}})

    #         def hyperencode(self):
    #             # TODO: fix size, and run test!
    #             return [jnp.zeros((BATCH_SIZE, HYPER_ENCODER_LEN)) for _ in range(6)]

    #     def mock_init(self):
    #         self.module = MockModule()
    #         self.module.scan_layers = False
    #         self._input_vocabulary = mock.Mock(eos_id=1)
    #         self._output_vocabulary = mock.Mock(eos_id=1)
    #         self._decode_fn = decoding.beam_search
    #         self._inputs_bidirectional_attention = False

    #     with mock.patch.object(HyperEncoderDecoderModel, "__init__", new=mock_init):
    #         model = HyperEncoderDecoderModel()
    #     model.predict_batch_with_aux({}, batch)

    #     fake_inputs = jnp.ones_like(batch["encoder_input_tokens"])
    #     fake_hyper_inputs = jnp.ones_like(batch["hyper_encoder_input_tokens"])
    #     fake_target = jnp.ones_like(batch["decoder_input_tokens"])

    #     cache_init_call = model.module.call_args_list[0]
    #     self.assertEqual(cache_init_call["args"][0], {"params": {}})
    #     np.testing.assert_allclose(cache_init_call["args"][1], fake_inputs)
    #     np.testing.assert_allclose(cache_init_call["args"][2], fake_hyper_inputs)
    #     np.testing.assert_allclose(cache_init_call["args"][3], fake_target)
    #     np.testing.assert_allclose(cache_init_call["args"][4], fake_target)
    #     self.assertEqual(
    #         cache_init_call["kwargs"],
    #         {"decode": True, "enable_dropout": False, "mutable": ["cache"]},
    #     )
