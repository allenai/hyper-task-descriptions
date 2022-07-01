import jax
import numpy as np
import seqio
from absl.testing import parameterized
from t5x import adafactor

from hyper_task_descriptions.modeling.hyper_network import (
    HyperT5Config,
    HyperTransformer,
)
from hyper_task_descriptions.modeling.hyper_transformer import HyperEncoderDecoderModel


def get_test_model(
    emb_dim,
    head_dim,
    num_heads,
    mlp_dim,
    dtype="float32",
    vocab_size=32128,
    num_encoder_layers=2,
    num_decoder_layers=2,
):
    config = HyperT5Config(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        vocab_size=vocab_size,
        dropout_rate=0,
        emb_dim=emb_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        mlp_dim=mlp_dim,
        dtype=dtype,
        mlp_activations=("gelu", "linear"),
    )
    # TODO: maybe configure adapter specific things too.
    module = HyperTransformer(config=config)
    vocab = seqio.test_utils.sentencepiece_vocab()
    optimizer_def = adafactor.Adafactor()
    return HyperEncoderDecoderModel(module, vocab, vocab, optimizer_def=optimizer_def)


class NetworkTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        batch_size, max_decode_len, input_len, hyper_input_len = 2, 3, 4, 5
        self.input_shapes = {
            "encoder_input_tokens": (batch_size, input_len),
            "hyper_encoder_input_tokens": (batch_size, hyper_input_len),
            "decoder_input_tokens": (batch_size, max_decode_len),
        }
        np.random.seed(42)
        self.batch = {
            "encoder_input_tokens": np.random.randint(3, 10, size=(batch_size, input_len)),
            "hyper_encoder_input_tokens": np.random.randint(
                3, 10, size=(batch_size, hyper_input_len)
            ),
            "decoder_input_tokens": np.random.randint(3, 10, size=(batch_size, max_decode_len)),
            "decoder_target_tokens": np.random.randint(3, 10, size=(batch_size, max_decode_len)),
        }

    def test_t5_1_1_regression(self):
        np.random.seed(0)
        batch_size, max_decode_len, input_len, hyper_input_len = 2, 3, 4, 5
        batch = {
            "encoder_input_tokens": np.random.randint(3, 10, size=(batch_size, input_len)),
            "hyper_encoder_input_tokens": np.random.randint(
                3, 10, size=(batch_size, hyper_input_len)
            ),
            "decoder_input_tokens": np.random.randint(3, 10, size=(batch_size, max_decode_len)),
            "decoder_target_tokens": np.random.randint(3, 10, size=(batch_size, max_decode_len)),
        }
        model = get_test_model(
            emb_dim=13,
            head_dim=16,
            num_heads=8,
            mlp_dim=32,
            vocab_size=10,
            num_encoder_layers=1,
            num_decoder_layers=1,
        )
        params = model.get_initial_variables(jax.random.PRNGKey(42), self.input_shapes)["params"]
        loss, _ = jax.jit(model.loss_fn)(params, batch, jax.random.PRNGKey(1))
        self.assertAlmostEqual(loss, 15.268721, delta=0.05)

        predicted, scores = model.predict_batch_with_aux(params, batch)
        # predicted.shape = 2 x 3 (batch_size x max_decode_len) (best option)
        np.testing.assert_array_equal(predicted, [[2, 6, 1], [2, 6, 5]])
        # scores.shape = 2 (batch_size) (best option)
        np.testing.assert_allclose(scores["scores"], [-3.5013323, -2.8256376], rtol=1e-3)
