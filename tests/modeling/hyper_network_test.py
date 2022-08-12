import jax
import numpy as np
from absl.testing import parameterized

from hyper_task_descriptions.common.testing import get_test_model


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
        self.assertAlmostEqual(loss, 16.475815, delta=0.05)

        predicted, scores = model.predict_batch_with_aux(params, batch)
        # predicted.shape = 2 x 3 (batch_size x max_decode_len) (best option)
        np.testing.assert_array_equal(predicted, [[2, 6, 1], [2, 6, 5]])
        # scores.shape = 2 (batch_size) (best option)
        np.testing.assert_allclose(scores["scores"], [-3.5013323, -2.8256376], rtol=1e-3)
