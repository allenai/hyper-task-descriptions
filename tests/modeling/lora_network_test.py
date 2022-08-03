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

    def test_t5_1_1_lora(self):
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
            do_lora=True,
            lora_ranks=(4, None, 4, None),
            lora_hyper_gen=False,
        )
        params = model.get_initial_variables(jax.random.PRNGKey(42), self.input_shapes)["params"]

        assert "lora_qa_gen" not in params["hyper"]
        assert "lora_a" in params["encoder"]["layers_0"]["attention"]["query"]
        assert "lora_a" not in params["encoder"]["layers_0"]["attention"]["key"]
        assert "lora_a" in params["encoder"]["layers_0"]["attention"]["value"]
        assert "lora_a" not in params["encoder"]["layers_0"]["attention"]["out"]

        loss, _ = jax.jit(model.loss_fn)(params, batch, jax.random.PRNGKey(1))
        self.assertAlmostEqual(loss, 17.351202, delta=0.05)

        predicted, scores = model.predict_batch_with_aux(params, batch)
        # predicted.shape = 2 x 3 (batch_size x max_decode_len) (best option)
        np.testing.assert_array_equal(predicted, [[8, 2, 8], [1, 0, 0]])
        # scores.shape = 2 (batch_size) (best option)
        np.testing.assert_allclose(scores["scores"], [-4.49669, -2.08134], rtol=1e-3)

    def test_t5_1_1_hyper_lora(self):
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
            do_lora=True,
            lora_hyper_gen=True,
            lora_ranks=(4, None, 4, None),
        )
        params = model.get_initial_variables(jax.random.PRNGKey(42), self.input_shapes)["params"]

        assert "lora_a" not in params["encoder"]["layers_0"]["attention"]["query"]
        assert "lora_qa_gen" in params["hyper"]
        assert "lora_ka_gen" not in params["hyper"]
        assert "lora_va_gen" in params["hyper"]
        assert "lora_oa_gen" not in params["hyper"]

        loss, _ = jax.jit(model.loss_fn)(params, batch, jax.random.PRNGKey(1))
        self.assertAlmostEqual(loss, 17.351202, delta=0.05)

        predicted, scores = model.predict_batch_with_aux(params, batch)
        # predicted.shape = 2 x 3 (batch_size x max_decode_len) (best option)
        np.testing.assert_array_equal(predicted, [[8, 2, 8], [1, 0, 0]])
        # scores.shape = 2 (batch_size) (best option)
        np.testing.assert_allclose(scores["scores"], [-4.49669, -2.08134], rtol=1e-3)


# if __name__ == "__main__":
#     nt = NetworkTest()
#     nt.setUp()
#     nt.test_t5_1_1_regression()
