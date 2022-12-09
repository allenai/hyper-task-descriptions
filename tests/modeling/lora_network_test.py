import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from t5x.examples.t5 import layers
from t5x.examples.t5.network import T5Config, Transformer

from hyper_task_descriptions.common.testing import (
    get_test_model,
    get_vanilla_test_model,
)
from hyper_task_descriptions.modeling.hyper_network import (
    HyperDecoder,
    HyperDecoderLayer,
    HyperEncoder,
    HyperEncoderLayer,
    HyperT5Config,
    HyperTransformer,
)


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

        self.lora_config = HyperT5Config(
            emb_dim=13,
            head_dim=16,
            num_heads=8,
            mlp_dim=32,
            vocab_size=10,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout_rate=0,
            dtype="float32",
            mlp_activations=("gelu", "linear"),
            lora_ranks=(None, None, None, None),
            use_lora=False,
            use_prefix=False,
            use_adapter=False,
        )

        self.vanilla_config = T5Config(
            emb_dim=13,
            head_dim=16,
            num_heads=8,
            mlp_dim=32,
            vocab_size=10,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout_rate=0,
            dtype="float32",
            mlp_activations=("gelu", "linear"),
        )

    def test_lora_encoder_layer(self):
        from flax import linen as nn

        config = self.lora_config

        rel_emb = layers.RelativePositionBiases(
            num_buckets=32,
            max_distance=128,
            num_heads=8,
            dtype=jnp.float32,
            embedding_init=nn.initializers.variance_scaling(1.0, "fan_avg", "uniform"),
            name="relpos_bias",
        )

        batch_size, in_features = 3, 4
        inputs = jnp.array(np.random.randn(batch_size, in_features, 13))  # emb_dim = 13
        lora_encoder_layer = HyperEncoderLayer(config=config, relative_embedding=rel_emb)
        params = lora_encoder_layer.init(jax.random.PRNGKey(42), inputs)
        output = lora_encoder_layer.apply(params, inputs)
        assert output.shape == (batch_size, in_features, 13)

        # Sanity check:
        from t5x.examples.t5.network import EncoderLayer

        vconfig = self.vanilla_config
        encoder_layer = EncoderLayer(config=vconfig, relative_embedding=rel_emb)
        vparams = encoder_layer.init(jax.random.PRNGKey(42), inputs)
        voutput = encoder_layer.apply(vparams, inputs)
        assert (output == voutput).all()

    def test_lora_encoder(self):
        from flax import linen as nn

        config = self.lora_config
        shared_embedding = layers.Embed(
            num_embeddings=config.vocab_size,
            features=config.emb_dim,
            dtype=config.dtype,
            attend_dtype=jnp.float32,  # for logit training stability
            embedding_init=nn.initializers.normal(stddev=1.0),
            one_hot=True,
            name="token_embedder",
        )
        batch_size, input_len = 2, 4
        inputs = np.random.randint(3, 10, size=(batch_size, input_len))
        lora_encoder = HyperEncoder(config=config, shared_embedding=shared_embedding)
        params = lora_encoder.init(jax.random.PRNGKey(42), encoder_input_tokens=inputs)

        output = lora_encoder.apply(params, inputs)

        assert output.shape == (batch_size, input_len, config.emb_dim)

        # Sanity check:
        from t5x.examples.t5.network import Encoder

        encoder = Encoder(config=self.vanilla_config, shared_embedding=shared_embedding)
        vparams = encoder.init(jax.random.PRNGKey(42), encoder_input_tokens=inputs)
        voutput = encoder.apply(vparams, inputs)

        assert (output == voutput).all()

    def test_lora_decoder_layer(self):
        from flax import linen as nn

        config = self.lora_config

        rel_emb = layers.RelativePositionBiases(
            num_buckets=32,
            max_distance=128,
            num_heads=8,
            dtype=jnp.float32,
            embedding_init=nn.initializers.variance_scaling(1.0, "fan_avg", "uniform"),
            name="relpos_bias",
        )

        batch_size, in_features = 3, 4
        inputs = jnp.array(np.random.randn(batch_size, in_features, 13))  # emb_dim = 13
        encoded = jnp.array(np.random.randn(batch_size, in_features, 13))
        lora_decoder_layer = HyperDecoderLayer(config=config, relative_embedding=rel_emb)
        params = lora_decoder_layer.init(jax.random.PRNGKey(42), inputs, encoded=encoded)
        output = lora_decoder_layer.apply(params, inputs, encoded)
        assert output.shape == (batch_size, in_features, 13)

        # Sanity check:
        from t5x.examples.t5.network import DecoderLayer

        vconfig = self.vanilla_config
        decoder_layer = DecoderLayer(config=vconfig, relative_embedding=rel_emb)
        vparams = decoder_layer.init(jax.random.PRNGKey(42), inputs, encoded=encoded)
        voutput = decoder_layer.apply(vparams, inputs, encoded)
        assert (output == voutput).all()

    def test_lora_decoder(self):
        np.random.seed(0)
        from flax import linen as nn

        config = self.lora_config
        shared_embedding = layers.Embed(
            num_embeddings=config.vocab_size,
            features=config.emb_dim,
            dtype=config.dtype,
            attend_dtype=jnp.float32,  # for logit training stability
            embedding_init=nn.initializers.normal(stddev=1.0),
            one_hot=True,
            name="token_embedder",
        )
        batch_size, max_decode_len, input_len = 2, 3, 4
        decoder_input_tokens = np.random.randint(3, 10, size=(batch_size, max_decode_len))
        encoded = jnp.array(np.random.randn(batch_size, input_len, config.emb_dim))
        lora_decoder = HyperDecoder(config=config, shared_embedding=shared_embedding)
        params = lora_decoder.init(
            jax.random.PRNGKey(42), encoded=encoded, decoder_input_tokens=decoder_input_tokens
        )

        output = lora_decoder.apply(params, encoded, decoder_input_tokens)

        # assert output.shape == (batch_size, max_decode_len, config.vocab_size)

        # Sanity check:
        from t5x.examples.t5.network import Decoder

        decoder = Decoder(config=self.vanilla_config, shared_embedding=shared_embedding)
        vparams = decoder.init(
            jax.random.PRNGKey(42), encoded=encoded, decoder_input_tokens=decoder_input_tokens
        )
        voutput = decoder.apply(vparams, encoded, decoder_input_tokens)

        assert (output == voutput).all()

    def test_lora_transformer(self):
        np.random.seed(0)
        batch_size, max_decode_len, input_len = 2, 3, 4
        batch = {
            "encoder_input_tokens": np.random.randint(3, 10, size=(batch_size, input_len)),
            "hyper_encoder_input_tokens": np.random.randint(3, 10, size=(batch_size, 4)),
            "decoder_input_tokens": np.random.randint(3, 10, size=(batch_size, max_decode_len)),
            "decoder_target_tokens": np.random.randint(3, 10, size=(batch_size, max_decode_len)),
        }

        config = HyperT5Config(
            emb_dim=13,
            head_dim=16,
            num_heads=8,
            mlp_dim=32,
            vocab_size=10,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dropout_rate=0,
            dtype="float32",
            mlp_activations=("gelu", "linear"),
            lora_ranks=(None, None, None, None),
            use_adapter=False,
            use_lora=False,
            use_prefix=False,
        )

        encoder_shape = self.input_shapes["encoder_input_tokens"]
        hyper_encoder_shape = self.input_shapes["hyper_encoder_input_tokens"]
        decoder_shape = self.input_shapes["decoder_input_tokens"]

        module = HyperTransformer(config=config)
        params = module.init(
            jax.random.PRNGKey(42),
            jnp.ones(encoder_shape, jnp.float32),
            jnp.ones(hyper_encoder_shape, jnp.float32),
            jnp.ones(decoder_shape, jnp.float32),
            jnp.ones(decoder_shape, jnp.float32),
            encoder_positions=None,
            decoder_positions=None,
            encoder_segment_ids=None,
            decoder_segment_ids=None,
            decode=False,
            enable_dropout=False,
        )

        output = module.apply(params, **batch)
        assert output.shape == (batch_size, max_decode_len, 10)

        # Sanity check
        vconfig = T5Config(
            emb_dim=13,
            head_dim=16,
            num_heads=8,
            mlp_dim=32,
            vocab_size=10,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dropout_rate=0,
            dtype="float32",
            mlp_activations=("gelu", "linear"),
        )

        vmodule = Transformer(config=vconfig)
        vparams = vmodule.init(
            jax.random.PRNGKey(42),
            jnp.ones(encoder_shape, jnp.float32),
            jnp.ones(decoder_shape, jnp.float32),
            jnp.ones(decoder_shape, jnp.float32),
            encoder_positions=None,
            decoder_positions=None,
            encoder_segment_ids=None,
            decoder_segment_ids=None,
            decode=False,
            enable_dropout=False,
        )

        batch.pop("hyper_encoder_input_tokens")
        voutput = vmodule.apply(vparams, **batch)
        assert (output == voutput).all()

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
            lora_ranks=(None, None, None, None),
            use_lora=False,
            use_adapter=False,
            use_prefix=False,
            use_instructions=False,
        )
        params = model.get_initial_variables(jax.random.PRNGKey(42), self.input_shapes)["params"]

        # assert "lora_qa_gen" not in params["hyper"]
        assert "hyper" in params
        assert "lora_a" not in params["encoder"]["layers_0"]["attention"]["key"]
        assert "lora_a" not in params["encoder"]["layers_0"]["attention"]["out"]

        loss, _ = jax.jit(model.loss_fn)(params, batch, jax.random.PRNGKey(1))
        self.assertAlmostEqual(loss, 15.268721, delta=0.05)

        predicted, scores = model.predict_batch_with_aux(params, batch)
        # predicted.shape = 2 x 3 (batch_size x max_decode_len) (best option)
        np.testing.assert_array_equal(predicted, [[2, 6, 1], [2, 6, 5]])
        # scores.shape = 2 (batch_size) (best option)
        np.testing.assert_allclose(scores["scores"], [-3.501333, -2.825637], rtol=1e-3)

        # Sanity check
        vmodel = get_vanilla_test_model(
            emb_dim=13,
            head_dim=16,
            num_heads=8,
            mlp_dim=32,
            vocab_size=10,
            num_encoder_layers=1,
            num_decoder_layers=1,
        )
        input_shapes = self.input_shapes
        input_shapes.pop("hyper_encoder_input_tokens")
        vparams = vmodel.get_initial_variables(jax.random.PRNGKey(42), input_shapes)["params"]
        vloss, _ = vmodel.loss_fn(vparams, batch, jax.random.PRNGKey(1))
        self.assertAlmostEqual(vloss, 15.268721, delta=0.05)

    @parameterized.named_parameters(
        dict(testcase_name="no_use_prefix", use_prefix=False),
        dict(testcase_name="use_prefix", use_prefix=True),
    )
    def test_t5_1_1_hyper_lora(self, use_prefix: bool):
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
            num_prefix_tokens=1,
            use_lora=True,
            lora_ranks=(
                4,
                None,
                4,
                None,
            ),
            use_prefix=use_prefix,
        )
        params = model.get_initial_variables(jax.random.PRNGKey(42), self.input_shapes)["params"]

        assert "lora_qa" in params["hyper"]
        assert "lora_va" in params["hyper"]

        # Check initialization
        assert params["hyper"]["lora_qb"]["wi"]["kernel"].sum() == 0

        if use_prefix:
            assert "prefix_value" in params["hyper"]
            assert "prefix_key" in params["hyper"]
        else:
            assert "prefix_value" not in params["hyper"]
            assert "prefix_key" not in params["hyper"]

        loss, _ = jax.jit(model.loss_fn)(params, batch, jax.random.PRNGKey(1))
        if use_prefix:
            self.assertAlmostEqual(loss, 15.374477, delta=0.05)
        else:
            self.assertAlmostEqual(loss, 15.293617, delta=0.05)

        predicted, scores = model.predict_batch_with_aux(params, batch)
        # predicted.shape = 2 x 3 (batch_size x max_decode_len) (best option)
        if use_prefix:
            np.testing.assert_array_equal(predicted, [[9, 3, 3], [2, 6, 1]])
        else:
            np.testing.assert_array_equal(predicted, [[2, 6, 1], [2, 6, 5]])
        # scores.shape = 2 (batch_size) (best option)
        if use_prefix:
            np.testing.assert_allclose(scores["scores"], [-3.765547, -3.206083], rtol=1e-3)
        else:
            np.testing.assert_allclose(scores["scores"], [-3.501333, -2.825637], rtol=1e-3)

    def test_t5_1_1_hyper_lora_ia3(self):
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
            num_prefix_tokens=1,
            use_lora=True,
            lora_ranks=(
                4,
                None,
                4,
                None,
            ),
            use_ia3=True,
            ia3_ranks=(
                None,
                4,
                4,
                None,
            ),
            use_prefix=False,
            use_adapter=False,
        )
        params = model.get_initial_variables(jax.random.PRNGKey(42), self.input_shapes)["params"]

        assert "lora_qa" in params["hyper"]
        assert "lora_va" in params["hyper"]
        assert "ia3_ka" in params["hyper"]
        assert "ia3_va" in params["hyper"]

        assert "prefix_value" not in params["hyper"]
        assert "prefix_key" not in params["hyper"]

        loss, _ = jax.jit(model.loss_fn)(params, batch, jax.random.PRNGKey(1))
        self.assertAlmostEqual(loss, 15.304657, delta=0.05)

        predicted, scores = model.predict_batch_with_aux(params, batch)
        # predicted.shape = 2 x 3 (batch_size x max_decode_len) (best option)
        np.testing.assert_array_equal(predicted, [[2, 6, 1], [2, 6, 5]])
        # scores.shape = 2 (batch_size) (best option)
        np.testing.assert_allclose(scores["scores"], [-3.501333, -2.825637], rtol=1e-4)


# if __name__ == "__main__":
#     nt = NetworkTest()
#     nt.setUp()
#     nt.test_t5_1_1_regression()
