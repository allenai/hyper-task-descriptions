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
from hyper_task_descriptions.modeling.hyper_network import HyperT5Config
from hyper_task_descriptions.modeling.lora_network import (
    LoraDecoder,
    LoraDecoderLayer,
    LoraEncoder,
    LoraEncoderLayer,
    LoraTransformer,
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
            lora_ranks=(4, None, 4, None),
            lora_hyper_gen=False,
            use_prefix=False,
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
        lora_encoder_layer = LoraEncoderLayer(config=config, relative_embedding=rel_emb)
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
        lora_encoder = LoraEncoder(config=config, shared_embedding=shared_embedding)
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
        lora_decoder_layer = LoraDecoderLayer(config=config, relative_embedding=rel_emb)
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
        lora_decoder = LoraDecoder(config=config, shared_embedding=shared_embedding)
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
            # "hyper_encoder_input_tokens": np.random.randint(
            #     3, 10, size=(batch_size, hyper_input_len)
            # ),
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
            lora_ranks=(4, None, 4, None),
            lora_hyper_gen=False,
            use_prefix=False,
        )

        encoder_shape = self.input_shapes["encoder_input_tokens"]
        decoder_shape = self.input_shapes["decoder_input_tokens"]

        module = LoraTransformer(config=config)
        params = module.init(
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

        voutput = vmodule.apply(vparams, **batch)
        assert (output == voutput).all()

    def test_t5_1_1_lora(self):
        np.random.seed(0)
        batch_size, max_decode_len, input_len = 2, 3, 4
        batch = {
            "encoder_input_tokens": np.random.randint(3, 10, size=(batch_size, input_len)),
            # "hyper_encoder_input_tokens": np.random.randint(
            #     3, 10, size=(batch_size, hyper_input_len)
            # ),
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

        # assert "lora_qa_gen" not in params["hyper"]
        assert "hyper" not in params
        assert "lora_a" in params["encoder"]["layers_0"]["attention"]["query"]
        assert "lora_a" not in params["encoder"]["layers_0"]["attention"]["key"]
        assert "lora_a" in params["encoder"]["layers_0"]["attention"]["value"]
        assert "lora_a" not in params["encoder"]["layers_0"]["attention"]["out"]

        loss, _ = jax.jit(model.loss_fn)(params, batch, jax.random.PRNGKey(1))
        self.assertAlmostEqual(loss, 12.66808, delta=0.05)

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
        vparams = model.get_initial_variables(jax.random.PRNGKey(42), self.input_shapes)["params"]
        vloss, _ = vmodel.loss_fn(vparams, batch, jax.random.PRNGKey(1))
        self.assertAlmostEqual(vloss, 12.66808, delta=0.05)

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
        self.assertAlmostEqual(loss, 15.268721, delta=0.05)

        predicted, scores = model.predict_batch_with_aux(params, batch)
        # predicted.shape = 2 x 3 (batch_size x max_decode_len) (best option)
        np.testing.assert_array_equal(predicted, [[2, 6, 1], [2, 6, 5]])
        # scores.shape = 2 (batch_size) (best option)
        np.testing.assert_allclose(scores["scores"], [-3.501333, -2.825637], rtol=1e-3)


# if __name__ == "__main__":
#     nt = NetworkTest()
#     nt.setUp()
#     nt.test_t5_1_1_regression()
