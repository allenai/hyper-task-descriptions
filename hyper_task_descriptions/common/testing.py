import logging
import os
import shutil
import tempfile
from pathlib import Path

import jax


class HyperTaskDescriptionsTestCase:
    """
    A custom testing class that

    * disables some of the more verbose logging,
    * creates and destroys a temp directory as a test fixture
    """

    PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
    """
    Root of the git repository.
    """

    MODULE_ROOT = PROJECT_ROOT / "hyper_task_descriptions"
    """
    Root of the tango module.
    """

    TESTS_ROOT = PROJECT_ROOT / "tests"
    """
    Root of the tests directory.
    """

    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"
    """
    Root of the test fixtures directory.
    """

    def setup_method(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
        )

        # Disabling some of the more verbose logging statements that typically aren't very helpful
        # in tests.
        logging.getLogger("urllib3.connectionpool").disabled = True

        # Create a temporary scratch directory.
        self.TEST_DIR = Path(tempfile.mkdtemp(prefix="hyper_task_description_tests"))
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)


def get_test_model(
    emb_dim,
    head_dim,
    num_heads,
    mlp_dim,
    dtype="float32",
    vocab_size=32128,
    num_prefix_tokens=1,
    num_encoder_layers=2,
    num_decoder_layers=2,
    use_instructions=True,
    use_lora=False,
    lora_ranks=(None, None, None, None),
    use_ia3=False,
    ia3_ranks=(None, None, None, None),
    use_prefix=False,
    use_adapter=False,
):
    import seqio
    from t5x import adafactor

    from hyper_task_descriptions.modeling.hyper_network import (
        HyperT5Config,
        HyperTransformer,
    )
    from hyper_task_descriptions.modeling.hyper_transformer import (
        HyperEncoderDecoderModel,
    )

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
        num_prefix_tokens=num_prefix_tokens,
        use_lora=use_lora,
        lora_ranks=lora_ranks,
        use_ia3=use_ia3,
        ia3_ranks=ia3_ranks,
        use_prefix=use_prefix,
        use_adapter=use_adapter,
        use_instructions=use_instructions,
        hyperencoder_model="google/t5-small-lm-adapt",
    )
    module = HyperTransformer(config=config)
    vocab = seqio.test_utils.sentencepiece_vocab()
    optimizer_def = adafactor.Adafactor()
    return HyperEncoderDecoderModel(module, vocab, vocab, optimizer_def=optimizer_def)


def get_vanilla_test_model(
    emb_dim,
    head_dim,
    num_heads,
    mlp_dim,
    dtype="float32",
    vocab_size=32128,
    num_encoder_layers=2,
    num_decoder_layers=2,
):
    import seqio
    from t5x import adafactor
    from t5x.examples.t5.network import T5Config, Transformer
    from t5x.models import EncoderDecoderModel

    config = T5Config(
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
    module = Transformer(config=config)
    vocab = seqio.test_utils.sentencepiece_vocab()
    optimizer_def = adafactor.Adafactor()
    return EncoderDecoderModel(module, vocab, vocab, optimizer_def=optimizer_def)


def get_prng_key(seed: int = 23) -> jax.random.PRNGKeyArray:
    return jax.random.PRNGKey(seed)


def get_multiple_keys(key: jax.random.PRNGKeyArray, multiple: int = 1) -> jax.random.PRNGKeyArray:
    return jax.random.split(key, multiple)
