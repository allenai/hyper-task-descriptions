import os

os.environ["JAX_DISABLE_JIT"] = "true"
import os

import jax.numpy as jnp
import numpy as np
import optax
from t5x import partitioning, utils

from hyper_task_descriptions import utils as hyper_utils
from hyper_task_descriptions.hf_vocab import HuggingfaceVocabulary
from hyper_task_descriptions.modeling.hyper_interactive_model import (
    HyperInteractiveModel,
)
from hyper_task_descriptions.modeling.hyper_network import (
    HyperT5Config,
    HyperTransformer,
)
from hyper_task_descriptions.modeling.hyper_transformer import (
    HyperEncoderDecoderContrastiveModel,
)

# You'll need permissions to access the checkpoint
checkpoint_path = "checkpoint_1109000"
dtype = "bfloat16"
restore_mode = "specific"

# partitioning
partitioner = partitioning.PjitPartitioner(num_partitions=1, model_parallel_submesh=None)

# Setup the config
module = HyperTransformer(
    config=HyperT5Config(
        vocab_size=32128,
        dtype=jnp.float32,
        emb_dim=2048,
        num_heads=32,
        num_encoder_layers=24,
        num_decoder_layers=24,
        head_dim=64,
        mlp_dim=5120,
        mlp_activations=("gelu", "linear"),
        dropout_rate=0.0,
        layer_embedding_method="component",
        use_adapter=True,
        use_prefix=True,
        adapter_size=64,
        num_prefix_tokens=15,
        hyperencoder_model="google/t5-large-lm-adapt",
    )
)

model = HyperEncoderDecoderContrastiveModel(
    module=module,
    input_vocabulary=HuggingfaceVocabulary("t5-base"),
    output_vocabulary=HuggingfaceVocabulary("t5-base"),
    optimizer_def=hyper_utils.multi_transform(
        transforms={
            "hyper": optax.adam(
                learning_rate=utils.create_learning_rate_scheduler(
                    base_learning_rate=1e-2, decay_factor=0
                )
            ),
            "freeze": optax.adam(
                learning_rate=utils.create_learning_rate_scheduler(
                    base_learning_rate=1e-4, decay_factor=0
                )
            ),
            "roberta": optax.adam(
                learning_rate=utils.create_learning_rate_scheduler(
                    base_learning_rate=1e-4, decay_factor=0
                )
            ),
        },
        param_labels=hyper_utils.match_any_optax_trip([".*hyper/[eap].*"], [".*hyper.*"]),
    ),
)

batch_size = 1
task_feature_lengths = {"inputs": 38, "targets": 18, "hyper_inputs": 38, "task_names": 1}
output_dir = "tmp_output_dir"
input_shapes = {
    "encoder_input_tokens": np.array([1, 38]),
    "hyper_encoder_input_tokens": np.array([1, 38]),
    "decoder_target_tokens": np.array([1, 18]),
    "decoder_input_tokens": np.array([1, 18]),
    "decoder_loss_weights": np.array([1, 18]),
    "task_names": np.array([1, 1]),
}

interactive_model = HyperInteractiveModel(
    batch_size=batch_size,
    task_feature_lengths=task_feature_lengths,
    output_dir=output_dir,
    partitioner=partitioner,
    model=model,
    dtype=dtype,
    restore_mode=restore_mode,
    checkpoint_path=checkpoint_path,
    input_shapes=input_shapes,
)

test_examples = [
    {
        "input": "What is the capital of France?",
        "hyper_input": "The captial of france is",
        "target": "Paris",
        "task_name": [0],
    }
]


examples_and_predictions, scores = interactive_model.predict_with_aux(examples=test_examples)
predictions = [prediction for _, prediction in examples_and_predictions]
print(f"Predictions: {predictions}\n")
print(f"Scores: {scores}\n")
