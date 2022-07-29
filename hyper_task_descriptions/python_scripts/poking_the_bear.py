"""
Export the roberta model from the hypernetwork into a roberta huggingface model.
"""

import argparse

from jax import nn
from jax import numpy as jnp
from t5x import checkpoints
from transformers import FlaxRobertaModel, RobertaModel, RobertaTokenizer

from hyper_task_descriptions.modeling.losses import cosine_similarity


def extract_roberta_model(t5x_checkpoint_path, flax_dump_folder_path):
    t5x_model = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    roberta_model = FlaxRobertaModel.from_pretrained("hamishivi/fixed-roberta-base")
    roberta_model.params = t5x_model["target"]["hyper"]["encoder"]
    roberta_model.save_pretrained(flax_dump_folder_path)
    model = RobertaModel.from_pretrained(flax_dump_folder_path, from_flax=True)
    model.save_pretrained(flax_dump_folder_path + "_pytorch")


def get_contrastive_head(model, t5x_model, tok, text):
    outputs = model(tok(text, return_tensors="np")["input_ids"])[0].mean(axis=1)
    out1 = outputs @ t5x_model["target"]["hyper"]["contrastive_head"]["wi"]["kernel"]
    out1 = nn.gelu(out1)
    out1 = out1 @ t5x_model["target"]["hyper"]["contrastive_head"]["wo"]["kernel"]
    return out1


def get_bottleneck_representation(model, t5x_model, tok, text, layer_index):
    outputs = model(tok(text, return_tensors="np")["input_ids"])[0].mean(axis=1).squeeze(0)
    layer_embeds = t5x_model["target"]["hyper"]["embedding"]
    outputs = jnp.concatenate([outputs, layer_embeds[layer_index]], axis=0)
    out1 = outputs @ t5x_model["target"]["hyper"]["intermediate_hypernet"]["wi"]["kernel"]
    out1 = nn.gelu(out1)
    return out1


def get_adapter_down_params(model, t5x_model, tok, text, layer_index):
    outputs = model(tok(text, return_tensors="np")["input_ids"])[0].mean(axis=1).squeeze(0)
    layer_embeds = t5x_model["target"]["hyper"]["embedding"]
    outputs = jnp.concatenate([outputs, layer_embeds[layer_index]], axis=0)
    out1 = outputs @ t5x_model["target"]["hyper"]["intermediate_hypernet"]["wi"]["kernel"]
    out1 = nn.gelu(out1)
    out1 = out1 @ t5x_model["target"]["hyper"]["adapter_down_mlp"]["wi"]["kernel"]
    return out1


def play_with_model(t5x_checkpoint_path):
    tok = RobertaTokenizer.from_pretrained("roberta-base")
    t5x_model = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    roberta_model = FlaxRobertaModel.from_pretrained("hamishivi/fixed-roberta-base")
    roberta_model.params = t5x_model["target"]["hyper"]["encoder"]
    get_cont = lambda x, y: get_contrastive_head(  # noqa: E731, F841
        roberta_model, t5x_model, tok, x
    )
    get_bot = lambda x, y: get_bottleneck_representation(  # noqa: E731
        roberta_model, t5x_model, tok, x, y
    )
    get_adapter = lambda x, y: get_adapter_down_params(  # noqa: E731
        roberta_model, t5x_model, tok, x, y
    )
    for f, name in zip([get_bot, get_adapter], ["bot", "ada"]):
        print(name)
        layers = 1 if name == "cont" else 48
        print("diff task")
        for i in range(layers):
            print(
                cosine_similarity(
                    f("Given that [MASK] Does it follow that [MASK] Yes or no?", i),
                    f(
                        'The word "[MASK]" has multiple meanings. Does it have the \
                            same meaning in sentences 1 and 2? Yes or no?',
                        i,
                    ),
                )
            )
        print("same task")
        for i in range(layers):
            print(
                cosine_similarity(
                    f("Given that [MASK] Does it follow that [MASK] Yes or no?", i),
                    f('Given that [MASK] Therefore, it must be true that "[MASK]"? Yes or no?', i),
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--t5x_checkpoint_path",
        "-t",
        default=None,
        type=str,
        required=True,
        help="Path the TX5 checkpoint.",
    )
    parser.add_argument(
        "--flax_dump_folder_path",
        "-d",
        default=None,
        type=str,
        required=True,
        help="Path the TX5 checkpoint.",
    )
    args = parser.parse_args()
    # extract_roberta_model(args.t5x_checkpoint_path, args.flax_dump_folder_path)
    play_with_model(args.t5x_checkpoint_path)
