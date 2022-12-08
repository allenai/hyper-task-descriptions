"""
Export the roberta model from the hypernetwork into a roberta huggingface model.
"""
# flake8: noqa
import argparse

import jax
from jax import numpy as jnp
from transformers import AutoTokenizer, FlaxT5EncoderModel, T5EncoderModel

from t5x import checkpoints


def extract_roberta_model(t5x_checkpoint_path, flax_dump_folder_path):
    t5x_model = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    hf_model = FlaxT5EncoderModel.from_pretrained("google/t5-large-lm-adapt")
    hf_model.params = t5x_model["target"]["hyper"]["encoder"]
    hf_model.save_pretrained(flax_dump_folder_path)
    model = T5EncoderModel.from_pretrained(flax_dump_folder_path, from_flax=True)
    model.save_pretrained(flax_dump_folder_path + "_pytorch")


def get_model_output(model, tok, text):
    return model(tok(text, return_tensors="np")["input_ids"])[0]


def get_attention_values(model, t5x_model, tok, text):
    output = model(tok(text, return_tensors="np")["input_ids"])[0][0]
    layer_embeds = t5x_model["target"]["hyper"]["component_embedding"]
    # run attn
    attn_weights = jnp.einsum("qd,kd->qk", layer_embeds, output)
    attn_weights = jax.nn.softmax(attn_weights)
    return attn_weights


def play_with_model(t5x_checkpoint_path):
    tok = AutoTokenizer.from_pretrained("t5-base")
    t5x_model = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    # hf_model = FlaxT5EncoderModel.from_pretrained("google/t5-large-lm-adapt", from_pt=True)
    # hf_model.params = t5x_model["target"]["hyper"]["encoder"]
    # get_out = lambda x: get_model_output(hf_model, tok, x)
    # get_attn = lambda x: get_attention_values(hf_model, t5x_model, tok, x)
    import pdb

    pdb.set_trace()


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
        required=False,
        help="Path the TX5 checkpoint.",
    )
    args = parser.parse_args()
    # extract_roberta_model(args.t5x_checkpoint_path, args.flax_dump_folder_path)
    play_with_model(args.t5x_checkpoint_path)
