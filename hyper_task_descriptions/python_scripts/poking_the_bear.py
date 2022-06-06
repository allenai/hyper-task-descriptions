"""
Export the roberta model from the hypernetwork into a roberta huggingface model.
"""

import argparse

from t5x import checkpoints
from transformers import FlaxRobertaModel, RobertaModel


def extract_roberta_model(t5x_checkpoint_path, flax_dump_folder_path):
    t5x_model = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    roberta_model = FlaxRobertaModel.from_pretrained("hamishivi/fixed-roberta-base")
    roberta_model.params = t5x_model["target"]["hyper"]["encoder"]
    roberta_model.save_pretrained(flax_dump_folder_path)
    model = RobertaModel.from_pretrained(flax_dump_folder_path, from_flax=True)
    model.save_pretrained(flax_dump_folder_path + "_pytorch")


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
    extract_roberta_model(args.t5x_checkpoint_path, args.flax_dump_folder_path)
