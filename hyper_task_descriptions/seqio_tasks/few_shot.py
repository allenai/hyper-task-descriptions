# Copyright 2021 The FLAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for creating few-shot learning tasks. Lightly modified from FLAN implementation."""
import functools
from typing import Optional

import seqio
import tensorflow as tf


@seqio.map_over_dataset
def get_fewshot_num_tokens(example, output_features: seqio.preprocessors.OutputFeaturesType):
    """Computes number of tokens for examples from FewshotDataSource."""
    for split in ["train", "eval"]:
        for key in ["inputs", "targets"]:
            if key not in output_features:
                raise ValueError(
                    "Feature `%s` not in `output_features`. Cannot perform tokenization." % key
                )
            vocab = output_features[key].vocabulary
            example[split][f"{key}_num_tokens"] = tf.reduce_sum(
                tf.ones_like(vocab.encode_tf(example[split][key]), dtype=tf.int32), axis=-1
            )
    return example


@seqio.utils.map_over_dataset
def remove_trailing_spaces(example, features):
    """Removes trailing white spaces from `inputs`."""
    new_example = dict(example)
    for name in features:
        new_example[name] = tf.strings.strip(example[name])
    return new_example


@seqio.map_over_dataset
def prune_fewshot_examples_by_length(example, max_input_length):
    """Prunes execessive exemplars by max input length."""
    inputs_num_tokens = example["train"]["inputs_num_tokens"]
    targets_num_tokens = example["train"]["targets_num_tokens"]
    total_num_tokens = inputs_num_tokens + targets_num_tokens
    total_num_tokens_cm = tf.cumsum(total_num_tokens)
    bool_mask = total_num_tokens_cm <= (max_input_length - example["eval"]["inputs_num_tokens"])

    # Prunes execssive exemplars.
    for name in ["inputs", "targets", "inputs_num_tokens", "targets_num_tokens"]:
        example["train"][name] = tf.boolean_mask(example["train"][name], bool_mask)

    example["eval"]["num_exemplars"] = tf.size(example["train"]["inputs_num_tokens"])
    return example


@seqio.utils.map_over_dataset
def add_delimiter_after_x(ex, x_y_delimiter=" X "):
    new_ex = dict(ex)
    new_ex["inputs"] = tf.strings.join([ex["inputs"], x_y_delimiter])
    return new_ex


def register_few_shot_version_of_task(
    base_task_name: str,
    new_task_name: str,
    num_shots: int,
    x_y_delimiter: str = " X ",
    inputs_prefix: str = "0 ",
    targets_prefix: str = "1 ",
    example_separator: str = " X ",
    prune_exemplars: bool = False,
    max_input_length: Optional[int] = None,
):
    """Registers a few-shot version of a Task."""
    task = seqio.TaskRegistry.get(base_task_name)

    # The list of preprocessors to run on individual exemplars.
    single_ex_preprocessors = list(task.preprocessors)

    def remove_preprocessors_if_present():
        """Removes single-example preprocessors if they are present."""
        num_to_remove = len(single_ex_preprocessors)
        for _ in range(num_to_remove):
            single_ex_preprocessors.pop()

    # Remove all preprocessors, since they just do tokenization and eos, which we will do later.
    remove_preprocessors_if_present()

    # There should be a delimiter between the x and y of each example. Added here.
    @seqio.utils.map_over_dataset
    def add_delimiter_after_x(ex):
        new_ex = dict(ex)
        new_ex["inputs"] = tf.strings.join([ex["inputs"], x_y_delimiter])
        return new_ex

    single_ex_preprocessors.append(add_delimiter_after_x)

    # Form few-shot examples.
    few_shot_data_source = seqio.experimental.FewshotDataSource(
        original_source=task.source,
        num_shots=num_shots,
        train_preprocessors=single_ex_preprocessors,
        eval_preprocessors=single_ex_preprocessors,
        train_split="train",
        train_feature_keys=("inputs", "targets"),
    )
    # These are the preprocessors we run *after* we have formed few-shot examples.
    # Note that we re-introduce the tokenization steps here.
    full_ex_preprocessors = []

    if prune_exemplars:
        # Prunes excessive exemplars according to the max input length.
        if not max_input_length:
            raise ValueError(
                "To prune exemplars, `max_input_length` needs to be provided: %s."
                % max_input_length
            )
        full_ex_preprocessors.extend(
            [
                get_fewshot_num_tokens,
                functools.partial(
                    prune_fewshot_examples_by_length, max_input_length=max_input_length
                ),
            ]
        )

    full_ex_preprocessors.append(
        functools.partial(
            seqio.experimental.fewshot_preprocessor,
            inputs_prefix=inputs_prefix,
            targets_prefix=targets_prefix,
            example_separator=example_separator,
        )
    )

    full_ex_preprocessors.append(functools.partial(remove_trailing_spaces, features=["inputs"]))
    # these always get added!
    full_ex_preprocessors.append(seqio.preprocessors.tokenize)
    full_ex_preprocessors.append(seqio.preprocessors.append_eos)
    full_ex_preprocessors.append(
        seqio.CacheDatasetPlaceholder(required=False),
    )

    seqio.TaskRegistry.add(
        name=new_task_name,
        source=few_shot_data_source,
        output_features=task.output_features,
        preprocessors=full_ex_preprocessors,
        postprocess_fn=task.postprocess_fn,
        metric_fns=task.metric_fns,
    )
