# Copyright 2022 The T5 Authors.
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

# modified prefix lm pretraining, using 6-way split.
# the idea here is to encourage (a) heavy usage of the hypernet and (b) close coordination b/w enc and hnet.

import functools
import os

import seqio
import tensorflow as tf
from t5.data import preprocessors

from hyper_task_descriptions.hf_vocab import HuggingfaceVocabulary

TaskRegistry = seqio.TaskRegistry

seqio.add_global_cache_dirs(["gs://hamishi-us-bucket/c4_pretrain_data"])

t5_vocab = HuggingfaceVocabulary("t5-base")

words_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "numeric_task", "words.txt"
)

words = [line.strip() for line in open(words_path, "r").readlines()]


def pack_hypertune(ds, sequence_length, pad_id=0):
    """Setup example for prefix lm. no packing becuz im lazy"""

    @seqio.utils.map_over_dataset(num_seeds=6)
    def create_example(example, seeds):
        # we split the target into 4 parts:
        # a -> hypernet
        # b (short) -> encoder
        # c (short) -> decoder
        # d -> hypernet again.
        # currently using random lengths
        split_point_1 = tf.random.stateless_uniform(
            (), minval=1, maxval=example["targets"].shape[0] - 6, seed=seeds[0], dtype=tf.int32
        )
        split_point_2 = tf.random.stateless_uniform(
            (),
            minval=split_point_1,
            maxval=example["targets"].shape[0] - 5,
            seed=seeds[1],
            dtype=tf.int32,
        )
        split_point_3 = tf.random.stateless_uniform(
            (),
            minval=split_point_2,
            maxval=example["targets"].shape[0] - 4,
            seed=seeds[2],
            dtype=tf.int32,
        )
        split_point_4 = tf.random.stateless_uniform(
            (),
            minval=split_point_2,
            maxval=example["targets"].shape[0] - 3,
            seed=seeds[3],
            dtype=tf.int32,
        )
        split_point_5 = tf.random.stateless_uniform(
            (),
            minval=split_point_2,
            maxval=example["targets"].shape[0] - 2,
            seed=seeds[4],
            dtype=tf.int32,
        )
        # '1' as eos to mark end of first part of input
        hyper_inputs = tf.concat(
            [
                example["targets"][:split_point_1],
                [1],
                example["targets"][split_point_2:split_point_3],
                [1],
                example["targets"][split_point_5:]
            ], axis=0
        )
        inputs = tf.concat(
            [
                example["targets"][split_point_1:split_point_2],
                [1],
                example["targets"][split_point_3:split_point_4]
            ], axis=0
        )
        targets = example["targets"][split_point_4:split_point_5]

        example["targets"][split_point_2:split_point_3]

        return {
            "inputs": inputs,
            "hyper_inputs": hyper_inputs,
            "targets": targets,
            "task_names": [0],
        }

    return create_example(ds)


seqio.TaskRegistry.add(
    "c4_pretrain",
    source=seqio.TfdsDataSource(tfds_name="c4/en:3.1.0", splits=["train", "validation"]),
    preprocessors=[
        functools.partial(
            preprocessors.rekey,
            key_map={
                "inputs": None,
                "targets": "text",
                # "roberta_targets": "text"  # for roberta vocab
            },
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.targets_for_prefix_lm_objective,
        pack_hypertune,
        seqio.preprocessors.append_eos,
    ],
    output_features={
        "inputs": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "targets": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "hyper_inputs": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "task_names": seqio.Feature(seqio.PassThroughVocabulary(1), add_eos=False, dtype=tf.int32),
    },
    metric_fns=[],
)
