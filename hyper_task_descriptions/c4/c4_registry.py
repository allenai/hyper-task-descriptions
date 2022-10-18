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

# modified prefix lm pretraining, using 3-way split.

import os
import functools
import random
import seqio
import tensorflow as tf
from t5.data import preprocessors

from hyper_task_descriptions.hf_vocab import HuggingfaceVocabulary

TaskRegistry = seqio.TaskRegistry

seqio.add_global_cache_dirs(["gs://hamishi-us-bucket/c4_pretrain_data"])

t5_vocab = HuggingfaceVocabulary("t5-base")

words_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "numeric_task",
    "words.txt"
)

words = [
    line.strip() for line in open(words_path, "r").readlines()
]

def pack_prefix_lm_encoder_decoder_random_inputs(ds, sequence_length, pad_id=0):
    """Setup example for prefix lm. no packing becuz im lazy"""

    @seqio.utils.map_over_dataset(num_seeds=2)
    def create_example(example, seeds):
        split_point_1 = tf.random.stateless_uniform(
            (), minval=1, maxval=example["targets"].shape[0], seed=seeds[0], dtype=tf.int32
        )
        hyper_inputs = example["targets"][:split_point_1]
        targets = example["targets"][split_point_1:]

        #inputs = t5_vocab._encode_tf(random.choice(words))
        # We want the length _after_ tokenization to be sequence_length['inputs']
        inputs = t5_vocab._encode_tf(' '.join(random.choices(words, k=sequence_length['inputs'] // 4)))
        return {
            "inputs": inputs,
            "hyper_inputs": hyper_inputs,
            "targets": targets,
            "task_names": [0],
        }

    return create_example(ds)


# only compatible when we use the T5 encoder as our hypernetwork
seqio.TaskRegistry.add(
    "c4_pretrain",
    source=seqio.TfdsDataSource(tfds_name="c4/en:3.0.1", splits=["train", "validation"]),
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
        pack_prefix_lm_encoder_decoder_random_inputs,
    ],
    output_features={
        "inputs": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "targets": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "hyper_inputs": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "task_names": seqio.Feature(seqio.PassThroughVocabulary(1), add_eos=False, dtype=tf.int32),
    },
    metric_fns=[],
)