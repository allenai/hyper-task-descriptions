#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The Google Research Authors and The HuggingFace Team All rights reserved.
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
"""
Apply partitioning to roberta after the fact.
This is because t5x has partition specs defined at init for params,
but for hf they do not. To avoid rewriting the model, we instead define
partition maps, and apply this after instantiating the model.
"""

# utils adapted from https://github.com/google-research/google-research/blob/master/flax_models/t5x/partitions.py

import re

from flax.core.frozen_dict import freeze
from flax.linen.partitioning import AxisMetadata
from flax.traverse_util import flatten_dict, unflatten_dict

# Sentinels
_unmatched = object()

# For specifying empty leaf dict `{}`
empty_dict = object()


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace


# replicate the hidden dim and shard feed-forward and head dim
# See https://github.com/google-research/t5x/blob/main/docs/usage/partitioning.md for details on dims
# i'm not sure if it works well since the dims are diff to underlying t5 model, but should be fine.
def _get_partition_rules():
    return [
        # embeddings
        (
            ("embeddings", r"[\w_]+", "embedding"),
            AxisMetadata(
                (
                    "vocab",
                    "embed",
                )
            ),
        ),
        # attention
        (
            ("attention", "self", r"(query|key|value)", "kernel"),
            AxisMetadata(
                (
                    "embed",
                    "joined_kv",
                )
            ),
        ),
        (("attention", "self", r"(query|key|value)", "bias"), AxisMetadata(("joined_kv",))),
        (
            ("attention", "output", "dense", "kernel"),
            AxisMetadata(
                (
                    "joined_kv",
                    "embed",
                )
            ),
        ),
        (("attention", "output", "dense", "bias"), AxisMetadata(("embed",))),
        # intermediate
        (
            ("intermediate", "dense", "kernel"),
            AxisMetadata(
                (
                    "embed",
                    "mlp",
                )
            ),
        ),
        (("intermediate", "dense", "bias"), AxisMetadata(("mlp",))),
        # output
        (
            ("output", "dense", "kernel"),
            AxisMetadata(
                (
                    "mlp",
                    "embed",
                )
            ),
        ),
        (("output", "dense", "bias"), AxisMetadata(("embed",))),
        # layer norms
        (("LayerNorm", "bias"), AxisMetadata(("embed",))),
        (("LayerNorm", "scale"), AxisMetadata(("embed",))),
        # pooler
        (
            ("pooler", "dense", "kernel"),
            AxisMetadata(
                (
                    "embed",
                    "embed",
                )
            ),
        ),
        (("pooler", "dense", "bias"), AxisMetadata(("embed",))),
    ]


def set_partitions(in_dict):
    rules = _get_partition_rules()
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))
