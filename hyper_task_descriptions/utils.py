# Copyright 2022 Google.
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
Adapted from prompt-tuning repo, just need the regex traversal.
https://github.com/google-research/prompt-tuning/blob/a7d507fbb01c0a5d24c9726bd61dbf966024c0d0/prompt_tuning/train/utils.py
"""
import re
from typing import Any, Callable, Optional, Sequence, Tuple

import flax
import optax
from flax.core import frozen_dict
from t5x import optimizers, partitioning

PartitionRule = Tuple[str, Optional[partitioning.PartitionSpec]]

GOOGLE_BUCKET_PATH = "gs://hamishi-tpu"


def match_any(regexes: Sequence[str]) -> Callable[[str, Any], bool]:
    """A traversal that checks if the parameter name matches any regex.
    This is returns a closure over the actual traversal function that takes the
    parameter name and value. The return value of this should be in input to the
    Traversal used in the MultiOptimizer.
    Args:
      regexes: A list of regular expressions that denote which parameter should be
        updated by this optimizer.
    Returns:
      A function that takes the name and value of a parameter and return True if
      that parameter should be updated by the optimizer.
    """
    regexes = tuple(re.compile(regex) for regex in regexes)

    def _match_any(path, _):
        """True if path matches any regex in regexs, false otherwise."""
        return any(regex.fullmatch(path) for regex in regexes)

    return _match_any


def inverse_match_any(regexes: Sequence[str]) -> Callable[[str, Any], bool]:
    """Inverse of the above"""
    regexes = tuple(re.compile(regex) for regex in regexes)

    def _match_any(path, _):
        """False if path matches any regex in regexs, true otherwise."""
        return not any(regex.fullmatch(path) for regex in regexes)

    return _match_any


def flattened_traversal(fn):
    """Returns function that is called with `(path, param)` instead of pytree."""

    def mask(tree):
        flat = flax.traverse_util.flatten_dict(tree, sep="/")
        masked_tree = flax.traverse_util.unflatten_dict(
            {k: fn(k, v) for k, v in flat.items()}, sep="/"
        )
        return frozen_dict.freeze(masked_tree)

    return mask


def match_any_optax(regexes: Sequence[str]) -> Callable[[str, Any], bool]:
    regexes = tuple(re.compile(regex) for regex in regexes)

    def _match_any(path, _):
        if any(regex.fullmatch(path) for regex in regexes):
            return "train"
        else:
            return "freeze"

    label_fn = flattened_traversal(lambda path, _: _match_any(path, _))
    return label_fn


def match_any_optax_trip(
    regexes: Sequence[str], hyper_regexes: Sequence[str]
) -> Callable[[str, Any], bool]:
    regexes = tuple(re.compile(regex) for regex in regexes)
    hyper_regexes = tuple(re.compile(regex) for regex in hyper_regexes)

    def _match_any(path, _):
        if any(regex.fullmatch(path) for regex in regexes):
            return "roberta"
        elif any(regex.fullmatch(path) for regex in hyper_regexes):
            return "hyper"
        else:
            return "freeze"

    label_fn = flattened_traversal(lambda path, _: _match_any(path, _))
    return label_fn


# inverse match, mainly for adamw weight decay - see partial_adamw.gin for an example of how this is applied
def match_any_optax_inverse(regexes: Sequence[str]) -> Callable[[str, Any], bool]:
    regexes = tuple(re.compile(regex) for regex in regexes)
    label_fn = flattened_traversal(
        lambda path, _: "freeze" if any(regex.fullmatch(path) for regex in regexes) else "train"
    )
    return label_fn


# t5x doesnt wrap this but i need it
multi_transform = optimizers.wrap_optax_optimizer(optax.multi_transform)

# non-wrapped version for use with the wrapped multi_transform


def chain(transformations: Sequence[optax.GradientTransformation]) -> optax.GradientTransformation:
    return optax.chain(*transformations)
