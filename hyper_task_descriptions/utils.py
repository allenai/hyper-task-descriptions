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

from t5x import partitioning
from t5x.checkpoints import RestoreStateTransformationFn

PartitionRule = Tuple[str, Optional[partitioning.PartitionSpec]]


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
