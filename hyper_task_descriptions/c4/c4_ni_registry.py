'''
Combined pretraining-finetuning mixture
'''
import seqio

from hyper_task_descriptions.c4 import c4_registry  # noqa: F401
from hyper_task_descriptions.ni_tasks import ni_registry  # noqa: F401

# one for each ni version
# idk what the best mixing rate is. trying out 50-50 rn.
seqio.MixtureRegistry.add(
    "c4_ni",
    [("c4_pretrain", 1), ("natural_instructions", 1)]
)

seqio.MixtureRegistry.add(
    "c4_ni_def",
    [("c4_pretrain", 1), ("natural_instructions_def", 1)]
)
