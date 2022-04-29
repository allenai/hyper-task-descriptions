"""
This file defines 8 mixtures that was used in the T-Zero paper:
- t0_train: T0 training mixture
- t0+_train: T0+ training mixture
- t0++_train: T0++ training mixture
- t0_eval_score_eval: T0 main evaluation mixture (Figure 4 for instance)
- t0_train_score_eval: Evaluation mixture for checkpoint selection on T0 (validation splits of the training sets)
- t0_train_one_og_prompt: T0 (p=1) training mixture for  - one original-task prompt per dataset. Figure 6
- t0_train_all_og_prompts: T0 (p=5.7) training mixture for - all original-task prompts for all datasets. Figure 6
- bias_fairness_eval_score_eval: Bias & fairness evaluation mixture. Appendix B3

Adapted from T0 repo.
"""

import functools
from typing import Dict, List, Tuple

import seqio
from promptsource import templates

from hyper_task_descriptions.seqio_tasks import utils
from hyper_task_descriptions.seqio_tasks.t0_tasks import (
    D4_TRAIN_SCORE_EVAL_TASK_BLACKLIST,
    D4_TRAIN_SKIP_EVAL,
    TASK_BLACKLIST,
    add_task,
    create_mixture_lists,
    load_t0_csv,
)

t0_train, t0_eval, gsheet = load_t0_csv()

all_templates = templates.TemplateCollection()
all_templates.remove("anli")  # Need to special-case ANLI due to weird split conventions

mixtures = create_mixture_lists(t0_train, t0_eval, gsheet)
# 3 stages of training/ablation: D4 -> GPT -> SuperGLUE
t0_train_mixture: Dict[str, List[str]] = mixtures[0]
t0_eval_mixture: Dict[str, List[str]] = mixtures[1]
mixture_cap: Dict[str, int] = mixtures[2]
single_original_task: Dict[Tuple[str, str], str] = mixtures[3]
all_original_tasks: List[str] = mixtures[4]

# Special case for ANLI, which has weirdly-named splits and rounds that should be subsets
dataset_name, subset_name = ("anli", None)
dataset = all_templates.get_dataset(dataset_name, subset_name)
for anli_round in ("r1", "r2", "r3"):
    for template_name in all_templates.get_dataset(dataset_name, subset_name).all_template_names:
        task_name = utils.get_task_name(dataset_name, subset_name, template_name) + f"_{anli_round}"
        split_mapping = {
            "train": f"train_{anli_round}",
            "validation": f"dev_{anli_round}",
            "test": f"test_{anli_round}",
        }
        add_task(dataset_name, subset_name, template_name, all_templates, task_name, split_mapping)

        template = dataset[template_name]
        if template.metadata.original_task:
            t0_eval_mixture["BASE"].append(task_name)  # TODO or add to ANLI special mixture
        # TODO use template.metadata.answer_choices here for rank eval

# create mixture set.
seqio.MixtureRegistry.add(
    "t0_train",
    [task for task in t0_train_mixture["BASE"] if task not in TASK_BLACKLIST],
    default_rate=lambda t: mixture_cap[t.name],
)

seqio.MixtureRegistry.add(
    "t0+_train",
    [
        task
        for task in t0_train_mixture["BASE"] + t0_train_mixture["GPT_EVAL"]
        if task not in TASK_BLACKLIST
    ],
    default_rate=lambda t: mixture_cap[t.name],
)

seqio.MixtureRegistry.add(
    "t0++_train",
    [
        task
        for task in t0_train_mixture["BASE"]
        + t0_train_mixture["GPT_EVAL"]
        + t0_train_mixture["SGLUE"]
        if task not in TASK_BLACKLIST
    ],
    default_rate=lambda t: mixture_cap[t.name],
)

seqio.MixtureRegistry.add(
    "t0_eval_score_eval",
    [
        task
        for task in seqio.TaskRegistry.names()
        if task.endswith("_score_eval")
        and task.split("_score_eval")[0] in t0_eval_mixture["BASE"]
        and task.split("_score_eval")[0] not in TASK_BLACKLIST
    ],
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
)

# a small set of tasks to develop over
seqio.MixtureRegistry.add(
    "small_dev_tasks",
    [
        task
        for task in t0_train_mixture["BASE"]
        + t0_train_mixture["GPT_EVAL"]
        + t0_train_mixture["SGLUE"]
        if task not in TASK_BLACKLIST
    ][:10],
    default_rate=lambda t: mixture_cap[t.name],
)

seqio.MixtureRegistry.add(
    "t0_train_score_eval",
    [
        task
        for task in seqio.TaskRegistry.names()
        if task.endswith("_score_eval")
        and task.split("_score_eval")[0] in t0_train_mixture["BASE"]
        and task.split("_score_eval")[0] not in TASK_BLACKLIST
        and task not in D4_TRAIN_SCORE_EVAL_TASK_BLACKLIST
        and not any([skip in task for skip in D4_TRAIN_SKIP_EVAL])
        and task.split("_score_eval")[0] in all_original_tasks
    ],
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
)

seqio.MixtureRegistry.add(
    "t0_train_one_og_prompt",
    [
        task
        for task in single_original_task.values()
        if task in t0_train_mixture["BASE"] and task not in TASK_BLACKLIST
    ],
    default_rate=lambda t: mixture_cap[t.name],
)

seqio.MixtureRegistry.add(
    "t0_train_all_og_prompts",
    [
        task
        for task in all_original_tasks
        if task in t0_train_mixture["BASE"] and task not in TASK_BLACKLIST
    ],
    default_rate=lambda t: mixture_cap[t.name],
)

seqio.MixtureRegistry.add(
    "bias_fairness_eval_score_eval",
    [
        task
        for task in seqio.TaskRegistry.names()
        if task.endswith("_score_eval")
        and task.split("_score_eval")[0] in t0_eval_mixture["BIAS_FAIRNESS"]
    ],
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
)
