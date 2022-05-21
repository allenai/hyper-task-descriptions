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

import datasets
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
from tqdm import tqdm

MAX_EXAMPLES_PER_DATASET = 500_000

t0_train, t0_eval, gsheet = load_t0_csv()

all_templates = templates.TemplateCollection()
all_templates.remove("anli")  # Need to special-case ANLI due to weird split conventions
all_datasets = sum(t0_train.values(), []) + sum(t0_eval.values(), [])

# 3 stages of training/ablation: D4 -> GPT -> SuperGLUE
t0_train_mixture: Dict[str, List[str]] = {key: [] for key in t0_train}
t0_eval_mixture: Dict[str, List[str]] = {key: [] for key in t0_eval}
mixture_cap: Dict[str, int] = {}
single_original_task: Dict[Tuple[str, str], str] = {}
all_original_tasks: List[str] = []
for dataset_name, subset_name in tqdm(all_templates.keys):
    if (dataset_name, subset_name) not in all_datasets:
        all_templates.remove(dataset_name, subset_name)
        continue
    dataset = all_templates.get_dataset(dataset_name, subset_name)
    num_templates = len(dataset.all_template_names)
    train_size = gsheet[(dataset_name, subset_name)]["train_size"]
    if train_size == "":
        train_size = 0
    else:
        train_size = int(train_size)
    if train_size > MAX_EXAMPLES_PER_DATASET:
        cap = MAX_EXAMPLES_PER_DATASET // num_templates
    else:
        cap = train_size
    for template_name in tqdm(dataset.all_template_names):
        dataset_splits = utils.get_dataset_splits(dataset_name, subset_name)
        split_mapping = {k: k for k in dataset_splits.keys()}
        dataset = datasets.load_dataset(dataset_name, subset_name)
        dataset = dataset[split_mapping['train']]
        template = all_templates.get_dataset(dataset_name, subset_name)[template_name]
        dataset = utils.apply_template_split(dataset, template)
        print(dataset[0]['hyper_inputs'])