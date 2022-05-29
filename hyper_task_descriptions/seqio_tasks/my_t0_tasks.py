from typing import Dict, List, Tuple
import pkg_resources

import seqio
from promptsource import templates

from hyper_task_descriptions.seqio_tasks import utils
from hyper_task_descriptions.seqio_tasks.t0_tasks import (
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

# create a mixture for every dataset
prefixes_list_filepath = pkg_resources.resource_filename(__name__, "all_t0_task_prefixes.txt")
dataset_names = [
    line.strip() for line in open(prefixes_list_filepath, "r").readlines() if line.strip()
]

for dataset_name in dataset_names:
    seqio.MixtureRegistry.add(
        f"{dataset_name}_train",
        [
            task
            for task in t0_train_mixture["BASE"]
            if (task not in TASK_BLACKLIST and dataset_name in task)
        ],
        default_rate=lambda t: mixture_cap[t.name],
    )

# create a mixture of mixtures using our custom dataset function
seqio.MixtureRegistry.add(
    "t0_double_train",
    [f"{dataset_name}_train" for dataset_name in dataset_names],
    default_rate=lambda t: mixture_cap[t.name],
    sampling_fn=utils.double_sample_from_datasets,
)
