"""
Defines 't0_small_train', a minimal set of tasks to allow local dev without loading all of P3.
"""
import seqio

from hyper_task_descriptions.seqio_tasks.t0_tasks import (
    TASK_BLACKLIST,
    create_mixture_lists,
    load_t0_csv,
)

t0_train, t0_eval, gsheet = load_t0_csv()

# only want one instance.
t0_train["GPT_EVAL"] = []
t0_train["SGLUE"] = []
t0_eval["BIAS_FAIRNESS"] = []
t0_train["BASE"] = t0_train["BASE"][:1] #[("super_glue", "rte"), ("super_glue", "cb")]
t0_eval["BASE"] = t0_eval["BASE"][:1] #[("super_glue", "rte"), ("super_glue", "cb")]

# download the dataset infos
mixtures = create_mixture_lists(t0_train, t0_eval, gsheet)

# create our singular mixture
t0_train_mixture = mixtures[0]
mixture_cap = mixtures[2]
seqio.MixtureRegistry.add(
    "t0_small_train",
    [task for task in t0_train_mixture["BASE"] if task not in TASK_BLACKLIST],
    default_rate=lambda t: mixture_cap[t.name],
)
