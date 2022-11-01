import sys

import matplotlib.pyplot as plt

with open(sys.argv[1], "r") as f:
    data = [x.strip().split("\t") for x in f.readlines()][1:]

tasks = sorted(list(set([x[0] for x in data])))


def get_model_data(model_idx, data):
    task_values = {}
    for line in data:
        task = line[0]
        if task not in task_values:
            task_values[task] = []
        if model_idx < len(line):
            task_values[task].append(float(line[model_idx]))
        else:
            task_values[task].append(0)
    return task_values


my_t0_3b = get_model_data(2, data)
original_t0_3b = get_model_data(3, data)
scratch_t0_3b = get_model_data(4, data)

original_t0_11b = get_model_data(6, data)
scratch_t0_11b = get_model_data(7, data)


fig, axs = plt.subplots(2, 6, figsize=(20, 10))

models_to_eval = [original_t0_3b, scratch_t0_3b, original_t0_11b, scratch_t0_11b]
model_names = ["T0 3B", "T03Bp", "T011B", "T011Bp"]
print(tasks)
for i, task in enumerate(tasks):
    vert_idx = i // 6
    horiz_idx = i % 6
    axs[vert_idx, horiz_idx].set_title(task if task else "avg")
    axs[vert_idx, horiz_idx].set_xticklabels(model_names)
    axs[vert_idx, horiz_idx].boxplot([x[task] for x in models_to_eval if task in x])
    for j, model in enumerate(models_to_eval):
        axs[vert_idx, horiz_idx].scatter([j + 1] * len(model[task]), model[task])
plt.tight_layout()
plt.savefig("boxplot.png")
# plt.show()
