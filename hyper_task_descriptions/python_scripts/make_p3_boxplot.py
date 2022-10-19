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
        task_values[task].append(float(line[model_idx]))
    return task_values


original_t0_3b = get_model_data(3, data)
my_t0_3b = get_model_data(2, data)
original_t0_11b = get_model_data(6, data)


fig, axs = plt.subplots(2, 6)

models_to_eval = [original_t0_3b, my_t0_3b, original_t0_11b]
model_names = ["T0 3B", "T03B+16K", "T011B"]
print(tasks)
for i, task in enumerate(tasks):
    vert_idx = i // 6
    horiz_idx = i % 6
    axs[vert_idx, horiz_idx].set_title(task if task else "avg")
    axs[vert_idx, horiz_idx].set_xticklabels(model_names)
    axs[vert_idx, horiz_idx].boxplot([x[task] for x in models_to_eval])
    for j, model in enumerate(models_to_eval):
        axs[vert_idx, horiz_idx].scatter([j + 1] * len(model[task]), model[task])

plt.show()
