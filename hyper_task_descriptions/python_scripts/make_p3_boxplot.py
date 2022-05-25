import sys

import matplotlib.pyplot as plt

with open(sys.argv[1], "r") as f:
    data = [x.strip().split(",") for x in f.readlines()][1:]

tasks = sorted(list(set([x[0] for x in data])))


def get_model_data(model_idx, data):
    task_values = {}
    for line in data:
        task = line[0]
        if task not in task_values:
            task_values[task] = []
        task_values[task].append(float(line[model_idx]))
    return task_values


t0_3b_vals = get_model_data(2, data)
t0_11b_vals = get_model_data(3, data)
t0p_11b_vals = get_model_data(4, data)
t0pp_11b_vals = get_model_data(5, data)
t03b4k_vals = get_model_data(6, data)
my_vals = get_model_data(7, data)
my_vals_13 = get_model_data(8, data)

fig, axs = plt.subplots(2, 6)

models_to_eval = [t0_3b_vals, t0_11b_vals, t03b4k_vals, my_vals, my_vals_13]
model_names = ["T0 3B", "T0", "T03B+4K", "Mine+4k", "Mine+14k"]
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
