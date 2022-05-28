import os
import pathlib
import re

# Params.
map_names = [
    "27m_vs_30m",
    "2c_vs_64zg",
    "3s5z_vs_3s6z",
    "3s_vs_5z",
    "5m_vs_6m",
    "6h_vs_8z",
    "8m_vs_9m",
    "corridor",
    "MMM2"
]

min_learning_steps = 8000000
nb_seeds_needed = 3


map_to_model = {map_name: [] for map_name in map_names}
seed = {map_name: 0 for map_name in map_names}

this_directory = pathlib.Path(__file__).parent.resolve()
machines = filter(lambda dir: not dir.endswith(".py"), os.listdir(this_directory))

for machine in machines:
    path_to_results = this_directory / "sacred"
    for map_name in os.listdir(path_to_results):
        if map_name in map_names:
            runs_dir = os.listdir(path_to_results / map_name / "qmix")
            runs = sorted(filter(lambda x: x.isdigit(), runs_dir), reverse=True, key=lambda x: int(x))
            for run in runs:
                output_file = path_to_results / map_name / "qmix" / run / "cout.txt"
                output = open(output_file, "r", encoding="utf-8").read()
                models = re.findall(r"Saving models to results/models/qmix__\S+", output)
                if len(models) == 0:
                    continue
                last_model = models[-1]
                last_save_step = int(last_model.split("/")[-1])
                if last_save_step < min_learning_steps:
                    continue
                model = last_model.split("/")[-2]
                if seed[map_name] >= nb_seeds_needed:
                    continue
                origin = f"sacred/{map_name}/qmix/{run}"
                map_to_model[map_name].append(
                    (seed[map_name], f"results_smac1/models/{model}", origin))
                seed[map_name] += 1

# Debug:
for k, v in map_to_model.items():
    print(k, v)

print("\n\nCopy this\n\n")
# Output to copy:
for map_name, models in map_to_model.items():
    for model in models:
        print(
            f'["{map_name}_seed_{model[0]}"]="{model[1]}"'
        )

