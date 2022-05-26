import os
import pathlib
import re

# Params.
map_names = [
    "10gen_protoss",
    "10gen_terran",
    "10gen_zerg",
]
nbs_units = [5, 10, 20]
min_learning_steps = 8000000
nb_seeds_needed = 3


map_and_unit_to_model = {(map_name, nb_units): [] for map_name in map_names for nb_units in nbs_units}
seed = {(map_name, nb_units): 0 for map_name in map_names for nb_units in nbs_units}

this_directory = pathlib.Path(__file__).parent.resolve()
machines = filter(lambda dir: not dir.endswith(".py"), os.listdir(this_directory))

for machine in machines:
    path_to_results = this_directory / machine / "sacred"
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
                nb_units = int(re.findall("'n_units': (\d+)", output)[-1])
                if nb_units not in nbs_units:
                    continue
                if seed[(map_name, nb_units)] >= nb_seeds_needed:
                    continue
                origin = f"{machine}/sacred/{map_name}/qmix/{run}"
                map_and_unit_to_model[(map_name, nb_units)].append(
                    (seed[(map_name, nb_units)], f"results_smac2/{machine}/models/{model}", origin))
                seed[(map_name, nb_units)] += 1

# Debug:
for k, v in map_and_unit_to_model.items():
    print(k, v)

print("\n\nCopy this\n\n")
# Output to copy:
for (map_name, nb_units), models in map_and_unit_to_model.items():
    for model in models:
        print(
            f'["{nb_units}_{map_name[2:]}_seed_{model[0]}"]="{model[1]}"'
        )

