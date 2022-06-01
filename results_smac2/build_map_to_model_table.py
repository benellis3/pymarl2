import os
import re
from pathlib import Path
from distutils.dir_util import copy_tree
from tqdm import tqdm

map_names = [
    "10gen_protoss",
    "10gen_terran",
    "10gen_zerg",
]
nbs_units = [5, 10, 20]
min_learning_steps = 10000000
nb_seeds_needed = 3


map_and_unit_to_model = {
    (map_name, nb_units): [] for map_name in map_names for nb_units in nbs_units
}
SEED = {(map_name, nb_units): 0 for map_name in map_names for nb_units in nbs_units}

RESULTS_DIR = Path("/Users/benellis/src/qmix_smacv2_results")


def is_valid_qmix_run(output):
    # check the hyperparam values to check same as the paper
    eps_anneal_time_match = re.search(r"'epsilon_anneal_time': 100000,", output)
    if not eps_anneal_time_match:
        return False
    td_lambda_match = re.search(r"'td_lambda': 0.4,", output)
    if not td_lambda_match:
        return False
    return True


def move_model_and_logs_to_new_dir(
    map_name, nb_units, seed, last_save_step, model, curr_logs_dir
):
    map_nb_units_dir = RESULTS_DIR / map_name / str(nb_units) / str(seed)
    map_nb_units_dir.mkdir(parents=True, exist_ok=True)
    model_dir = map_nb_units_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    curr_model_dir = RESULTS_DIR / "models" / model / str(last_save_step)
    copy_tree(str(curr_model_dir), str(model_dir / str(last_save_step)))

    # copy the logs across as well
    logs_dir = map_nb_units_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    copy_tree(str(curr_logs_dir), str(logs_dir))


def main():
    path_to_results = RESULTS_DIR / "sacred"
    for map_name in os.listdir(path_to_results):
        if map_name in map_names:
            runs_dir = os.listdir(path_to_results / map_name / "qmix")
            runs = sorted(
                filter(lambda x: x.isdigit(), runs_dir),
                reverse=True,
                key=int,
            )
            for run in tqdm(runs):
                logs_dir = path_to_results / map_name / "qmix" / run
                output_file = logs_dir / "cout.txt"
                output = open(output_file, "r", encoding="utf-8").read()
                if not is_valid_qmix_run(output):
                    continue
                models = re.findall(
                    r"Saving models to results/models/qmix__\S+", output
                )
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
                if SEED[(map_name, nb_units)] >= nb_seeds_needed:
                    continue
                # origin = f"{machine}/sacred/{map_name}/qmix/{run}"
                # map_and_unit_to_model[(map_name, nb_units)].append(
                #     (
                #         SEED[(map_name, nb_units)],
                #         f"results_smac2/{machine}/models/{model}",
                #         origin,
                #     )
                # )
                move_model_and_logs_to_new_dir(
                    map_name,
                    nb_units,
                    SEED[(map_name, nb_units)],
                    last_save_step,
                    model,
                    logs_dir,
                )
                SEED[(map_name, nb_units)] += 1

    # Debug:
    # for k, v in map_and_unit_to_model.items():
    #     print(k, v)

    # print("\n\nCopy this\n\n")
    # # Output to copy:
    # for (map_name, nb_units), models in map_and_unit_to_model.items():
    #     for model in models:
    #         print(f'["{nb_units}_{map_name[2:]}_seed_{model[0]}"]="{model[1]}"')


if __name__ == "__main__":
    main()
