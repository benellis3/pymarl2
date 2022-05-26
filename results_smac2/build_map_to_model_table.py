import os
import pathlib


map_and_unit_to_model = dict()

this_directory = pathlib.Path(__file__).parent.resolve()
machines = filter(lambda dir: not dir.endswith(".py"), os.listdir(this_directory))

for machine in machines:
    path_to_results = this_directory / machine / "sacred"
    for map_name in os.listdir(path_to_results):
        if map_name.startswith("10gen_"):
            runs_dir = os.listdir(path_to_results / map_name / "qmix")
            runs = sorted(runs_dir, reverse=True)
            for run in runs:
                run_output = path_to_results / map_name / "qmix" / run / "cout.txt"



