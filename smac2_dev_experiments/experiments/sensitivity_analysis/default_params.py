import argparse
import time

launch_time = time.strftime("%Y-%m-%d_%H-%M-%S")

params = dict(
    project="SMAC2-masking-runs",
    entity="oxwhirl",
    group="dev-pymarl2-smac1-sensitivity",
    launch_time=launch_time,
    smac_version=1,
    map_name="27m_vs_30m",
    nb_units=20,
    seed=0,
)


def parse_arguments(params=params):
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f'--{k}', default=v)
    return parser.parse_args()
