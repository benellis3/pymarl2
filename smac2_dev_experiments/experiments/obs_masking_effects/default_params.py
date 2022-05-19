import argparse
import time

launch_time = time.strftime("%Y-%m-%d_%H-%M-%S")

params = dict(
    project="SMAC2-masking-runs",
    entity="oxwhirl",
    group="dev-pymarl2-smac1",
    launch_time=launch_time,
    save_best=True,
    eval_every=5,
    early_stopper_patience=10,
    early_stopper_min_delta=0,
    epochs=500,
    batch_size=512,
    learning_rate=0.005,
    map_name="3s_vs_5z",
    mask_name="ally_health",
)


def parse_arguments(params=params):
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f'--{k}', default=v)
    return parser.parse_args()
