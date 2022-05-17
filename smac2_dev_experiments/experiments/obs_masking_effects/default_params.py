import argparse
import time

launch_time = time.strftime("%Y-%m-%d_%H-%M-%S")

params = dict(
    project="SMAC-v2-obs-masking-effects",
    entity="oxwhirl",
    group="dev-pymarl2",
    launch_time=launch_time,
    save_models=True,
    save_best=True,
    eval_every=5,
    early_stopper_patience=10,
    early_stopper_min_delta=0,
    epochs=300,
    batch_size=512,
    learning_rate=0.005,
    map_name="3s_vs_5z",
    mask_name="nothing",
)


def parse_arguments(params=params):
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f'--{k}', default=v)
    return parser.parse_args()
