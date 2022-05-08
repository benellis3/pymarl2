import argparse
import time

launch_time = time.strftime("%Y-%m-%d_%H-%M-%S")

params = dict(
    project="SMAC-v2-obs-masking-effects",
    entity="oxwhirl",
    group="dev-pymarl2",
    launch_time=launch_time,
    save_models=True,
    save_every=50,
    map_name="3s_vs_5z",
    epochs=200,
    batch_size=128,
    learning_rate=0.005,
    mask_name="everything",
    mask_state=0,
)


def parse_arguments(params=params):
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f'--{k}', default=v)
    return parser.parse_args()
