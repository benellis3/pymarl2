# Experiments on observation sensitivity.

Assumes you already have trained models.

### Build Docker Image.
In `docker/Dockerfile`. Build with `docker/build.sh`.

### Install Starcraft2.
With `./install_sc2.sh`.

### Collect data.
Data consists of:
- Episodes stored in a ReplayBuffer,
- The model/policy that generated the episodes, and
- The config associated to the training the generated the episodes.
- The feature names associated to the training.

Edit (specify maps and model locations) and run `pymarl2/collect_data.sh`.

Data will be saved in `pymarl2/smac2_dev_experiments/data/smac${version}/map_name/${dataset_data}`.

Copy/Move the datasets to `.../map_name/train` and `.../map_name/evaluate`.

### Running the Masking Experiment.

The parameters of the experiment are passed
according to `smac2_dev_experiments/experiments/obs_masking_effects/default_params.py`.

Run `smac2_dev_experiments/run_in_docker smac2_dev_experiments/experiments/obs_masking_effects/main.py`
