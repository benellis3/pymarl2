# Experiments on observation sensitivity.

Assumes you already have trained models saved in `pymarl2/results` or `pymarl2/results_smac_2`.

## Prereqs:

### Build Docker Image.
In `docker/Dockerfile`. Edit the version of SMAC to use in `install_dependencies.sh`.
Build with `cd docker && ./build.sh`.

### Install Starcraft2.
With `./install_sc2.sh`.

### Export your WANDB_API_KEY as an environment variable

## Collecting data.
Data consists of:
- Episodes stored in a ReplayBuffer,
- The model/policy that generated the episodes, and
- The config associated to the training the generated the episodes.
- The feature names associated to the training.

Data collection is done in `./smac${version}_dev_experiments/collect_data_scripts/`
1. Check that maps are mapped to model weights in `collect_data_smac${version}.sh`. 
Otherwise get the mapping by running  `results_smac${version}/build_map_to_model_table.py`
2. Edit (specify maps, units) and run
`./smac${version}_dev_experiments/collect_data_scripts/collect_data_smac2.sh qmix debug`.
(modify `debug` to `train` or `evaluate`)

Data will be saved in `pymarl2/smac2_dev_experiments/data/smac_${version}/map_name/${seed}/${task_name}`.

Copy the dataset to the other machines you'll be using.

## Running the Masking Experiment.

The parameters of the experiment are passed
according to `smac2_dev_experiments/experiments/obs_masking_effects/default_params.py`.

You can edit them manually there, pass then with command-line arguments, or run a wandb sweep.

### Create a wandb sweep
Create a `sweep.yaml` file as in `smac2_dev_experiments/experiments/obs_masking_effects/sweeps/smac_2/example.yaml`.

Initialize the sweep with `./smac2_dev_experiments/create_sweep.sh obs_masking_effects/sweeps/smac_2/example.yaml`.

Remarks: workers can pull any configuration from the sweep, so ideally you want to segment your sweeps.
E.g. a sweep for large datasets which will have at most one worker running in parallel per machine
and a sweep for small datasets having n workers in parallel per machine.

Launch a worker with `./smac2_dev_experiments/run_sweep.sh <SWEEP_ID> <GPUS to use>`.
