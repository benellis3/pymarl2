# SMAC Feature Inferrability & Relevance Experiment

_Appendix to SMACv2: A New Benchmark for Cooperative Multi-Agent Reinforcement Learning_

## Contents:

1. [Installation](#installation)
2. [Runnning the Experiment](#running-the-experiment)

## Installation

We use Docker to manage environments.

1. Build an image for SMAC and SMACv2.

   ```shell
   cd docker
   ./build.sh 1
   ./build.sh 2
   ```

2. Install StarCraft II (SC2.4.10) and SMAC maps:

   ```shell
   bash install_sc2.sh
   ```

   This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

3. Configure `wandb`

   We use Weights & Biases to manage ethe logs of our experiments. If you want to use `wandb` then

   ```shell
   export WANDB_API_KEY=<your_wandb_api_key>
   ```

   Otherwise, you can run the experiments offline.

## Running the Experiment

We describe how to run the experiment for a specific SMAC scenario.

1. Collect a dataset for a scenario:

   A dataset consists of replay episodes collected with an already trained QMIX policy for the given scenario.
   The weights of the trained QMIX policies are saved in `results_smac1` and `results_smac2_final_run`.
   The parameter used for their training, and the full logs of their training can be found in their respective `logs/cout.txt` file.

   To generate the dataset associated to the first seed on the `5_gen_protoss` scenario you can run

   ```shell
   # USAGE: ./run_docker <GPU_to_use> <SMAC_version>
   ./run_docker.sh 0 2 src/main.py --config=qmix --env-config=sc2_gen_protoss with use_wandb=False env_args.capability_config.n_units=5 env_args.capability_config.start_positions.n_enemies=5 checkpoint_path=results_smac2_final_run/10gen_protoss/5/0/models evaluate=True buffer_size=8192 test_nepisode=8192 save_eval_buffer=True save_eval_buffer_path=smac2_dev_experiments/data/smac_2 saving_eval_seed=0 saving_eval_type=train
   ./run_docker.sh 0 2 src/main.py --config=qmix --env-config=sc2_gen_protoss with use_wandb=False env_args.capability_config.n_units=5 env_args.capability_config.start_positions.n_enemies=5 checkpoint_path=results_smac2_final_run/10gen_protoss/5/0/models evaluate=True buffer_size=4096 test_nepisode=4096 save_eval_buffer=True save_eval_buffer_path=smac2_dev_experiments/data/smac_2 saving_eval_seed=0 saving_eval_type=evaluate
   ```
   A table mapping model to the location of their weights can be found in `smac2_dev_experiments/collect_data_scripts/smac2.sh` or `.../smac1.sh`.
   The full list of commands used to generate the datasets of SMACv2 can be found in `smac2_dev_experiments/collect_data_scripts/commands_smac2.txt`.

2. Run a masking experiment:
   1. Edit `smac2_dev_experiments/experiments/obs_masking_effects/default_params.py` to specify correct `wandb` project, entity, and group if you're using wandb.
   2. Also edit the file if you want to specify a different scenario/seed.
   3. Edit `mode="online"` in `smac2_dev_experiments/experiments/obs_masking_effects/main.py` is using wandb.
   4. run `./smac2_dev_experiments/run_in_docker.sh 0 2 python smac2_dev_experiments/experiments/obs_masking_effects/main.py`
   5. results will overwrite `smac2_dev_experiments/experiments/obs_masking_effects/results/smac_2/5_gen_protoss/0/ally_health`
`
