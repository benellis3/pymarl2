#!/bin/bash
trap 'onCtrlC' INT
# set -x
function onCtrlC () {
  echo 'Ctrl+C is captured'
  for pid in $(jobs -p); do
    kill -9 $pid
  done

  kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')
  exit 1
}

config=$1  # qmix
tag=$2  # train, evaluate, or debug.
maps=${3:-sc2_gen_protoss,sc2_gen_terran,sc2_gen_zerg}
units=${8:-5}

# Edit this.
declare -A weight_location
weight_location=(
["5_gen_protoss_seed_0"]="results_smac2_unit_ranges/10gen_protoss/5/0/models"
["5_gen_protoss_seed_1"]="results_smac2_unit_ranges/10gen_protoss/5/1/models"
["5_gen_protoss_seed_2"]="results_smac2_unit_ranges/10gen_protoss/5/2/models"
["10_gen_protoss_seed_0"]="results_smac2_unit_ranges/10gen_protoss/10/0/models"
["10_gen_protoss_seed_1"]="results_smac2_unit_ranges/10gen_protoss/10/1/models"
["10_gen_protoss_seed_2"]="results_smac2_unit_ranges/10gen_protoss/10/2/models"
["20_gen_protoss_seed_0"]="results_smac2_unit_ranges/10gen_protoss/20/0/models"
["20_gen_protoss_seed_1"]="results_smac2_unit_ranges/10gen_protoss/20/1/models"
["20_gen_protoss_seed_2"]="results_smac2_unit_ranges/10gen_protoss/20/2/models"
["5_gen_terran_seed_0"]="results_smac2_unit_ranges/10gen_terran/5/0/models"
["5_gen_terran_seed_1"]="results_smac2_unit_ranges/10gen_terran/5/1/models"
["5_gen_terran_seed_2"]="results_smac2_unit_ranges/10gen_terran/5/2/models"
["10_gen_terran_seed_0"]="results_smac2_unit_ranges/10gen_terran/10/0/models"
["10_gen_terran_seed_1"]="results_smac2_unit_ranges/10gen_terran/10/1/models"
["10_gen_terran_seed_2"]="results_smac2_unit_ranges/10gen_terran/10/2/models"
["20_gen_terran_seed_0"]="results_smac2_unit_ranges/10gen_terran/20/0/models"
["20_gen_terran_seed_1"]="results_smac2_unit_ranges/10gen_terran/20/1/models"
["20_gen_terran_seed_2"]="results_smac2_unit_ranges/10gen_terran/20/2/models"
["5_gen_zerg_seed_0"]="results_smac2_unit_ranges/10gen_zerg/5/0/models"
["5_gen_zerg_seed_1"]="results_smac2_unit_ranges/10gen_zerg/5/1/models"
["5_gen_zerg_seed_2"]="results_smac2_unit_ranges/10gen_zerg/5/2/models"
["10_gen_zerg_seed_0"]="results_smac2_unit_ranges/10gen_zerg/10/0/models"
["10_gen_zerg_seed_1"]="results_smac2_unit_ranges/10gen_zerg/10/1/models"
["10_gen_zerg_seed_2"]="results_smac2_unit_ranges/10gen_zerg/10/2/models"
["20_gen_zerg_seed_0"]="results_smac2_unit_ranges/10gen_zerg/20/0/models"
["20_gen_zerg_seed_1"]="results_smac2_unit_ranges/10gen_zerg/20/1/models"
["20_gen_zerg_seed_2"]="results_smac2_unit_ranges/10gen_zerg/20/2/models"
)
echo ${weight_location["20_gen_zerg_seed_0"]}
threads=${4:-10000}
args=${5:-}
gpus=${6:-0,1,2,3,4,5,6,7}
times=${7:-3}  # corresponds to seeds.

maps=(${maps//,/ })
units=(${units//,/ })
gpus=(${gpus//,/ })
args=(${args//,/ })


if [ ! $config ] || [ ! $tag ]; then
    echo "Please enter the correct command."
    echo "./smac_1 qmix (debug|train|evalute)"
    exit 1
fi

echo "CONFIG:" $config
echo "MAP LIST:" ${maps[@]}
echo "THREADS:" $threads
echo "ARGS:"  ${args[@]}
echo "GPU LIST:" ${gpus[@]}
echo "TIMES:" $times

declare -A buffer_size
buffer_size=(
["train"]=8192
["evaluate"]=4096
["debug"]=8
)

# run parallel
count=0
for map in "${maps[@]}"; do
    for unit in "${units[@]}"; do
        for((seed=0;seed<times;seed++)); do
            gpu=${gpus[$(($count % ${#gpus[@]}))]}
            ./run_docker.sh $gpu 2 python3 src/main.py \
            --config="$config" \
            --env-config="$map" \
            with \
            use_wandb=False \
            env_args.capability_config.n_units=$unit \
            env_args.capability_config.n_enemies=$unit \
            checkpoint_path="${weight_location["${unit}_${map:4}_seed_${seed}"]}" \
            evaluate=True \
            buffer_size=${buffer_size["${tag}"]} \
            test_nepisode=${buffer_size["${tag}"]} \
            save_eval_buffer=True \
            save_eval_buffer_path="smac2_dev_experiments/data/smac_2_unit_ranges" \
            saving_eval_seed=$seed \
            saving_eval_type=$tag \
            "${args[@]}" &

            count=$(($count + 1))
            if [ $(($count % $threads)) -eq 0 ]; then
                wait
            fi
            # sleep 5
        done
    done
done
wait

