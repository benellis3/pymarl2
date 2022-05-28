#!/bin/bash
trap 'onCtrlC' INT
set -x
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
units=${8:-5,10,20}

# Edit this.
declare -A weight_location
weight_location=(
["5_gen_protoss_seed_0"]="results_smac2/leo/models/qmix__2022-05-21_16-49-53"
["5_gen_protoss_seed_1"]="results_smac2/leo/models/qmix__2022-05-21_16-49-49"
["5_gen_protoss_seed_2"]="results_smac2/leo/models/qmix__2022-05-21_16-49-45"
["10_gen_protoss_seed_0"]="results_smac2/orion/models/qmix__2022-05-21_16-30-30"
["10_gen_protoss_seed_1"]="results_smac2/orion/models/qmix__2022-05-21_16-30-23"
["10_gen_protoss_seed_2"]="results_smac2/orion/models/special_weights"
["20_gen_protoss_seed_0"]="results_smac2/saruman/models/qmix__2022-05-21_19-50-39"
["20_gen_protoss_seed_1"]="results_smac2/saruman/models/qmix__2022-05-21_19-50-34"
["20_gen_protoss_seed_2"]="results_smac2/saruman/models/qmix__2022-05-21_19-50-31"
["5_gen_terran_seed_0"]="results_smac2/leo/models/qmix__2022-05-21_16-50-06"
["5_gen_terran_seed_1"]="results_smac2/leo/models/qmix__2022-05-21_16-50-01"
["5_gen_terran_seed_2"]="results_smac2/leo/models/qmix__2022-05-21_16-49-57"
["10_gen_terran_seed_0"]="results_smac2/orion/models/qmix__2022-05-21_16-30-41"
["10_gen_terran_seed_1"]="results_smac2/orion/models/qmix__2022-05-21_16-30-36"
["10_gen_terran_seed_2"]="results_smac2/orion/models/qmix__2022-05-21_16-30-30"
["20_gen_terran_seed_0"]="results_smac2/gandalf/models/qmix__2022-05-21_19-16-20"
["20_gen_terran_seed_1"]="results_smac2/gandalf/models/qmix__2022-05-21_19-16-18"
["20_gen_terran_seed_2"]="results_smac2/saruman/models/qmix__2022-05-21_19-50-49"
["5_gen_zerg_seed_0"]="results_smac2/leo/models/qmix__2022-05-21_16-50-18"
["5_gen_zerg_seed_1"]="results_smac2/leo/models/qmix__2022-05-21_16-50-14"
["5_gen_zerg_seed_2"]="results_smac2/leo/models/qmix__2022-05-21_16-50-10"
["10_gen_zerg_seed_0"]="results_smac2/orion/models/qmix__2022-05-21_16-30-53"
["10_gen_zerg_seed_1"]="results_smac2/orion/models/qmix__2022-05-21_16-30-50"
["10_gen_zerg_seed_2"]="results_smac2/orion/models/qmix__2022-05-21_16-30-47"
["20_gen_zerg_seed_0"]="results_smac2/gandalf/models/qmix__2022-05-21_19-16-09"
["20_gen_zerg_seed_1"]="results_smac2/gandalf/models/qmix__2022-05-21_19-16-05"
["20_gen_zerg_seed_2"]="results_smac2/gandalf/models/special_weights"
)

threads=${4:-8}
args=${5:-}
gpus=${6:-0,1,2,3,4,5,6,7}
times=${7:-3}  # corresponds to seeds.

maps=(${maps//,/ })
units=(${units//,/ })
gpus=(${gpus//,/ })
args=(${args//,/ })


declare -A buffer_size
buffer_size=(
["train"]=8192
["evaluate"]=4096
["debug"]=32
)

if [ ! $config ] || [ ! $tag ]; then
    echo "Please enter the correct command."
    echo "bash run.sh config_name map_name_list (experinments_threads_num arg_list gpu_list experinments_num)"
    exit 1
fi

echo "CONFIG:" $config
echo "MAP LIST:" ${maps[@]}
echo "THREADS:" $threads
echo "ARGS:"  ${args[@]}
echo "GPU LIST:" ${gpus[@]}
echo "TIMES:" $times
echo "TDLAMBDAS:" ${td_lambdas[@]}
echo "EPSANNEALS:" ${eps_anneals[@]}

# run parallel
count=0
for map in "${maps[@]}"; do
    for unit in "${units[@]}"; do
        for((seed=0;seed<times;seed++)); do
            gpu=${gpus[$(($count % ${#gpus[@]}))]}
            ./run_docker.sh $gpu python3 src/main.py \
            --config="$config" \
            --env-config="$map" \
            with \
            use_wandb=False \
            env_args.capability_config.n_units=$unit \
            env_args.capability_config.start_positions.n_enemies=$unit \
            checkpoint_path="${weight_location["${unit}_${map:4}_seed_${seed}"]}" \
            evaluate=True \
            buffer_size=${buffer_size["${tag}"]} \
            test_nepisode=${buffer_size["${tag}"]} \
            save_eval_buffer=True \
            save_eval_buffer_path="smac2_dev_experiments/data/smac_2" \
            saving_eval_seed=$seed \
            saving_eval_type=$tag
            "${args[@]}" &

            count=$(($count + 1))
            if [ $(($count % $threads)) -eq 0 ]; then
                wait
            fi
            sleep 5
        done
    done
done
wait

