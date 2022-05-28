#!/bin/bash
trap 'onCtrlC' INT

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
maps=${3:-2c_vs_64zg,8m_vs_9m,3s_vs_5z,5m_vs_6m,3s5z_vs_3s6z,corridor,6h_vs_8z,27m_vs_30m,MMM2}

# Edit this.
declare -A weight_location
weight_location=(
  ["27m_vs_30m_seed_0"]="results_smac1/models/qmix__2022-04-16_21-40-03"
  ["27m_vs_30m_seed_1"]="results_smac1/models/qmix__2022-04-16_21-38-35"
  ["27m_vs_30m_seed_2"]="results_smac1/models/qmix__2022-04-16_21-37-23"
  ["2c_vs_64zg_seed_0"]="results_smac1/models/qmix__2022-04-14_20-41-57"
  ["2c_vs_64zg_seed_1"]="results_smac1/models/qmix__2022-04-14_20-40-09"
  ["2c_vs_64zg_seed_2"]="results_smac1/models/qmix__2022-04-14_20-38-41"
  ["3s5z_vs_3s6z_seed_0"]="results_smac1/models/qmix__2022-04-15_22-52-30"
  ["3s5z_vs_3s6z_seed_1"]="results_smac1/models/qmix__2022-04-15_22-51-17"
  ["3s5z_vs_3s6z_seed_2"]="results_smac1/models/qmix__2022-04-15_22-50-00"
  ["3s_vs_5z_seed_0"]="results_smac1/models/qmix__2022-04-15_22-44-08"
  ["3s_vs_5z_seed_1"]="results_smac1/models/qmix__2022-04-14_20-49-08"
  ["3s_vs_5z_seed_2"]="results_smac1/models/qmix__2022-04-14_20-47-43"
  ["5m_vs_6m_seed_0"]="results_smac1/models/qmix__2022-04-15_22-48-20"
  ["5m_vs_6m_seed_1"]="results_smac1/models/qmix__2022-04-15_22-46-24"
  ["5m_vs_6m_seed_2"]="results_smac1/models/qmix__2022-04-15_22-45-18"
  ["6h_vs_8z_seed_0"]="results_smac1/models/qmix__2022-04-16_21-35-48"
  ["6h_vs_8z_seed_1"]="results_smac1/models/qmix__2022-04-16_21-33-55"
  ["6h_vs_8z_seed_2"]="results_smac1/models/qmix__2022-04-16_21-32-19"
  ["8m_vs_9m_seed_0"]="results_smac1/models/qmix__2022-04-14_20-46-20"
  ["8m_vs_9m_seed_1"]="results_smac1/models/qmix__2022-04-14_20-44-58"
  ["8m_vs_9m_seed_2"]="results_smac1/models/qmix__2022-04-14_20-43-08"
  ["corridor_seed_0"]="results_smac1/models/qmix__2022-04-16_21-30-41"
  ["corridor_seed_1"]="results_smac1/models/qmix__2022-04-16_21-29-39"
  ["corridor_seed_2"]="results_smac1/models/qmix__2022-04-15_22-53-45"
  ["MMM2_seed_0"]="results_smac1/models/qmix__2022-04-18_18-05-44"
  ["MMM2_seed_1"]="results_smac1/models/qmix__2022-04-18_18-04-20"
  ["MMM2_seed_2"]="results_smac1/models/qmix__2022-04-18_18-02-35"
)

threads=${4:-8}
args=${5:-}
gpus=${6:-0,1,2,3,4,5,6,7}
times=${7:-3}

maps=(${maps//,/ })
gpus=(${gpus//,/ })
args=(${args//,/ })

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

declare -A buffer_size
buffer_size=(
["train"]=8192
["evaluate"]=4096
["debug"]=32
)

# run parallel
count=0
for map in "${maps[@]}"; do
    for((seed=0;seed<times;seed++)); do
        gpu=${gpus[$(($count % ${#gpus[@]}))]}  
        group="${config}-${map}-${tag}"
        ./run_docker.sh $gpu python3 src/main.py \
          --config="$config" \
          --env-config=sc2 \
          with \
          env_args.map_name="$map" \
          use_wandb=False \
          checkpoint_path="${weight_location["${map}_seed_${seed}"]}" \
          evaluate=True \
          buffer_size=$buffer_size \
          test_nepisode=$buffer_size \
          save_eval_buffer=True \
          save_eval_buffer_path="smac2_dev_experiments/data/smac_1/" \
          saving_eval_seed=seed \
          "${args[@]}" &

        count=$(($count + 1))     
        if [ $(($count % $threads)) -eq 0 ]; then
            wait
        fi
        sleep 5
    done
done
wait
