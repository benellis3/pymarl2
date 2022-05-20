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
tag=$2  # Used by wandb if use_wandb=True.
maps=${3:-2c_vs_64zg,8m_vs_9m,3s_vs_5z,5m_vs_6m,3s5z_vs_3s6z,corridor,6h_vs_8z,27m_vs_30m,MMM2}

# Edit this.
declare -A weight_location
weight_location=(
  ["2c_vs_64zg"]="results/models/qmix__2022-04-14_20-40-09"
  ["8m_vs_9m"]="results/models/qmix__2022-04-14_20-43-08"
  ["3s_vs_5z"]="results/models/qmix__2022-04-14_20-47-43"
  ["5m_vs_6m"]="results/models/qmix__2022-04-15_22-45-18"
  ["3s5z_vs_3s6z"]="results/models/qmix__2022-04-15_22-50-00"
  ["corridor"]="results/models/qmix__2022-04-15_22-53-45"
  ["6h_vs_8z"]="results/models/qmix__2022-04-16_21-32-19"
  ["27m_vs_30m"]="results/models/qmix__2022-04-16_21-37-23"
  ["MMM2"]="results/models/qmix__2022-04-18_18-02-35"
)

threads=${4:-8} # 2
args=${5:-}    # ""
gpus=${6:-0,1,2,3,4,5,6,7}    # 0,1
times=${7:-1}   # 5

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


# run parallel
count=0
for map in "${maps[@]}"; do
    for((i=0;i<times;i++)); do
        gpu=${gpus[$(($count % ${#gpus[@]}))]}  
        group="${config}-${map}-${tag}"
        ./run_docker.sh $gpu python3 src/main.py \
          --config="$config" \
          --env-config=sc2 \
          with \
          env_args.map_name="$map" \
          group="$group" \
          use_wandb=False \
          checkpoint_path="${weight_location["$map"]}" \
          evaluate=True \
          buffer_size=8192 \
          test_nepisode=8192 \
          save_eval_buffer=True \
          save_eval_buffer_path="smac2_dev_experiments/data/smac1" \
          "${args[@]}" &

        count=$(($count + 1))     
        if [ $(($count % $threads)) -eq 0 ]; then
            wait
        fi
        # for random seeds
        sleep 5
    done
done
wait
