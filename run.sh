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
tag=$2
maps=${3:-27m_vs_30m,8m_vs_9m,5m_vs_6m,corridor,2c_vs_64zg,10m_vs_11m,6h_vs_8z,MMM2,3s5z_vs_3s6z}   # MMM2 left out
threads=${4:-9} # 2
# td_lambdas=${9:-0.6}
# eps_anneals=${10:-100000}
args=${5:-}    # ""
gpus=${6:-0,1,2,3,4,5,6,7}    # 0,1
times=${7:-3}   # 5

maps=(${maps//,/ })
gpus=(${gpus//,/ })
args=(${args//,/ })
td_lambdas=(${td_lambdas//,/ })
eps_anneals=(${eps_anneals//,/ })

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
    for((i=0;i<times;i++)); do
                    
        gpu=${gpus[$(($count % ${#gpus[@]}))]}  
        group="${config}-${map}-${tag}"
        ./run_docker.sh $gpu python3 src/main.py --config="$config" --env-config="sc2" with group="$group" use_wandb=True env_args.map_name=$map save_model=True env_args.obs_starcraft=False env_args.obs_timestep_number=True "${args[@]}" &
                    
        count=$(($count + 1))     
                    
        if [ $(($count % $threads)) -eq 0 ]; then
            wait
        fi
        # for random seeds
        sleep $((RANDOM % 3 + 3))
    done
done
wait
