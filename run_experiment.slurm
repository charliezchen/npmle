#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=two_mixture
#SBATCH --time=8:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x-%j-%a.out
#SBATCH --mail-type=ALL

N_start=1000
N_step=1000

local_machine="CHARLLECHEN-MB0"
mixture=2
mixture_two_distance=1
folder="two_mixture_save_data_seeding"

if [ $(hostname) == $local_machine ]; then
    for i in {0..30}; do
        echo $(($N_start + $i * $N_step))
    done
else 
    mypython -u frank_wolfe.py --mixture $mixture \
        --mixture-two-distance $mixture_two_distance \
        --folder $folder --config config.yml \
        --N $((N_start + $SLURM_ARRAY_TASK_ID * $N_step))
fi

# python3 -u frank_wolfe.py --mixture 1 --folder one_mixture_save_data \
#             --config config.yml -N $N
# # python3 -u frank_wolfe.py --mixture 2 --mixture-two-distance 1 --folder two_mixture_1
# 
# 
# python3 -u frank_wolfe.py --mixture 1 --folder one_mixture_local \
#     --config config_small.yml --N $N

