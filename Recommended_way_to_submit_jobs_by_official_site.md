Now there are only two ways to submit jobs on Taito clusters

# 1. Directly Run Program in the Terminal.
An example could be:

```
module load python-env/2.7.10-ml
export PYTHONPATH=$PYTHONPATH:$THE_PATH_OF_YOUR_PROJECT_FOLDER
srun -N 1 -n 1 --mem-per-cpu=48000 -t72:00:00 --gres=gpu:p100:1 -p gpu python Blstm_rawJoint.py
```

The second line will add your project directory to the PYTHONPATH, so that you own functions can be called properly.


# 2. Submit Batch Job
An example could be:

```
sbatch xxx.sh
```

And the content inside of xxx.sh could be:

```
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem-per-cpu=48000
#SBATCH

module purge
module load gcc cuda
module list

srun python Blstm_rawJoint.py
```

