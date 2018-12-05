Now there are only two ways to submit jobs on Taito clusters

# 1. Directly Run Program in the Terminal.
An example could be:

```
module load python-env/2.7.10-ml
export PYTHONPATH=$PYTHONPATH:$WRKDIR/DONOTREMOVE/git/FeatureLearningAndGestureRecognition/src
srun -N 1 -n 1 --mem-per-cpu=48000 -t72:00:00 --gres=gpu:p100:1 -p gpu python Blstm_rawJoint.py
```
