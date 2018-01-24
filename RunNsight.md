srun -N 1 -n 1 --mem-per-cpu=32000 -t72:00:00 --gres=gpu:p100:1 -p gpu --x11=first /appl/opt/cuda/9.0/bin/nsight
