import os

cmd = ('mpiexec -n 2 -hostfile hosts --map-by ppr:1:node '
       'python train.py --layers=10 --compress={compress} --seed={seed} --epochs=15')
for compress in [0, 1]:
    for seed in [1, 2, 3]:
        run = cmd.format(seed=seed, compress=compress)
        os.system(run)
