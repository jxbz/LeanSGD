import os

cmd = ('mpiexec -n 2 -hostfile hosts --map-by ppr:1:node '
       'python train.py --layers=10 --compress={compress} --seed={seed} --epochs=13')
for seed in range(6):
    for compress in [0, 1]:
        run = cmd.format(seed=seed, compress=compress)
        os.system(run)
