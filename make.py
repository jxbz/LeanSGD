import os
import subprocess


cmd = ('python train.py --widen-factor=1 --layers=28 '
       '--seed={seed} --num_workers={num_workers} --epochs=20')
cmd = ('python train.py --widen-factor=1 --layers=10 '
       '--seed={seed} --num_workers={num_workers} --epochs=20')
for seed in range(10):
    for num_workers in [1]:
        run = cmd.format(seed=seed, num_workers=num_workers)
        print(run)
        subprocess.call(run.split(' '))

# increasing the depth of the network decreases the speedup observed.
# SOmething to do with GPU memory?
