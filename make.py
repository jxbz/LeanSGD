import os
import subprocess


cmd = ('python train.py --widen-factor=1 --layers=10 '
       '--seed={seed} --num_workers={num_workers} --epochs=10')
for seed in range(10):
    for num_workers in [2, 1, 3, 4, 6, 8, 16]:
        run = cmd.format(seed=seed, num_workers=num_workers)
        print(run)
        subprocess.call(run.split(' '))

# increasing the depth of the network decreases the speedup observed.
# SOmething to do with GPU memory?
