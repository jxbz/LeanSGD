import os

layers = 52
cmd = ('mpirun -n {n_workers} -hostfile hosts --map-by ppr:1:node '
       'python train.py --layers={layers} --epochs=0 --lr=0.01 '
       '--momentum=0 --nesterov=0 --weight-decay=0 --widen-factor=6')

args = [f'--compress=1 --svd_rank={rank} --svd_rescale=1' for rank in [2, 4, 8]]
args += ['--compress=0', '--qsgd=1']
cmds = [cmd + ' ' + arg for arg in args]

#  for n_workers in [8, 4, 2]:
n_workers = 4
for cmd in cmds:
    run = cmd.format(n_workers=n_workers, layers=layers)
    print('#'*30 + '\n' + run)
    os.system(run)
