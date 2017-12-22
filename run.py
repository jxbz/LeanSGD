import os

cmd = ('mpirun -n 2 -hostfile hosts --map-by ppr:1:node '
       'python train.py --layers=94 --epochs=20 --lr=0.05 '
       '--momentum=0 --nesterov=0 --weight-decay=0')
args = ['--compress=0', '--compress=1 --svd_rank=0 --svd_rescale=1', '--qsgd=1']
cmds = [cmd + ' ' + arg for arg in args]
for run in cmds:
    print('#'*30 + '\n' + run)
    os.system(run)
