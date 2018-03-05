import os
from pprint import pprint

pre = 'mpirun -n {n_workers} -hostfile hosts --map-by ppr:1:node '
#  pre = ''
cmd = (pre + 'python train.py --epochs=0 --lr=0.001 '
       '--momentum=0 --nesterov=0 --weight-decay=0 ')
width_depth = ['--widen-factor=4 --layers=52', '--layers=142']
cmds = [cmd + x for x in width_depth]

#  args = ['--code=qsgd','--code=qsvd --scheme=qsgd',
        #  '--code=terngrad']
#  args += [f'--code=svd --compress=1 --svd_rank={rank} --svd_rescale=1'
         #  for rank in [3, 6]]
#  args += ['--compress=0 --code=sgd']
#  args = ['--code=svd --compress=1 --svd_rank=6 --svd_rescale=1',
        #  '--compress=0 --code=sgd']
args = ['--code=svd --svd_rank=6', '--code=svd -svd_rank=3']
cmds = [cmd + ' ' + arg for arg in args for cmd in cmds]
pprint(cmds)
print("len(cmds) =", len(cmds))

for n_workers in [2]:
    for cmd in cmds:
        run = cmd.format(n_workers=n_workers)
        print('#'*30 + '\n' + run)
        os.system(run)
