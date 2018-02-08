import os

epochs = 0
layers = 94
cmd = ('mpirun -n {n_workers} -hostfile hosts --map-by ppr:1:node '
       'python train.py --layers={layers} --epochs={epochs} ')

args = ['--compress=1 --svd_rank=0 --svd_rescale=1',
        '--qsgd=1',
        '--compress=0']
for n_workers in [1, 2, 4, 8, 16]:
    for arg in args:
        run = cmd + arg
        run = run.format(layers=layers, epochs=epochs, n_workers=n_workers)
        print(run)
        os.system(run)

# 1 worker: tSVD
