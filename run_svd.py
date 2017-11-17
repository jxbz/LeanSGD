import os

epochs = 3
layers = 10
cmd = ('mpirun -n 4 -hostfile hosts --map-by ppr:1:node '
       'python train.py --layers={layers} --epochs={epochs} '
       '--compress={compress} --svd_rank={svd_rank} --svd_rescale={rescale}')

for svd_rank in [0, 1, 2, 4]:
    for compress in [1, 0]:
        for rescale in [0, 1]:
            if (not compress) and (rescale or svd_rank):
                continue
            run = cmd.format(layers=layers, epochs=epochs, compress=compress,
                             svd_rank=svd_rank, rescale=rescale)
            print(run)
            os.system(run)
