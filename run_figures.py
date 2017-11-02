import subprocess

layers = 52
cmd = 'python train.py --layers={layers} --epochs={epochs}'
for svd_rank in [1, None, 2, 4, 8]:
    if svd_rank == 1:
        epochs = 180
    else:
        epochs = 120
    if svd_rank is not None:
        run = cmd + ' --svd_rank={svd_rank}'
        kwargs = {'svd_rank': svd_rank}
    else:
        run = cmd + ''
        kwargs = {}
    run = run.format(epochs=epochs, layers=layers, **kwargs)
    subprocess.call(run.split(' '))

# {1: 38e3, 2: 72e3}
