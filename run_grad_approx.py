import subprocess

def train(cmd, **kwargs):
    run = cmd.format(**kwargs)
    subprocess.call(run.split(' '))

cmd = 'python train.py --approx_grad={approx_grad} --layers={layers} --epochs={epochs} --seed={seed} --num_workers=8'

epochs = 30

repeat = 0
for layers in [10]:
    for approx_grad in [1, 0]:
        if approx_grad == 1:
            for rank in [2, 4, 6, 8]:
                train(cmd + ' --svd_rank={rank}', epochs=epochs, rank=rank,
                      approx_grad=approx_grad, layers=layers, seed=repeat)
        elif approx_grad == 0:
            train(cmd, epochs=epochs, approx_grad=approx_grad,
                  layers=layers, seed=repeat)
