import subprocess

def train(cmd, **kwargs):
    run = cmd.format(**kwargs)
    subprocess.call(run.split(' '))

cmd = 'python train.py --approx_grad={approx_grad} --layers={layers} --epochs={epochs} --seed={seed}'

epochs = 10

for repeat in range(5):
    for layers in [34]:
        for approx_grad in [0, 1]:
            if approx_grad == 0:
                train(cmd, epochs=epochs, approx_grad=approx_grad,
                      layers=layers, seed=repeat)
                continue
            for rank in [1, 2, 4, 8]:
                train(cmd + ' --svd_rank={rank}', epochs=epochs, rank=rank,
                      approx_grad=approx_grad, layers=layers, seed=repeat)
