import subprocess

command = 'python train.py --approx_grad={approx_grad} --layers={layers} --epochs={epochs} --seed={seed}'

epochs = 10

for repeat in range(5):
    for layers in [10, 40, 160]:
        for approx_grad in [0, 1]:
            run = command.format(epochs=epochs, approx_grad=approx_grad,
                                 layers=layers, seed=repeat)
            subprocess.call(run.split(' '))
