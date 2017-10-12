import subprocess

command = 'python train.py --layers={layers} --epochs={epochs}'
args = {'epochs': 2}
for layers in range(28, 120, 6):
    if layers > 50:
        args['epochs'] = 1
    args['layers'] = layers
    run = command.format(**args)
    subprocess.call(run.split(' '))
