"""
Runs benchmark to see how long gradient and step_size computations take.

We would expect

    grad_time = O(num_params)
    step_size = O(num_params)

In the steady state when there are many parameters. We would not expect this when
there are few parameters. There are two scenarios to consider:

* there is no bottleneck and PyTorch can make effective use of caches
* there is significant overhead

We would expect the * slope * for small `num_params` to be smaller than large
params, but with more overhead.

Reasons why single machine memory bandwidth matters:

* GPU-CPU communication is slow(~5GB/s). L1 cache speed is about 25GB/s.
  L3 cache is about 5GB/s
* Use of caches can give 10-30x speedup

These are multiplicative.
"""
import subprocess

layers = [10, 16, 22, 28, 34, 40, 46, 52]
layers = range(10, 120, 6)
widen_factor = 1
for num_layers in layers:
    if (num_layers - 4) % 6 != 0:
        print(num_layers, (num_layers - 4) % 6)

cmd = ('python train.py --layers={layers} --epochs={epochs} --seed={seed} '
       '--widen-factor={widen_factor}')

epochs = 2
seed = 42
for num_layers in layers:
    print('#'*10 + '\n' + f'widen_factor = {widen_factor}, layers = {num_layers}')
    run = cmd.format(epochs=epochs, layers=num_layers, widen_factor=widen_factor,
                     seed=seed)
    subprocess.call(run.split(' '))
