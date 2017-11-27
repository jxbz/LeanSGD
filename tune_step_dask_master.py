import os

# assumes been launched already
# assumes one node has `dask-scheduler` run, IP from that
ip = '172.31.2.67:8786'
command = 'dask-worker --nprocs=1 --nthreads=1 {ip} >> ~/WideResNet-pytorch/dask.out 2>&1 &'
os.system(f'python cluster.py custom "{command}"')

# run on head node:
# sudo /home/ec2-user/anaconda3/bin/python tune_step.py
