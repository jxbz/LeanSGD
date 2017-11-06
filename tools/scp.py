import subprocess
import sys
import os

DNSs = ['ec2-54-191-123-251.us-west-2.compute.amazonaws.com',
        'ec2-34-213-224-63.us-west-2.compute.amazonaws.com',
        'ec2-34-215-180-57.us-west-2.compute.amazonaws.com']

key_file = '/Users/scott/Work/Developer/AWS/scott-key-dim.pem'

def up():
    for dns in DNSs:
        cmd = f'scp -i {key_file} ../*.py * ec2-user@{dns}:~/WideResNet-pytorch'
        os.system(cmd)

def down():
    outdir = '../WideResNet-pytorch-out/output/2017-11-03'
    os.system(f'rm -rf {outdir}')
    os.system(f'mkdir {outdir}')
    for i, dns in enumerate(DNSs):
        #  os.mkdir('../../WideResNet-pytorch-out/2017-11-02')
        #  os.system(f'mkdir {outdir}/{i}')

        cmd = f'scp -i {key_file} -r ec2-user@{dns}:~/WideResNet-pytorch/output/2017-11-03 {outdir}/{i}'
        os.system(cmd)

if __name__ == "__main__":
    fn = sys.argv[1]
    if fn == 'down':
        down()
    elif fn == 'up':
        up()

