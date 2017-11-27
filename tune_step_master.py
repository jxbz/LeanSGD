import os
import sys
from pprint import pprint

keyfile = '~/Work/Developer/AWS/scott-key-dim.pem'
run_on_node = 'ssh -i {keyfile} ec2-user@{dns} "{cmd}"'

with open('DNSs', 'r') as f:
    DNSs = [line.strip('\n') for line in f.readlines()]
print("Available DNSs:")
pprint(DNSs)

if len(sys.argv) != 2:
    print('Usage: python cluster.py [command]')
    sys.exit(1)
script_cmd = sys.argv[1]
if script_cmd == 'tune':
    cmd = ('cd WideResNet-pytorch; '
           'python tune_step.py --compress={compress} --svd_rank={rank} '
           '--svd_rescale={rescale} > tune.out 2>&1 &')

    dns_idx = 0
    run = cmd.format(compress=0, rank=-1, rescale=0)
    os.system(run_on_node.format(cmd=run, keyfile=keyfile, dns=DNSs[dns_idx]))
    dns_idx += 1
    for svd_rank in [0, 1, 2, 4, 8]:
        for svd_rescale in [0, 1]:
            if (not svd_rescale) and (svd_rank == 0):
                continue
            run = cmd.format(compress=1, rank=svd_rank,
                             rescale=svd_rescale)
            print(f"Running {run} on {DNSs[dns_idx]}")
            os.system(run_on_node.format(cmd=run, keyfile=keyfile,
                                         dns=DNSs[dns_idx]))
            dns_idx += 1

elif script_cmd == 'download':
    cmd = ('scp -r -i {keyfile} '
           'ec2-user@{dns}:~/WideResNet-pytorch/output/2017-11-20 '
           '~/Desktop/output/{i}/')
    os.system('mkdir ~/Desktop/output')
    for i, dns in enumerate(DNSs):
        os.system(f'mkdir ~/Desktop/output/{i}')
        os.system(cmd.format(keyfile=keyfile, dns=dns, i=i))

else:
    raise ValueError('Command not recognized')
