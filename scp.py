import os

key_file = '/Users/scott/Work/Developer/AWS/scott-key-dim.pem'
DNSs = ['ec2-54-218-89-5.us-west-2.compute.amazonaws.com',
        'ec2-34-210-73-148.us-west-2.compute.amazonaws.com',
        'ec2-54-191-209-162.us-west-2.compute.amazonaws.com']

cmd = 'scp -i {key_file} *.py hosts_file ec2-user@{dns}:~/WideResNet-pytorch/'

for dns in DNSs:
    run = cmd.format(key_file=key_file, dns=dns)
    os.system(run)
