#!/usr/bin/env bash

# Copy credentials to EC2 instanace
ec2_ip=$1  # first arg is IP address of instance
aws_dir=~/.aws/
git_credentials=~/.ssh/
aws_ssh_key=$aws_dir.aws_ssh/cosmic_rays_ssh_useast1.pem

# Copy files credentials to AWS
scp -ri "$aws_ssh_key" $aws_dir ec2-user@$ec2_ip:~
scp -ri "$aws_ssh_key" $git_credentials ec2-user@$ec2-ip:~

# log in to the instance
ssh -i "$aws_ssh_key" ec2-user@$ec2_ip