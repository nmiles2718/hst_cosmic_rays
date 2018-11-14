#!/usr/bin/env bash
ec2_ip=$1
aws_dir=~/.aws/
aws_ssh_key=$aws_dir.aws_ssh/cosmic_rays_ssh_useast1.pem


# Copy files credentials to AWS
scp -ri "$aws_ssh_key" $aws_dir ec2-user@$ec2_ip:~
