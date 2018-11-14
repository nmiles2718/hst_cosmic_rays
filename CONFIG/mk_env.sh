#!/usr/bin/env bash

ec2_ip=$1  # first arg is IP address of instance
aws_dir=~/.aws/
aws_ssh_key=$aws_dir.aws_ssh/cosmic_rays_ssh_useast1.pem


# log in to the instance
ssh -i "$aws_ssh_key" ec2-user@$ec2_ip

# Create an environment for the pipeline
bash conda create --name aws_env --file ./CONFIG/aws_env.txt
