#! /bin/bash

exptime1=$1
exptime2=$2
crsigmas=$3
initgues=$4
python run_cr_rejection.py 20 -dir1 /Users/nmiles/hst_cosmic_rays/analyzing_cr_rejection/"$exptime1"_clean/ -dir2 /Users/nmiles/hst_cosmic_rays/analyzing_cr_rejection/"$exptime2"_clean/ -crsigmas $crsigmas -initgues $initgues