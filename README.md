# HST Cosmic Rays
This repository contains the HSTcosmicrays package that is used in the analysis
of every identifiable cosmic ray in any of the calibration darks taken by the 
following imagers on HST:
**CCD Imagers**
- ACS/WFC
- ACS/HRC
- STIS/CCD
- WFPC2 (all four detectors PC, WF1, WF2, and WF3)
- WFC3/UVIS

**IR Imagers (Work in progress)**
- NICMOS (all three detectors NIC1, NIC2, and NIC3)
- WFC3/IR

The package has been designed to work locally and in the cloud on an AWS EC2 
instance.

#### Cosmic ray data extracted
- The width of the cosmic ray's energy distribution in sigmas
- The size of the cosmic ray in pixels (i.e. the number of pixels it affects) 
- A measure of the symmetry of the cosmic ray's energy distribution
- A list of all the pixels ever affected by the cosmic ray. 
- The cosmic ray incidence rate in CRs/s/cm<sup>2</sup> for the exposure.
- The total energy deposited by the cosmic ray in electrons.

#### Image metadata extracted
- Altitude of HST
- Date of the observation (YYYY-MM-DD HH:MM:SS)
- Exposure start time
- Exposure end time
- Latitude of HST at one minute intervals throughout the exposure
- Longitude of HST at one minute intervals throughout the exposure
- Integration time
  - EXPTIME + FLASHDUR (if relevant) + READOUTIME


#### Pipeline Structure
The configuration for the pipeline is set in the `pipeline_confi.yaml` file and 
contains all the instrument specific information required to run the pipeline on 
various HST instruments (e.g. ACS, WFC3, COS, STIS). The pipeline should be 
initiated from the command line by passing the desired command line args:
```console
(astroconda3) [nmiles@:nathan pipeline]$ python pipeline.py -h
usage: pipeline.py [-h] [-aws] [-download] [-process] [-ccd] [-ir]
                   [-chunks CHUNKS] [-analyze] [-use_dq] [-instr INSTR]
                   [-initialize] [-store_downloads]

optional arguments:
  -h, --help        show this help message and exit
  -aws              Flag for using AWS for downloads. Only to be used when the
                    pipeline is run on EC2
  -download         Download the data
  -process          Process the raw data
  -ccd              Switch for process CCD data
  -ir               Switch for process IR data
  -chunks CHUNKS    Number of chunks to break the entire dataset into
  -analyze          Run the analysis and extract cosmic ray statistics
  -use_dq           Use the DQ arrays to perform labeling
  -instr INSTR      HST instrument to process (acs_wfc, wfc3_uvis, stis_ccd,
                    acs_hrc)
  -initialize       Initialize the HDF5 files for the instrument. **Warning**:
                    Should only be included the first time the pipeline is run
                    becuase it will overwrite any pre-existing HDF5 files.
  -store_downloads  Switch for toggling on the storage of downloaded data

  ```
  It is designed to be lightweight with respect to data storage and so after
  each month-chunk of darks has been analyzed the pipeline will delete all 
  the downloaded files. This has the benefit of only requiring a modestly sized
  EC2 instance when run in the cloud on AWS. The `-store_downloads` argument 
  is currently a placeholder argument that will allow users to specify whether 
  or not they would like to retain all of the downloaded observations.