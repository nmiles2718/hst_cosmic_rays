# HST Cosmic Rays
The goal of this project is to analyze and characterize cosmic ray radiation using HST. We take a class based approach for generating the cosmic ray labels that are used to compute a variety of descriptive data. The data we extract are on a per image and a per cosmic ray basis

### Data extracted
- A measure of the cosmic ray's size using a metric equvivalent to a guassian sigma
- A measure of the cosmic ray's symmetry
- All of the pixels each cosmic ray affects
- The cosmic ray intensity in (CR/s)
- The angle of incidence of the cosmic ray


### Pipeline Structure
The configuration for the pipeline is set in the `pipeline_confi.yaml` file and contains all the instrument specific information required to run the pipeline on various HST instruments (e.g. ACS, WFC3, COS, STIS).
The labeling pipeline is controlled by the `do_labeling.py` script and it is the pipeline used to extract the desired information from the cosmic rays.
```console
(astroconda3) nmiles@nathan:lib$python do_labeling.py -h
usage: do_labeling.py [-h] [-acs] [-wfc3] [-stis] [-cos] [-initialize]

optional arguments:
  -h, --help   show this help message and exit
  -acs         Analyze ACS data
  -wfc3        Analyze WFC3 data
  -stis        Analyze STIS data
  -cos         Analyze COS data
  -initialize  Use this flag if this is the first time the pipeline has been
               run. It will ensure all HDF5 files are properly initialized
               with the correct structure.
  ```