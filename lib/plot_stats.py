#!/usr/bin/env python

from astropy.io import fits
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class PlotData(object):
    def __init__(self, fname):
        self.fname = fname
        self.data = {}

    def read_file(self):
        with h5py.File(self.fname, mode='r') as fobj:
            grp_keys = list(f.keys())



if __name__ == '__main__':
    main()