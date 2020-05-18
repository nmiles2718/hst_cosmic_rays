.. _hst_cosmic_rays:

===============
HSTcosmicrays
===============

**Author**: Nathan Miles

HSTcosmicrays is a package designed to extract morphological properties of cosmic rays identified in calibration dark frames. The package is broken down into a series of modules which are then combined via the :py:class:`~pipeline.CosmicRayPipeline` object contained in the :py:mod:`~pipeline` module. At the present moment, the pipeline will analyze all of the following CCDs imagers:

    * `Advanced Camera for Surveys (ACS) <https://hst-docs.stsci.edu/display/ACSIHB/>`_
      
      * High Resolution Channel (HRC) [inactive]
      * Wide Field Channel (WFC) [active]

    * `Space Telescope Imaging Spectrograph (STIS) <https://hst-docs.stsci.edu/display/STISIHB/STIS+Instrument+Handbook>`_
      
      * CCD channel [active]

    * `Wide Field Camera 3 (WFC3) <https://hst-docs.stsci.edu/display/WFC3IHB/WFC3+Instrument+Handbook>`_
      
      * UVIS channel [active]

    * `Wide Field and Planetary Camera 2 (WFPC2) <http://documents.stsci.edu/hst/wfpc2/documents/handbooks/cycle17/wfpc2_ihb.pdf>`_
      
      * Wide Field Detector 1, 2, and 3 (WF1, WF2, WF3) [inactive]
      * Planetary Camera (PC) [inactive]

In the future, we will extend the pipeline to handle the following IR imagers
as well:

    * `Near Infrared Camera and Multi-Object Spectrometer (NICMOS) <http://www.stsci.edu/hst/nicmos/documents/handbooks/current_NEW/nicmos_ihb.pdf>`_

      * Near Infrared Camera 1, 2, and 3 (NIC1, NIC2, NIC3) [inactive]


    * `Wide Field Camera 3 (WFC3) <https://hst-docs.stsci.edu/display/WFC3IHB/WFC3+Instrument+Handbook>`_

      * IR channel [active]


Overview
--------


The pipeline uses a configuration file to obtain variety of instrument specific configuration items, as well as some global configuration items that are subsequently used throughout the pipeline.  The pipeline is designed to be lightweight with respect to storage requirements of the *downloaded* data. A summary of the steps taken by the pipeline are as follows:

    #. Initialization (:py:mod:`~utils.initialize`)

       * Determine the proper date range based off information stored in configuration file

         * Generate a list of one month intervals and exclude any periods of prolonged inactivity (e.g. instrument failures)

       * Determine the date ranges, if any, that have already been analyzed. These will automatically be skipped by the pipeline

       * If specified, initialize a series of HDF5 files used to store all of the extracted data.

    #. Download (:py:mod:`~download.download`)

       * Submit a query to MAST searching for all files in a one month interval that match the filetypes specified in the configuration file.

       * Download all found files to the directory specified in the configuration file.

    #. Process (:py:mod:`~process.process`)

       * For CCD imagers on active instruments, the dark frames will be processed through the respective instruments cosmic ray rejection routine.

       * For IR imagers, the FITS file containing all of the N inividual reads and there corresponding extensions will be decomposed into a N different
         FITS file, one for each read.

    #. Label (:py:mod:`~label.labeler`)

       * Perform a connected-component labeling analysis to identify groups of cosmic ray affected pixels as a single distinct object.


    #. Analyze

       * Analyze all of the identified cosmic rays and compute a variety of
         statistics (:py:class:`~stat_utils.statshandler`)
       * For each observation extract relevant metadata (:py:class:`~utils.metadata`)


    #. Clean Up

       * After processing all dark frames found in the one month interval,
         write out the computed statistics and delete the downloaded data.


Contents:

.. toctree::
   :maxdepth: 2

   examples
   pipeline
   pipeline_config
   download
   label
   process
   stat_utils
   utils




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
