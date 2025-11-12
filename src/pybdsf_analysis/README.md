<h1>Namespace pybdsf_analysis</h1>

<hr>

The namespace pybdsf_analysis is used for defining classes that can be used in scripts meant to run PyBDSF analysis on a folder of fits files.

This namespace defines three classes:
 - RecursiveFileAnalyzer
    - A class which can either perform a function on files within its directory or unwrap all files in its directory. Supports extention filtering.
 - HistogramErrorDrawer
    - A utility class to draw a histogram with error bars according to astropy.stats.poisson_conf_interval
 - ImageAnalyzer( RecursiveFileAnalyzer )
    - A general-purpose PyBDSF fits image analyzer. Can also convert raw data to a FITS image with the defaults for this project.

These classes are implemented in the <b>scripts</b> namespace