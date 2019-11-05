SECAT: Size-Exclusion Chromatography Algorithmic Toolkit
============

*SECAT* is an algorithm for the network-centric data analysis of SEC-SWATH-MS data. The tool is implemented as a multi-step command line application.

Dependencies
------------

*SECAT* depends on several Python packages (listed in ``setup.py``) and the ``viper`` R/Bioconductor package, accessed via ``rpy2``. SECAT has been tested on Linux (CentOS 7) and macOS (10.14) operating systems and might run on other versions too.

Please install ``viper`` from [Bioconductor](https://doi.org/doi:10.18129/B9.bioc.viper) prior to SECAT.

Installation
------------

We strongly advice to install *SECAT* in a Python [*virtualenv*](https://virtualenv.pypa.io/en/stable/). *SECAT* is compatible with Python 3.7 and higher and installation should require a few minutes with a correctly set-up Python environment.

Install the development version of *SECAT* from GitHub:

````
pip install git+https://github.com/grosenberger/secat.git@master
````

Install the stable version of *SECAT* from the Python Package Index (PyPI):

````
pip install secat
````


Running SECAT
-------------

SECAT requires 1-4h running time with a SEC-SWATH-MS data set of two conditions and three replicates each, covering about 5,000 proteins and 80,000 peptides on a typical desktop computer with 4 CPU cores and 16GB RAM.

The exemplary input data (``HeLa-CC.tgz`` and ``Common.tgz`` are required) can be found on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3515928.svg)](https://doi.org/10.5281/zenodo.3515928)

The data set includes the expected output as SQLite-files. Note: Since the ``PyProphet`` semi-supervised learning step is initialized by a randomized seed, the output might vary slightly from run-to-run with numeric deviations. To completely reproduce the results, the pretrained PyProphet classifier can be applied to as described in the ``secat learn`` step. The Zenodo repository contains all parameters and instructions to reproduce the SECAT analysis results of the other data sets.

*SECAT* consists of the following steps:


**1. Data preprocessing**

First, the input quantitative proteomics matrix and parameters are preprocessed to a single file:

````
secat preprocess
--out=hela_string_negative.secat \ # Output filename
--sec=input/hela_sec_mw.csv \ # SEC annotation file
--net=../common/9606.protein.links.v11.0.txt.gz \ # Reference PPI network
--posnet=../common/corum_targets.txt.gz \ # Reference positive interaction network for learning
--negnet=../common/corum_decoys.txt.gz \ # Reference negative interaction network for learning
--uniprot=../common/uniprot_9606_20190402.xml.gz \ # Uniprot reference XML file
--min_interaction_confidence=0 # Minimum interaction confidence
input/hela_normsw.tsv \ # Input data files
````

**2. Signal processing**

Next, the signal processing is conducted in a parallelized fashion:

````
secat score --in=hela_string_negative.secat --threads=8
````

**3. PPI detection**

The statistical confidence of the PPI is evaluated by machine learning:

````
secat learn --in=hela_string_negative.secat --threads=5
````

**4. PPI quantification**

Quantitative features are generated for all PPIs and proteins:

````
secat quantify --in=hela_string_negative.secat --control_condition=inter
````

**5. Export of results**

CSV tables can be exported for import in downstream tools, e.g. Cytoscape:

````
secat export --in=hela_string_negative.secat
````

**6. Plotting of chromatograms**

PDF reports can be generated for the top (or selected) results:

````
secat plot --in=hela_string_negative.secat
````

**7. Report of statistics**

Statistics reports can be generated for the top (or selected) results:

````
secat statistics --in=hela_string_negative.secat
````

**Further options and default parameters**

All options and the default parameters can be displayed by:
````
secat --help
secat preprocess --help
secat score --help
secat learn --help
secat quantify --help
secat export --help
secat plot --help
secat statistics --help
````
