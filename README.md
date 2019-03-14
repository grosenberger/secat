SECAT: Size-Exclusion Chromatography Algorithmic Toolkit
============

*SECAT* is an algorithm for the network-centric data analysis of SEC-SWATH-MS data. The tool is implemented as a multi-step command line application.

Installation
------------

We strongly advice to install *SECAT* in a Python [*virtualenv*](https://virtualenv.pypa.io/en/stable/). *SECAT* is compatible with Python 3.7 and higher.

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

*SECAT* consists of the following steps:


**1. Data preprocessing**

First, the input quantitative proteomics matrix and parameters are preprocessed to a single file:

````
secat preprocess
--out=ccl2kyoto_string_negative_spectronaut.secat \ # Output filename
--sec=input/ccl2kyoto_sec_mw_spectronaut.csv \ # SEC annotation file
--net=input/9606.protein.links.v10.5.txt.gz \ # Reference PPI network
--posnet=input/corum_targets.txt \ # Reference positive interaction network for learning
--negnet=input/corum_decoys.txt \ # Reference negative interaction network for learning
--uniprot=input/uniprot_9606_20180420.xml.gz \ # Uniprot reference XML file
--columns run_id sec_id sec_mw condition_id replicate_id R.FileName PG.ProteinAccessions EG.PrecursorId FG.Quantity \ # Columns of the proteomics data
input/Spectronaut/20181028_021159_SEC_HeLa_CCL2_Kyoto_peptides_swnormalized.tsv \ # Input data files
````

**2. Signal processing**

Next, the signal processing is conducted in a parallelized fashion:

````
secat score --in=ccl2kyoto_string_negative_spectronaut.secat --threads=8
````

**3. PPI detection**

The statistical confidence of the PPI is evaluated by machine learning:

````
secat learn --in=ccl2kyoto_string_negative_spectronaut.secat --threads=5
````

**4. PPI quantification**

Quantitative features are generated for all PPIs and proteins:

````
secat quantify --in=ccl2kyoto_string_negative_spectronaut.secat --control_condition=ccl2
````

**5. Export of results**

CSV tables can be exported for import in downstream tools, e.g. Cytoscape:

````
secat export --in=ccl2kyoto_string_negative_spectronaut.secat
````

**6. Plotting of chromatograms**

PDF reports can be generated for the top (or selected) results:

````
secat plot --in=ccl2kyoto_string_negative_spectronaut.secat
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
````
