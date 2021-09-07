# Analysis code for Blockus et al. *Cell Reports* 2021

This repository contains code used to ultimately generate Figure 5 from:

Blockus H., Rolotti S.V., Szoboszlay M., Peze-Heidsieck E., Ming T., Schroeder A., Apostolo N.,
Vennekens K.M., Katsamba P., Bahna F., Mannepalli S., Ahlsen G., Honig B., Shapiro L., de Wit J.,
Losonczy A., & Polleux F.
Synaptogenic activity of the axon guidance molecule Robo2 underlies hippocampal circuit function.
*Cell Reports* 2021.

## Overview
The code contained here includes: 

* preprocessing steps after
[Suite2p](https://github.com/MouseLand/suite2p) motion correction and [SIMA](https://github.com/losonczylab/sima) signal extraction.
This includes spike deconvolution and place field calculations.

* A minimal version of the internal Losonczy Lab repo with all code necessary to enable pre-processing and analysis steps.
Note that this does not include the SQL database needed for storing all experiment metadata, these details would need to be filled in
for your own database at mini_repo.classes.database.ExperimentDatabase.connect().
See an older but far more thorough version of this repo [here](https://github.com/jzaremba/Zaremba_NatNeurosci_2017) for more details about installation and use.

* Data calculation scripts which load experiments used for analysis and save pre-computed metrics to be plotted.

* Plotting scripts to actually generate the panels of this figure (mostly contained in a jupyter notebook)

Repository layout:

    .
    ├── pre_processing/               # Files for spike deconvolution and place field detection
    ├── mini_repo/                    # Core mwe code for all processing and analysis
    ├── analysis_computation_scripts/ # Scripts to pre-compute metrics for later plotting
    ├── figures/                      # Scripts for generating figures
    └── README.md

### Raw data

Note that the raw data and metadata are not included here.
This means that many scripts will not run immediately, as they currently include calls to data/metadata that are not included. In particular, metadata calls require a SQL database instance, while data calls would require the ROIs and extracted signals for each experiment as well as the corresponding behavior data.

These steps are retained here to demonstrate what data was included in each analysis in the paper as well as to serve as a template for
your own data/analysis. To request the data from these experiments, or for more information on how to run your own analysis with these tools, please contact the authors.
