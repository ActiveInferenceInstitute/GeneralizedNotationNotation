# Statistical Parametric Mapping (SPM) Software Package: A Comprehensive Overview

Statistical Parametric Mapping (SPM) is a comprehensive, free and open-source software package designed for analyzing brain imaging data sequences, representing over 30 years of continuous development by an international community of neuroimaging researchers. SPM has evolved into one of the most widely used platforms in neuroscience research, providing sophisticated tools for statistical analysis of neuroimaging data across multiple modalities.[1][2][3]

## Core Functionality and Design

SPM is built on the fundamental concept of **statistical parametric mapping**, which refers to the construction and assessment of spatially extended statistical processes used to test hypotheses about functional imaging data. The software employs a **mass univariate approach**, fitting statistical models at each voxel to identify brain regions showing significant activation or anatomical differences.[4][1]

The current version, **SPM12**, was first released in October 2014 and represents a major update containing substantial theoretical, algorithmic, structural and interface enhancements over previous versions. SPM25.01, the latest major release reported in January 2025, incorporates novel analysis methods, optimizations of existing methods, and improved practices for open science and software development.[2][3][5][6][7]

## Supported Imaging Modalities

SPM is designed for the analysis of multiple neuroimaging modalities:[8][9][1]

- **Functional Magnetic Resonance Imaging (fMRI)** - for studying brain function and activation patterns
- **Positron Emission Tomography (PET)** - for metabolic and neurotransmitter imaging
- **Single Photon Emission Computed Tomography (SPECT)** - for cerebral blood flow studies
- **Electroencephalography (EEG)** - for electrical brain activity analysis
- **Magnetoencephalography (MEG)** - for magnetic field measurements of neural activity

Recent developments in SPM25 include enhanced support for **Optically Pumped Magnetometers (OPMs)**, a major innovation in MEG that enables free movement during neural recordings.[6]

## Key Analysis Methods and Frameworks

### Statistical Parametric Mapping
SPM's core methodology involves three key modules:[10]
1. **Smoothing** neuroimaging data spatially or temporally
2. **Fitting voxel-wise general linear models (GLMs)**
3. **Correcting for multiple comparisons** using random field theory (RFT), false discovery rate (FDR), or permutation methods

### Dynamic Causal Modeling (DCM)
SPM pioneered **Dynamic Causal Modeling**, a framework for specifying models of effective connectivity among brain regions. DCM uses nonlinear state-space models in continuous time to estimate directed influences between neuronal populations, providing insights into how brain regions interact during specific tasks or conditions.[11][12][13][14]

### Voxel-Based Morphometry (VBM)
**Voxel-Based Morphometry** is an automated computational approach that measures differences in local concentrations of brain tissue through voxel-wise comparison of multiple brain images. VBM enables comprehensive assessment of anatomical differences throughout the entire brain, making it valuable for studying neurological and psychiatric disorders.[15][16][17][18]

## Software Architecture and Requirements

SPM is implemented as a suite of MATLAB functions and subroutines with some externally compiled C routines. **Key requirements include**:[9][19]

- **MATLAB R2007a (7.4) to R2023b (9.15)** - no special toolboxes required[5][9]
- **MEX files** - pre-compiled binaries provided for Windows (32/64-bit), Linux (64-bit), and macOS (64-bit)[5]
- **NIFTI-1 file format** for image data (also reads legacy Analyze format)[5]

A **standalone version** is also available using the MATLAB Compiler, eliminating the need for a MATLAB license.[7][5]

## Preprocessing Pipeline

SPM provides comprehensive preprocessing capabilities essential for neuroimaging analysis:[20][21]

1. **Realignment** - corrects for head motion using 6-parameter rigid body transformation[21]
2. **Slice-timing correction** - adjusts for differences in slice acquisition timing[21]
3. **Coregistration** - aligns functional and anatomical images[20]
4. **Segmentation** - separates brain tissue into gray matter, white matter, and CSF[21]
5. **Normalization** - transforms images to standardized stereotactic space[20]
6. **Smoothing** - applies Gaussian filtering to increase signal-to-noise ratio[20]

## Extensions and Toolboxes

The SPM ecosystem includes numerous **extensions and toolboxes** developed by the neuroimaging community:[22][23][24]

### Major Toolboxes
- **CAT12** - Computational Anatomy Toolbox for advanced morphometric analysis[22]
- **CONN** - Functional connectivity analysis[24]
- **Marsbar** - Region of interest analysis[23]
- **WFU PickAtlas** - Atlas-based ROI creation[23]
- **FieldTrip** - Advanced M/EEG analysis[24]
- **DPABI** - Data Processing & Analysis of Brain Imaging[24]

These extensions expand SPM's capabilities for specialized analyses while maintaining integration with the core SPM framework.

## Historical Development and Impact

SPM was originally developed by **Karl Friston** at the Medical Research Council Cyclotron Unit for statistical analysis of PET data, with the first version (SPM classic) released in 1991. The software introduced many foundational statistical methods in neuroimaging:[25][1][2]

- Voxel-wise application of General Linear Models to neuroimaging data
- Convolution modeling of fMRI using hemodynamic response functions
- Multiple comparisons correction using Random Field Theory
- Event-related fMRI methodology
- Bayesian statistical approaches for neural modeling

## Current Status and Future Directions

SPM continues active development with regular updates and enhancements. **SPM25.01** represents the latest major release, incorporating:

- Enhanced **Bayesian statistical methods** including Parametric Empirical Bayes (PEB) and Bayesian Model Reduction (BMR)[6]
- **Multi-Brain Toolbox** for population-average brain generation[6]
- Advanced **spectral decomposition methods** for M/EEG analysis[6]
- **Behavioral modeling** tools using Active Inference framework[6]

The software maintains its commitment to **open science principles**, being distributed freely under the GNU General Public License to promote collaboration across neuroimaging laboratories worldwide.[26][7]

SPM's comprehensive functionality, robust statistical foundations, and extensive community support have established it as an indispensable tool for neuroimaging research, continuing to drive advances in our understanding of brain structure and function across clinical and basic neuroscience applications.

[1] https://www.fil.ion.ucl.ac.uk/spm/
[2] https://arxiv.org/html/2501.12081v1
[3] https://www.emergentmind.com/articles/2501.12081
[4] https://www.fil.ion.ucl.ac.uk/spm/course/slides05-usa/pdf/Lec_03_GLM_Princ.pdf
[5] https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
[6] https://www.theoj.org/joss-papers/joss.08103/10.21105.joss.08103.pdf
[7] https://www.fil.ion.ucl.ac.uk/spm/docs/wikibooks/Download/
[8] https://hpc.nih.gov/apps/spm12.html
[9] https://www.mathworks.com/matlabcentral/fileexchange/68729-statistical-parametric-mapping
[10] https://pmc.ncbi.nlm.nih.gov/articles/PMC11962820/
[11] https://en.wikipedia.org/wiki/Dynamic_causal_modeling
[12] https://pmc.ncbi.nlm.nih.gov/articles/PMC2427062/
[13] https://pmc.ncbi.nlm.nih.gov/articles/PMC6711459/
[14] https://www.fil.ion.ucl.ac.uk/spm/docs/manual/dcm/dcm/
[15] https://en.wikipedia.org/wiki/Voxel-based_morphometry
[16] https://pmc.ncbi.nlm.nih.gov/articles/PMC6666603/
[17] https://pmc.ncbi.nlm.nih.gov/articles/PMC3570139/
[18] https://www.fil.ion.ucl.ac.uk/spm/course/slides20-oct/06_Voxel_Based_Morphometry.pdf
[19] https://github.com/spm/spm12
[20] https://andysbrainbook.readthedocs.io/en/latest/SPM/SPM_Short_Course/SPM_04_Preprocessing.html
[21] https://brainresearch.de/Methods/fMRI/Preprocessing/SPM_Preprocessing.html
[22] https://www.fil.ion.ucl.ac.uk/spm/ext/
[23] https://andysbrainbook.readthedocs.io/en/latest/SPM/SPM_Short_Course/SPM_Intermezzo_Toolboxes.html
[24] https://github.com/spm-toolboxes
[25] https://pmc.ncbi.nlm.nih.gov/articles/PMC3480642/
[26] https://www.osc.edu/resources/available_software/software_list/matlab/spm
[27] https://www.numberanalytics.com/blog/ultimate-guide-statistical-parametric-mapping-anatomical-research-methods
[28] https://mriquestions.com/uploads/3/4/5/7/34572113/spm12_manual.pdf
[29] http://brainimaging.waisman.wisc.edu/~oakes/spm/SPM99_Introduction.pdf
[30] https://spm1d.org
[31] https://jsheunis.github.io/2018-06-28-spm12-matlab-scripting-tutorial-1/
[32] https://www.nitrc.org/projects/bm_spm_viewer/
[33] https://www.fieldtriptoolbox.org/getting_started/othersoftware/spm/
[34] https://www.fil.ion.ucl.ac.uk/spm/docs/tutorials/vbm/VBM-getting-started/
[35] https://arxiv.org/abs/2501.12081
[36] https://www.fil.ion.ucl.ac.uk/spm/doc/spm12_manual.pdf
[37] https://www.numberanalytics.com/blog/spm-neuroimaging-analysis-techniques-applications
[38] https://pmc.ncbi.nlm.nih.gov/articles/PMC1994117/
[39] https://andysbrainbook.readthedocs.io/en/latest/CAT12/CAT12_Overview.html
[40] https://www.mps-ucl-centre.mpg.de/12310/moran_dcm.pdf
[41] https://www.numberanalytics.com/blog/ultimate-guide-spm-neuroimaging-techniques
[42] https://www.fil.ion.ucl.ac.uk/spm/docs/tutorials/dcm/dcm_fmri_first_level/
[43] https://www.fil.ion.ucl.ac.uk/spm/docs/tutorials/vbm/
[44] https://www.fil.ion.ucl.ac.uk/spm-statistical-parametric-mapping/
[45] https://www.youtube.com/watch?v=GcIvdqotdpY
[46] https://andysbrainbook.readthedocs.io/en/latest/SPM/SPM_Short_Course/SPM_06_Scripting.html
[47] https://www.fil.ion.ucl.ac.uk/spm/course/slides10-vancouver/02_General_Linear_Model.pdf
[48] https://www.youtube.com/watch?v=uO945o3yuL0
[49] https://brainimaging.waisman.wisc.edu/~oakes/teaching/Lectures/GLM_Analysis.pdf
[50] https://www.youtube.com/watch?v=zSqBoB1GrDk
[51] https://www.tnu.ethz.ch/fileadmin/user_upload/teaching/Methods_Models2016/02_Preprocessing_HS2016_Tutorial.pdf
[52] https://andysbrainbook.readthedocs.io/en/latest/SPM/SPM_Short_Course/SPM_Statistics/SPM_04_Stats_General_Linear_Model.html
[53] https://www.nitrc.org/projects/spm/
[54] https://www.youtube.com/watch?v=MWLDbsGeLVU
[55] https://www.youtube.com/watch?v=I7DHzbO8mvs
[56] https://web.conn-toolbox.org/resources/conn-extensions
[57] https://www.fil.ion.ucl.ac.uk/spm/docs/tutorials/fmri/event/preprocessing/