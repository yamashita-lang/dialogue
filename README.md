# Cortical representations of languages during natural dialogue
Yamashita M., Kubo R., and Nishimoto. S. 2023, [bioRxiv](https://doi.org/10.1101/2023.08.21.553821)

- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Data Description](#data-description)
- [Full Analysis](#full-analysis)
- [Visualization](#visualization)

# System Requirements

## Hardware requirements

- GPU for running himalaya

## Software requirements

Python 3 (for ridge regression anlysis and makeing plots)
- [NumPy](https://numpy.org/) (confirmed on ver. 1.26.4)
- [SciPy](https://scipy.org/) (confirmed on ver. 1.12.0)
- [Scikit-learn](https://scikit-learn.org/) (confirmed on ver. 1.4.0)
- [Himalaya](https://github.com/gallantlab/himalaya) (confirmed on ver. 0.4.2)
- [Matplotlib](https://matplotlib.org/) (confirmed on ver. 3.8.2)
- [seaborn](https://seaborn.pydata.org/) (confirmed on ver. 0.13.2)
- [Pycortex](https://github.com/gallantlab/pycortex) (confirmed on ver. 1.2)

FreeSurfer (for Pycortex; https://surfer.nmr.mgh.harvard.edu/) (confirmed on ver. 6.0)

R (for the linear mixed-effects model analysis)
- [lmerTest](https://cran.r-project.org/web/packages/lmerTest/) (confirmed on ver. 3.1-3)

# Installation Guide

```
git clone https://github.com/yamashita-lang/dialogue
```
Edit config.py file to localize your environment (e.g., /home/projects/open/dialogue/).

# Data Description

We share data on [OpenNeuro](https://openneuro.org/datasets/ds004669). Store derivative data  in derivative subdirectory  (e.g., /home/projects/open/dialogue/derivative/).
- Preprocessed fMRI (sub-OSU01_task-dialogue_bold.hdf)
- Head motion (motion parameters and framewise displacement; sub-OSU01_headmotion.hdf)
- Random embeddings (sub-OSU01_gpt_nprandom.hdf)
- Stimulus GPT embeddings (sub-OSU01_chatgptneox_cl-1.hdf)
- Individual fMRI target space (target.nii)
- FreeSurfer ROI information (vset_info.mat)
- Pycortex database (overlays.svg)

# Full Analysis

## 1. Prediction of BOLD responses
```
python modeling_wrapper.py 7
```
Perform banded ridge regression for each condition.  
- Expected inputs:
	- sub-OSU07_task-dialogue_bold.hdf
	- sub-OSU07_headmotion.hdf
	- sub-OSU07_gpt_nprandom.hdf
	- sub-OSU07_chatgptneox_cl-1.hdf
- Expected outputs:
	- sub-OSU07_headmotion_ses-1_intramodal.hdf
	- sub-OSU07_ses-1_task-dialogue_bold_res_headmotion.hdf
	- sub-OSU07_headmotion_gpt_nprandom_ses-1_intramodal.hdf
	- sub-OSU07_ses-1_task-dialogue_bold_res_gpt_nprandom.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_ses-1_layer-0_cl-1_intramodal.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_ses-1_layer-0_cl-1_crossmodal.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_prod_ses-1_layer-0_cl-1_intramodal.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_comp_ses-1_layer-0_cl-1_intramodal.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_unified_1_comp_ses-1_layer-0_cl-1_intramodal.hdf
- Expected run time: several tens of minutes for each condition

## 2. Permutation tests
```
python perm_wrapper.py 7
```
Perform permuration tests for each session.
- Expected outputs: 
	- sub-OSU07_headmotion_ses-1_intramodal_perm_1000.hdf
	- sub-OSU07_headmotion_gpt_nprandom_ses-1_layer-0_cl-1_intramodal_perm_1000.hdf
	- sub-OSU07_headmotion_gpt_nprandom_ses-1_layer-0_cl-1_intramodal_perm_1000.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_ses-1_layer-0_cl-1_intramodal_perm_1000.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_ses-1_layer-0_cl-1_crossmodal_perm_1000.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_prod_ses-1_layer-0_cl-1_intramodal_perm_1000.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_comp_ses-1_layer-0_cl-1_intramodal_perm_1000.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_unified_1_ses-1_layer-0_cl-1_intramodal_perm_1000.hdf
- Expected run time: several hours for each condition
	
## 3. Evalutation
```
python evaluating_wrapper.py 7
```
Average FDR corrected results across sessions. Variance partitioning.
- Expected outputs:
	- sub-OSU07_headmotion_intramodal.hdf
	- sub-OSU07_headmotion_gpt_nprandom_intramodal.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_layer-0_cl-1_intramodal.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_layer-0_cl-1_crossmodal.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_prod_layer-0_cl-1_intramodal.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_comp_layer-0_cl-1_intramodal.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_unified_1_layer-0_cl-1_intramodal.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_layer-0_cl-1_vp.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_1_layer-0_cl-1_best_vp.hdf

- Expected run time: 
	- FDR correction: several tens of minutes
	- Variance partitioning: several hours

```
python weightcorr_wrapper.py 7
```
Weight correlation for Separate Linguistic model and Unified Linguistic model.
- Expected outputs:
	- sub-OSU07_gpt_nprandom_chatgpt_1_layer-0_cl-1_weight_corr_tvoxels.hdf
	- sub-OSU07_gpt_nprandom_chatgpt_unified_1_layer-0_cl-1_weight_corr_tvoxels.hdf

- Expected run time: several hours for each condition

## 4. Principal component analysis
```
python pca_stimulus_wrapper.py
```
Princiapl component analysis for GPT embeddings.
- Expected outputs:
	- sub-OSU07_stim_chatgptneox_pca_prod_layer-0_cl-1.hdf
	- sub-OSU07_stim_chatgptneox_pca_comp_layer-0_cl-1.hdf
	- sub-OSU07_stim_chatgptneox_pcloads_prod_layer-0_cl-1_high_PC1.txt
	- sub-OSU07_stim_chatgptneox_pcloads_comp_layer-0_cl-1_high_PC1.txt
	
- Expected run time: several tens of minutes for each condition

```
python pca_weights_wrapper.py
```
Princiapl component analysis for GPT embeddings.
- Expected outputs:
	- sub-OSU07_pca_gpt_nprandom_chatgpt_1_layer-0_cl-1_prod.hdf
	- sub-OSU07_pca_gpt_nprandom_chatgpt_1_layer-0_cl-1_comp.hdf
	- sub-OSU07_pcloads_gpt_nprandom_chatgpt_1_layer-0_cl-1_prod.hdf
	- sub-OSU07_pcloads_gpt_nprandom_chatgpt_1_layer-0_cl-1_comp.hdf
	- sub-OSU07_pcloads_gpt_nprandom_chatgpt_1_layer-0_cl-1_prod_high_PC1.txt
	- sub-OSU07_pcloads_gpt_nprandom_chatgpt_1_layer-0_cl-1_prod_low_PC1.txt
	- sub-OSU07_pcloads_gpt_nprandom_chatgpt_1_layer-0_cl-1_comp_high_PC1.txt
	- sub-OSU07_pcloads_gpt_nprandom_chatgpt_1_layer-0_cl-1_comp_low_PC1.txt
	- sub-OSU07_pca_bootstrap_gpt_nprandom_chatgpt_1_layer-0_cl-1_prod.hdf
	- sub-OSU07_pca_bootstrap_gpt_nprandom_chatgpt_1_layer-0_cl-1_comp.hdf
	
- Expected run time: several hours for each condition

## 5. Summary and make figures
```
python summary_wrapper.py
```

## 6. Linear mixed-effects analysis
lme_analysis.R


# Visualization

## Cortical maps by Pycortex

example_pycortex.ipynb
