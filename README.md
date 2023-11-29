# Cortical representations of languages during natural dialogue
Yamashita M., Kubo R., and Nishimoto. S. 2023, [bioRxiv](https://doi.org/10.1101/2023.08.21.553821)

- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Data Description](#data-description)
- [Demo](#demo)
- [Full Analysis](#full-analysis)
- [Visualization](#visualization)

# System Requirements

## Hardware requirements

- 512GB memory system

## Software requirements

- MATLAB (confirmed on ver. R2019b, on Ubuntu 18.04.4 LTS.)
- MATLAB Statstics Toolbox
- MATLAB function for 'pca_bootstrap.m' ([Gale-Shapley stable marriage algorithm](https://jp.mathworks.com/matlabcentral/fileexchange/44262-gale-shapley-stable-marriage-algorithm)) 
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) (confirmed on ver. 6.0)
- [PyCortex](https://github.com/gallantlab/pycortex) (confirmed on ver. 1.2) 

# Installation Guide

```
git clone https://github.com/yamashita-lang/dialogue
```
Edit 'ProjectDir' in 'load_parameters_proj.m' file to your environment (e.g., /home/projects/open/dialogue/).

# Data Description

We share data on [OpenNeuro](https://openneuro.org/datasets/ds004669). 
- Derivative data, should be located in derivative subdirectory  (e.g., /home/projects/open/dialogue/derivative/).
1. Preprocessed fMRI
	- e.g., '/derivative/preprocessed_data/sub-OSU01/RespData_sub-OSU01_ses-1_task-dialogue_bold.mat'
2. Normalized stimulus features
	- e.g., '/derivative/preprocessed_data/sub-OSU01/StimData_CHATGPTNEOX_sub-OSU01_ses-1_task-dialogue.mat'
3. Individual fMRI target space
	- e.g., '/derivative/preprocessed_data/sub-OSU01/target_sub-OSU01.nii'
4. FreeSurfer ROI information
	- e.g., '/derivative/fsROI/sub-OSU01/vset_info.mat'
5. Stimulus GPT embeddings
	- e.g., '/derivative/preprocessed_data/sub-OSU01/Embedding_CHATGPTNEOX_Prod_sub-OSU01.mat'
6. Morpheme and syllable counts for each utterance
	- e.g., '/derivative/preprocessed_data/sub-OSU01/Lowlevel_Prod_sub-OSU01.mat'
7. PyCortex database
	- e.g., '/derivative/pycortex_db/sub-OSU01/overlays.svg'
- Transcribed utterances, which include sensitive information and will be shared upon reasonable request.

# Demo

This demo reproduces key findings. Please refer to Full Analysis section below for a more detailed analysis.
- Sample results for demo, should be located in results subdirectory (e.g, /home/projects/open/dialogue/derivative/results).

## Setting

```
ID='sub-OSU08';
mode=1; % for speech production, mode=2 for speech comprehension
```	

## Individual-1. Prediction of BOLD responses (Figure 1 and 2)

```
ridge_cerebcortex(ID)
```
Perform ridge regression for same-/cross-modality prediction for each session.  
- Expected inputs:
	- Preprocessed fMRI
	- Normalized stimulus features
- Expected outputs: RidgeResults_CHATGPTNEOX_sub-OSU08_ses-[1, ..., n].mat
- Expected run time: several hours for each session

## Individual-2. Unique variance partitioning

### Reduced model (Figure 3)

```
ridge_cerebcortex_uniqvarpart_reduced(ID)
```
Perform unique variance partitioning and calculate relative complement for each set of feature
- Expected inputs:
	- RidgeResults_CHATGPTNEOX_sub-OSU08_ses-[1, ..., n].mat
	- RidgeResults_CHATGPTNEOX_sub-OSU08_ses-[1, ..., n]_MainFeatures.mat
	- RidgeResults_CHATGPTNEOX_sub-OSU08_MainFeatures_FDRcorr_mean.mat
	- RidgeResults_CHATGPTNEOX_sub-OSU08_ses-[1, ..., n]_SingleFeature_1.mat
	- RidgeResults_CHATGPTNEOX_sub-OSU08_ses-[1, ..., n]_SingleFeature_7.mat
- Expected outputs:
	- RidgeResults_CHATGPTNEOX_sub-OSU08_SingleFeature_Intersection.nii
	- RidgeResults_CHATGPTNEOX_sub-OSU08_SingleFeature_Prod_RC.nii
	- RidgeResults_CHATGPTNEOX_sub-OSU08_SingleFeature_Comp_RC.nii
	- RidgeResults_CHATGPTNEOX_sub-OSU08_SingleFeature_VarPart.mat
- Expected run time: several tens of minutes
	
## Individual-3. Weight correlation (Figure 4)

```
weight_corr(ID)
```
Calculate semantic weight correlation between production and comprehension for each voxel
- Expected inputs:
	- RidgeResults_CHATGPTNEOX_sub-OSU08_ses-[1, ..., n].mat
	- RidgeResults_CHATGPTNEOX_sub-OSU08_FDRcorr_mean.mat
- Expected outputs:
	- WeightCorr_CHATGPTNEOX_sub-OSU08_FDRcorr.nii
	- WeightCorr_CHATGPTNEOX_sub-OSU08_FDRcorr.mat
- Expected run time: several tens of minutes

## Group PCA (Figure 5)

```
pca_weights(mode)
```
Perform principal component analysis for semantic weights combined across participants
- Expected inputs:
	- RidgeResults_CHATGPTNEOX_sub-OSU08_ses-[1, ..., n].mat
	- RidgeResults_CHATGPTNEOX_sub-OSU08_FDRcorr_mean.mat
-Expected outputs:
	- RidgeResults_CHATGPTNEOX_sub-OSU08_meanweight.mat
	- Weight_CHATGPTNEOX_sub-OSU08_Prod.mat
	- GroupWeight_CHATGPTNEOX_Prod.mat
	- PCA_Results_CHATGPTNEOX_Prod.mat
	- PCA_Result_CHATGPTNEOX_sub-OSU08_Prod_RGBmap_R.nii
- Expected run time: several hours

# Full Analysis

## Setting

```
ID='sub-OSU08'; % or 'sub-OSU01', …, 'sub-OSU08'
mode=1; % for speech production, mode=2 for speech comprehension
```

## Individual-1. Prediction of BOLD responses (Figure 1 and 2)

```
ridge_cerebcortex(ID)
```
See Demo section
	
```
ridge_fdrcorr(ID, mode) 
```
Perform FDR correction for regression results for each session.
- Expected outputs: RidgeResults_CHATGPTNEOX_sub-OSU01_ses-[1, ..., n].mat
- Expected run time: several tens of minutes for each session
	
```
ridge_session_average(ID) 
```
Average FDR corrected results across sessions.
- Expected outputs:
	- RidgeResults_CHATGPTNEOX_sub-OSU01_same_FDRcorr_mean.nii
	- RidgeResults_CHATGPTNEOX_sub-OSU01_cross_FDRcorr_mean.nii
	- RidgeResults_CHATGPTNEOX_sub-OSU01_FDRcorr_mean.mat
- Expected run time: several tens of minutes
	
## Individual-2. Unique variance partitioning

### Full model  (Extended Data Figure 3)

```
for rd_idx = 1:11
	ridge_cerebcortex_submodel(ID, rd_idx);
end
```
Perform ridge regression with removing a set of feature
- Expected outputs: RidgeResults_CHATGPTNEOX_sub-OSU01_ses-1_234567891011.mat
- Expected run time: several hours for each session, for each feature
	
```
ridge_cerebcortex_uniqvarpart(ID)
```
Perform unique variance partitioning and calculate relative complement for each set of feature
- Expected outputs:
	- 'RidgeResults_CHATGPTNEOX_sub-OSU01_UniqVP_POS_Prod.nii'
	- 'RidgeResults_CHATGPTNEOX_sub-OSU01_UniqVP_POS_Prod.mat'
- Expected run time: several tens of minutes for each feature
	
### Reduced model (Figure 3)

```
ridge_cerebcortex_mainfeatures(ID)
```
Perform ridge regression with semantic features
- Expected outputs: RidgeResults_CHATGPTNEOX_sub-OSU01_ses-1_MainFeatures.mat
- Expected run time: several hours for each session
	
```
ridge_cerebcortex_singlefeature(ID, mode)
```
Perform ridge regression with single-modality semantic features
- Expected outputs: RidgeResults_CHATGPTNEOX_sub-OSU01_ses-1_SingleFeature_1.mat
- Expected run time: several hours for each session
	
 ```
 ridge_fdrcorr_mainfeatures(ID)
 ```
Perform FDR correction for regression results for each session.
- Expected outputs: RidgeResults_CHATGPTNEOX_sub-OSU01_ses-1_MainFeatures.mat
- Expected run time: several tens of minutes for each session
 	
```
ridge_session_average_mainfeatures(ID)
```
Average FDR corrected results across sessions.
- Expected outputs:
	- RidgeResults_CHATGPTNEOX_subOSU01_MainFeatures_FDRcorr_mean.nii
  	- RidgeResults_CHATGPTNEOX_subOSU01_MainFeatures_FDRcorr_mean.mat
- Expected run time: several tens of minutes
  	
```
ridge_cerebcortex_uniqvarpart_reduced(ID)
```
See Demo section
	
```
best_uniqvarpart_rgb(ID)
```
Examaine best variance partitioning for each voxel
- Expected outputs:
	- RidgeResults_CHATGPTNEOX_sub-OSU01_VarPart_ccs.mat
	- RidgeResults_CHATGPTNEOX_sub-OSU01_VarPart_RGB.nii
- Expected run time: several tens of minutes
	
## Individual-3. Weight correlation (Figure 4)

```
weight_corr(ID)
```
See Demo section
	
## Group PCA (Figure 5)

```
pca_weights(mode)
```
See Demo section
 
```
pca_bootstrap(mode, 1)
```
Perform bootstrap resampling to test significance
- Expected outputs:
	- PCA_bootstrap_CHATGPTNEOX_Prod.mat
	- PCA_bootstrap_CHATGPTNEOX_Prod_pval.csv
- Expected run time: several hours

```
pca_bootstrap(mode, 2)
```
- Expected outputs: PCA_bootstrap_CHATGPTNEOX_Prod.csv
- Expected run time: several minutes

```
sig_pc=4; for speech production % sig_pc=6; for speech comprehension
pca_stats(mode, sig_pc, 1)
```
Analyse PCA results 
- Expected outputs:
	- Uniq_embedding_CHATGPTNEOX_Prod_sub-OSU01.mat
	- GroupEmb_Prod_uniq.mat
	- PCA_Result_CHATGPTNEOX_Prod_high_PC[1, ..., n].txt
	- PCA_Result_CHATGPTNEOX_Prod_low_PC[1, ..., n].txt
- Expected run time: several minutes

```
pca_stats(mode, sig_pc, 2)
```
- Expected outputs: PCA_Result_CHATGPTNEOX_Prod_PC_vs_LowLevelFeatures.csv
- Expected run time: several minutes

```
pca_stats(mode, sig_pc, 3)
```
- Expected outputs:
	- Corr_PCA_vs_POS_CHATGPTNEOX_Prod_sub-OSU01.mat
	- GroupMean_Corr_PCA_vs_POS_CHATGPTNEOX_Prod.mat
- Expected run time: several minutes

```
pca_stats(mode, sig_pc, 4)
```
- Expected output:
	- PCload_CHATGPTNEOX_Prod.mat
	- PCload_CHATGPTNEOX_Prod.txt
	- PCload_CHATGPTNEOX_Prod_RGB_[R,G,B].txt
- Expected run time: several minutes

# Visualization

## Cortical maps by PyCortex

pycortex_example.ipynb

## Results of Figure 5

make_fig5.ipynb 
