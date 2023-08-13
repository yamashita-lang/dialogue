# Cortical Representations of Languages during Natural Dialogue
Yamashita M., Kubo R., and Nishimoto. S. 2023, bioRxiv

![Figure1](https://github.com/yamashita-lang/dialogue/assets/141693211/7854d5eb-121b-4964-a85e-63830f2142fa)

## Data
Available at [OpenNeuro](https://openneuro.org/datasets/ds004669)
1. Preprocessed fMRI (e.g., '/derivative/preprocessed_data/sub-OSU01/RespData_sub-OSU01_ses-1_task-dialogue_bold.mat')
2. Normalized stimulus features (e.g., '/derivative/preprocessed_data/sub-OSU01/StimData_CHATGPTNEOX_sub-OSU01_ses-1_task-dialogue.mat')
3. Individual fMRI target space (e.g., '/derivative/preprocessed_data/sub-OSU01/target_sub-OSU01.nii')
4. FreeSurfer ROI information (e.g., '/derivative/fsROI/sub-OSU01/vset_info.mat')
5. PyCortex database (e.g., '/derivative/pycortex_db/sub-OSU01/overlays.svg')
6. Stimulus GPT embeddings (e.g., '/derivative/preprocessed_data/sub-OSU01/Embedding_CHATGPTNEOX_Prod_sub-OSU01.mat')
7. Morpheme and syllable counts for each utterance (e.g., '/derivative/preprocessed_data/sub-OSU01/Lowlevel_Prod_sub-OSU01.mat')
8. (Transcribed utterances, which include sensitive information and thus will be shared upon reasonable request)

## Requirements
1. MATLAB (We confirmed on MATLAB ver. R2019b, on Ubuntu 18.04.4 LTS.)
2. MATLAB function ([Gale-Shapley stable marriage algorithm](https://jp.mathworks.com/matlabcentral/fileexchange/44262-gale-shapley-stable-marriage-algorithm))
3. [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)
5. [PyCortex](https://github.com/gallantlab/pycortex)

## MATALB Usage

### Setting
	ID='sub-OSU01'; % or 'sub-OSU02', …, 'sub-OSU08'
 	mode=1; % for production
  	% mode=2; % for comprehension

### Analysis pipeline
	# Individual-level
	for do ss=1:8
 		do_individual_analysis(ss)
	end
   
	# Group-level
 	do_group_pca

### Individual-1. Prediction of BOLD responses (Figure 1 and 2)
	ridge_cerebcortex(ID)
	ridge_fdrcorr(ID, mode)
	ridge_session_average(ID)

### Individual-2. Unique variance partitioning

#### Full model  (Extended Data Figure 3)
	for rd_idx = 0:11
		ridge_cerebcortex_submodel(ID, rd_idx);
	end
	ridge_cerebcortex_uniqvarpart(ID)

#### Reduced model (Figure 3)
	ridge_cerebcortex_mainfeatures(ID)
	ridge_cerebcortex_singlefeature(ID, mode)
 	ridge_fdrcorr_mainfeatures(ID)
  	ridge_session_average_mainfeatures(ID)
	ridge_cerebcortex_uniqvarpart_reduced(ID)
	best_uniqvarpart_rgb(ID)

### Individual-3. Weight correlation (Figure 4)
	weight_corr(ID)

### Group PCA (Figure 5)
	pca_weights(mode)
 
	pca_bootstrap(mode, 1)
	pca_bootstrap(mode, 2)
 	
  	sig_pc=4; for production
  	% sig_pc=6; for comprehension
	
 	pca_stats(mode, sig_pc, 1)
  	pca_stats(mode, sig_pc, 2)
   	pca_stats(mode, sig_pc, 3)
	pca_stats(mode, sig_pc, 4)


## Visualization
### Cortical maps by PyCortex
	pycortex_example.ipynb

### Results of Figure 5
	make_fig5.ipynb 
