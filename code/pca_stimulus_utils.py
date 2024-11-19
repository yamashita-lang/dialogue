import copy
import numpy as np
import os
import tables
from scipy.stats import zscore, pearsonr
from sklearn.decomposition import PCA
from dialogue.io import get_ready_indiv_pcafile, get_features_ses, save_table_file, save_strings_hdf_file, load_strings_hdf_array
from dialogue.config import TOTAL_SESS, SUBJECTS_ALL, FEATURES_DIR, ANNT_DIR, PCA_DIR, RESULTS_DIR, N_PC, N_TOP_VOXELS, LAYERS, CONTEXT_LENGTH, N_PERM
from dialogue.modeling.pca_weights_utils import get_indiv_concatenated_stimlus



#######################################################################################################################################################################################
#######################################################################################################################################################################################
### pca_stim_wrapper.py
#######################################################################################################################################################################################
#######################################################################################################################################################################################
def indiv_stim_pca(subject_idx, modality_idx=0, layer=-1, cl=1, llm='chatgptneox'):
    if modality_idx == 0:
        modality = 'prod'
    elif modality_idx == 1:
        modality = 'comp'

    postfix = f"_layer-{layer}_cl-{cl}"
    stim_pca_file = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_stim_{llm}_pca_{SUBJECTS_ALL[subject_idx]}_{modality}{postfix}.hdf"
    
    ##### 1 Make/load stimuli concatenated across subjects
    indiv_utt, indiv_gpt, unique_utt_idx = get_indiv_concatenated_stimlus(subject_idx, modality_idx=modality_idx, layer=layer, cl=cl, llm=llm)
    print(f"indiv_gpt: {indiv_gpt.shape}")
    assert check_nan_inf(indiv_gpt), "indiv_gpt contains NaN or Inf"
    ##### 2 Stimulus PCA
    pca = PCA(n_components=N_PC)
    pca.fit(indiv_gpt.T)
    pc_var_explained = pca.explained_variance_ratio_
    pc_coefs = pca.components_
    pc_scores = pca.fit_transform(indiv_gpt.T).T
    print("PC coef: (n_components, n_features) =", pc_coefs.shape)
    print("PC scores: (n_components, n_voxels) =", pc_scores.shape)
    print("PCA explained variance =", pca.explained_variance_ratio_)
    print(f"saving stim_pca_file: {stim_pca_file}")
    save_table_file(stim_pca_file, dict(pc_coefs=pc_coefs, pc_scores_all=pc_scores, pc_var_explained=pc_var_explained))


def get_indiv_stim_pcloads_txt(subject_idx, modality_idx=0, layer=-1, cl=1, llm='chatgptneox'):
    indiv_utt, indiv_gpt, unique_utt_idx = get_indiv_concatenated_stimlus(subject_idx, modality_idx=modality_idx, layer=layer, cl=cl, llm=llm)
    if modality_idx == 0:
        modality = 'prod'
    elif modality_idx == 1:
        modality = 'comp'
    
    postfix=f"_layer-{layer}_cl-{cl}"
    stim_pca_file = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_stim_{llm}_pca_{modality}{postfix}.hdf"
  
    with tables.open_file(stim_pca_file, "r") as PCA_f:
        pc_coefs = PCA_f.root.pc_coefs.read()
        ### PC loadings by correlating PC coeff with embeddings
        print(f"pc_coefs: {pc_coefs.shape}")
        print(f"indiv_gpt: {indiv_gpt.shape}")
        n_target_pc = 10

        ### Extract most correlative utterances for each PC
        n_extract = 100
        for pc_i in range(n_target_pc):
            print(f"****** PC {pc_i+1}:")
            d_idx = np.argsort(-pc_coefs[pc_i, :])
            sorted_pc_coefs_high = np.sort(-pc_coefs[pc_i,:])
            sorted_indiv_utt_high = indiv_utt[d_idx]
            print(f"sorted_pc_coefs_high: {sorted_pc_coefs_high.shape}")
            print(f"sorted_pc_coefs_high: {-sorted_pc_coefs_high}")
            print(f"sorted_indiv_utt_high: {sorted_indiv_utt_high.shape}")
            print(f"sorted_indiv_utt_high: {sorted_indiv_utt_high}")
            print(f"indiv_utt on top: {indiv_utt[d_idx[0]]}")
            txtfile = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_stim_{llm}_pcloads_{modality}{postfix}_high_PC{pc_i+1}.txt"

            for jj in range(n_extract):
                with open(txtfile, "a") as f:
                    f.write(f"{jj+1}\t{-sorted_pc_coefs_high[jj]}\t{sorted_indiv_utt_high[jj]}\n")

            a_idx = np.argsort(pc_coefs[pc_i, :])
            sorted_pc_coefs_low = np.sort(pc_coefs[pc_i,:])
            sorted_indiv_utt_low = indiv_utt[a_idx]
            print(f"sorted_pc_coefs_low: {sorted_pc_coefs_low.shape}")
            print(f"sorted_pc_coefs_low: {sorted_pc_coefs_low}")
            print(f"sorted_indiv_utt_low: {sorted_indiv_utt_low.shape}")
            print(f"sorted_indiv_utt_low: {sorted_indiv_utt_low}")
            print(f"indiv_utt on top: {indiv_utt[d_idx[0]]}")
            txtfile = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_stim_{llm}_pcloads{modality}{postfix}_low_PC{pc_i+1}.txt"
            
            for jj in range(n_extract):
                with open(txtfile, "a") as f:
                    f.write(f"{jj+1}\t{sorted_pc_coefs_low[jj]}\t{sorted_indiv_utt_low[jj]}\n")


