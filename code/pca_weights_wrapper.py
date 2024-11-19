import os
import argparse
import tables
import numpy as np
from dialogue.io import get_ready_indiv_pcafile
from dialogue.features.dsutils import save_table_file
from dialogue.config import SUBJECTS_ALL, TOTAL_SESS, RESULTS_DIR, BOLD_DIR, FIG_DIR, PCA_DIR, LAYERS, CONTEXT_LENGTH, N_PC, N_TOP_VOXELS, N_PERM
from dialogue.visualization.vis_utils import plot_pca_expvar, plot_heatmap_npc
from dialogue.modeling.pca_weights_utils import (
        indiv_pca, get_indiv_concatenated_stimlus, get_indiv_pc_loadings, get_indiv_pcloads_txt, run_indiv_pca_bootstrap,
        summary_indiv_pca_expvar, summary_indiv_pc_scores, summary_indiv_pca_bootstrap, summary_indiv_pca_bootstrap_each
)

parser=argparse.ArgumentParser(description='Run pca for each subject')
parser.add_argument('subject_idx', type=int, help='subject_idx (e.g., 1 for sub-OSU01)')
args=parser.parse_args()
subject_idx=args.subject_idx-1

##
random_model = "gpt_nprandom"
model_prefix = "chatgpt"
subject = SUBJECTS_ALL[subject_idx]
alpha=0.001


##############################################################################################################################################################################
##############################################################################################################################################################################
### individual weight PCA
##############################################################################################################################################################################
##############################################################################################################################################################################
for cl_idx, cl in enumerate(CONTEXT_LENGTH):
    for layer_idx, layer in enumerate(LAYERS):
        model=f'{model_prefix}_{cl}'
        for modality_idx in [0, 1]:
            if modality_idx == 0:
                modality = 'prod'
            elif modality_idx == 1:
                modality = 'comp'

            ##############################################################################################################################################################
            ### indiv PCA
            indiv_pca(subject_idx, model, random_model, modality_idx=modality_idx, layer=layer, cl=cl, n_top_voxels=None)

            ##############################################################################################################################################################
            ### get PCA loadings
            get_indiv_concatenated_stimlus(subject_idx, modality_idx=modality_idx, layer=layer, cl=cl)
            get_indiv_pc_loadings(subject_idx, model, random_model, modality_idx=modality_idx, layer=layer, cl=cl, n_top_voxels=None)
            get_indiv_pcloads_txt(subject_idx, model, random_model, modality_idx=modality_idx, layer=layer, cl=cl, n_top_voxels=None)


##############################################################################################################################################################################
##############################################################################################################################################################################
### bootstrap
##############################################################################################################################################################################
##############################################################################################################################################################################
for cl_idx, cl in enumerate(CONTEXT_LENGTH):
    for layer_idx, layer in enumerate(LAYERS):
        model=f'{model_prefix}_{cl}'
        _, _, _, _, boot_prod_file, boot_comp_file, _, _ = get_ready_indiv_pcafile(subject, model, random_model, layer=layer, cl=cl, n_top_voxels=None)
        for modality_idx in [0, 1]:
            if modality_idx == 0:
                modality = 'prod'
                boot_file = boot_prod_file
            elif modality_idx == 1:
                modality = 'comp'
                boot_file = boot_comp_file

            run_indiv_pca_bootstrap(subject_idx, model, random_model, modality_idx=modality_idx, layer=layer, cl=cl, n_top_voxels=None)
      

##############################################################################################################################################################################
##############################################################################################################################################################################
### summary_pca_results
##############################################################################################################################################################################
##############################################################################################################################################################################
random_model = "gpt_nprandom"
model_prefix='chatgpt'
summary_indiv_pca_expvar(random_model, model_prefix)
summary_indiv_pc_scores(random_model, model_prefix)
summary_indiv_pca_bootstrap(random_model, model_prefix)

for subject_idx in range(len(SUBJECTS_ALL)):
    summary_indiv_pca_bootstrap_each(subject_idx, random_model, model_prefix)

##############################################################################################################################################################################
##############################################################################################################################################################################
### visualization
##############################################################################################################################################################################
##############################################################################################################################################################################
model = f'{model_prefix}_32'

for modality_idx in [0, 1]:
    if modality_idx == 0:
        modality = 'prod'
    elif modality_idx == 1:
        modality = 'comp'

    ################################################################################################################################################
    ### % variance explained
    for cl_idx, cl in enumerate(CONTEXT_LENGTH):
        tmp_model = f"{model_prefix}_{cl}"
        for layer_idx, layer in enumerate(LAYERS):
            postfix = f"layer-{layer}_cl-{cl}"

            summary_file = f"{RESULTS_DIR}{subject}_indiv_pca_bootstrap_{random_model}_{model}_{modality}_alpha{alpha}.hdf"
            with tables.open_file(summary_file, "r") as SUMMARY_f:
                stim_var_explained = PCA_f.root.explained_stims.read()
                weight_var_explained = PCA_f.root.explained_weights.read()

            tmp_weight_var_explained = weight_var_explained[layer_idx, cl_idx, :, :10]
            tmp_stim_var_explained = stim_var_explained[layer_idx, cl_idx, :, :10]
            
            ### Fig. 5a, e
            lineplot_file = f"{FIG_DIR}{subject}_lineplot_variance_explained_{random_model}_{tmp_model}_{postfix}_{modality}_{errorbar}.pdf"
            plot_pca_expvar(tmp_weight_var_explained, tmp_stim_var_explained, lineplot_file, plot_idx=0, modality_idx=modality_idx)

    ################################################################################################################################################
    ### Number of sign. principal components
    summary_file = f"{RESULTS_DIR}{subject}_indiv_pca_bootstrap_{random_model}_{model}_{modality}_alpha{alpha}.hdf"
    with tables.open_file(summary_file, "r") as BOOT_f:
        n_pcs = BOOT_f.root.n_pcs.read()
        
    ### Supplementary Fig. 18　- individual results
    heatmap_file = f"{FIG_DIR}{subject}_bootstrap_heatmap_{random_model}_{model}_{modality}_alpha{alpha}.pdf"
    plot_heatmap_npc(n_pcs, heatmap_file, modality_idx=modality_idx)


    ### Fig. 5b, f - mean across subjects
    group_npcs = np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL)), np.nan)
    for subj_idx in range(len(SUBJECTS_ALL)):
        tmp_data_file = f"{RESULTS_DIR}{subject}_indiv_pca_bootstrap_{random_model}_{model}_{modality}_alpha{alpha}.hdf"
        with tables.open_file(tmp_data_file, "r") as BOOT_f:
            group_npcs[:,:,subj_idx] = BOOT_f.root.n_pcs.read()
        
    mean_npcs = np.nanmean(group_npcs, axis=2)
    heatmap_file = heatmap_file = f"{FIG_DIR}bootstrap_heatmap_{random_model}_{model}_{modality}_alpha{alpha}_group_mean.pdf"
    plot_heatmap_npc(mean_npcs, heatmap_file, modality_idx=modality_idx)


