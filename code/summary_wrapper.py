from dialogue.github.modeling.config import SUBJECTS_ALL
from dialogue.github.modeling.summary_utils import (
        utterance_stats, summary_headmotion_model, summary_random_intramodal, summary_random_crossmodal, 
        summary_intramodal_context, summary_crossmodal_context, summary_intramodal_unified, find_max_context_layers,
        summary_weightcorrs_contexts_tvoxels, summary_weightcorrs_contexts_tvoxels_unified, summary_data_for_lmer,
        summary_variance_partitioning, summary_best_vp
)

from dialogue.github.modeling.barplot_utils import (
        barplot_uttered_scans, 
        plot_scores_all_group_layers_contexts, plot_scores_all_actual_vs_cross_vs_unified, 
        plot_vp_context_X_partitions, plot_vp_group_layers_contexts,
        plot_weightcorr_context_X_layers_vp_RGB, plot_weightcorr_context_X_layers_intra_cross_svoxels, plot_weightcorr_context_X_layers_unified,
)


model_base="chatgpt"
model = f"{model_base}_32"

unified_model_base = "chatgpt_unified"
unified_model =  f"{unified_model_base}_32"

##########################################################################################################################################################################
##########################################################################################################################################################################
### statistics
##########################################################################################################################################################################
##########################################################################################################################################################################  
### Supplementary Fig. 1: fMRI data samples * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
utterance_stats()
barplot_uttered_scans()


##########################################################################################################################################################################
##########################################################################################################################################################################
### Head Motion model
##########################################################################################################################################################################
##########################################################################################################################################################################
summary_headmotion_model(sig_idx=0)
summary_headmotion_model(sig_idx=1)


##########################################################################################################################################################################
##########################################################################################################################################################################
### Random embedding model
##########################################################################################################################################################################
##########################################################################################################################################################################
summary_random_intramodal(sig_idx=0)
summary_random_intramodal(sig_idx=1)
summary_random_crossmodal(sig_idx=0)
summary_random_crossmodal(sig_idx=1)


##########################################################################################################################################################################
##########################################################################################################################################################################
### Separate and Unified Linguistic model
##########################################################################################################################################################################
##########################################################################################################################################################################
### Fig. 2: Separate and Unified Linguistic model * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
basemodel = "gpt_nprandom"
summary_intramodal_context(model, basemodel, model_base, sig_idx=0)
summary_intramodal_context(model, basemodel, model_base, sig_idx=1)

summary_crossmodal_context(model, basemodel, model_base, sig_idx=0, svoxel_idx=0)
summary_crossmodal_context(model, basemodel, model_base, sig_idx=1, svoxel_idx=0)

summary_crossmodal_context(model, basemodel, model_base, sig_idx=0, svoxel_idx=1)
summary_crossmodal_context(model, basemodel, model_base, sig_idx=1, svoxel_idx=1)

summary_intramodal_unified(basemodel, unified_model, unified_model_base, sig_idx=0)
summary_intramodal_unified(basemodel, unified_model, unified_model_base, sig_idx=1)


### Supplementary Fig. 8 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
### Separate Linguistic model (Intramodal)
plot_scores_all_group_layers_contexts(model, basemodel, unified_model, plot_idx=0, test_idx=0, ymin=-0.06, ymax=0.1, sig_idx=0)
plot_scores_all_group_layers_contexts(model, basemodel, unified_model, plot_idx=1, test_idx=0, ymin=-0.06, ymax=0.1, sig_idx=0)

### Separate Linguistic model (Cross-modality)
plot_scores_all_group_layers_contexts(model, basemodel, unified_model, plot_idx=0, test_idx=1, ymin=-0.06, ymax=0.1)
plot_scores_all_group_layers_contexts(model, basemodel, unified_model, plot_idx=1, test_idx=1, ymin=-0.06, ymax=0.1)

### Unified model
plot_scores_all_group_layers_contexts(model, basemodel, unified_model, plot_idx=0, test_idx=2, ymin=-0.06, ymax=0.1)
plot_scores_all_group_layers_contexts(model, basemodel, unified_model, plot_idx=1, test_idx=2, ymin=-0.06, ymax=0.1)
    

### Fig. 2b * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
plot_scores_all_actual_vs_cross_vs_unified(model, basemodel, unified_model, plot_idx=0, ymin=-0.05, ymax=0.08)


##########################################################################################################################################################################
##########################################################################################################################################################################
### Variance partitioning
##########################################################################################################################################################################
##########################################################################################################################################################################
basemodel = "gpt_nprandom"
### Fig. 3 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
summary_variance_partitioning(model, basemodel, model_base)
summary_best_vp(model, basemodel, model_base)

### Fig. 3a, 4a * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
plot_vp_context_X_partitions(model, basemodel, unified_model, plot_idx=0, ymin=0, ymax=0.05, sig_idx=1, z_idx=1)

### Fig. 3b, 4b * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
find_max_context_layers(model, basemodel, unified_model, sig_idx=1)

### ***** Supplementary Fig-9 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
plot_vp_group_layers_contexts(basemodel, model, unified_model, plot_idx=0, modality_idx=0, sig_idx=0, ymin=0, ymax=0.05)
plot_vp_group_layers_contexts(basemodel, model, unified_model, plot_idx=0, modality_idx=1, sig_idx=0, ymin=0, ymax=0.05)
plot_vp_group_layers_contexts(basemodel, model, unified_model, plot_idx=0, modality_idx=2, sig_idx=0, ymin=0, ymax=0.05)

plot_vp_group_layers_contexts(basemodel, model, unified_model, plot_idx=1, modality_idx=0, sig_idx=0, ymin=0, ymax=0.05)
plot_vp_group_layers_contexts(basemodel, model, unified_model, plot_idx=1, modality_idx=1, sig_idx=0, ymin=0, ymax=0.05)
plot_vp_group_layers_contexts(basemodel, model, unified_model, plot_idx=1, modality_idx=2, sig_idx=0, ymin=0, ymax=0.05)



##########################################################################################################################################################################
##########################################################################################################################################################################
### Weight correlation
##########################################################################################################################################################################
##########################################################################################################################################################################
basemodel = "gpt_nprandom"
summary_weightcorrs_contexts_tvoxels(model, basemodel, model_base=model_base)

## Figure. 5 #############################################################################################################################################################
summary_weightcorrs_contexts_tvoxels_unified(basemodel, unified_model)

for sig_idx in [ 0 ]:
    for svoxel_idx in [ 0 ]:
        for plot_idx in [ 0, 1 ]:                
            ### Fig. 4c * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            plot_weightcorr_context_X_layers_vp_RGB(basemodel, model, plot_idx=0, ymin=-0.2, ymax=0.2, sig_idx=0)

            ### Supplementary Fig. 12a/b * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            plot_weightcorr_context_X_layers_vp_RGB(basemodel, model, plot_idx=1, ymin=-0.2, ymax=0.2, sig_idx=0)

            ### Fig. 2d and Supplementary Fig. 10b * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            plot_weightcorr_context_X_layers_intra_cross_svoxels(basemodel, model, unified_model, plot_idx=0, ymin=-0.2, ymax=0.4, sig_idx=0)

            ### Supplementary Fig. 10a * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            plot_weightcorr_context_X_layers_intra_cross_svoxels(basemodel, model, unified_model, plot_idx=1, ymin=-0.2, ymax=0.4, sig_idx=0)

            ### Fig. 3c & Supplementary Fig. 11b * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            plot_weightcorr_context_X_layers_unified(basemodel, model, unified_model, plot_idx=0, ymin=0, ymax=0.9, sig_idx=0)

            ### Supplementary Fig. 11a * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
            plot_weightcorr_context_X_layers_unified(basemodel, model, unified_model, plot_idx=1, ymin=0, ymax=1, sig_idx=0)


##########################################################################################################################################################################
##########################################################################################################################################################################
### linear mixed effects modeling
##########################################################################################################################################################################
##########################################################################################################################################################################
basemodel = "gpt_nprandom"
summary_data_for_lmer(basemodel, model, unified_model)


##########################################################################################################################################################################
##########################################################################################################################################################################
### individual weight PCA
##########################################################################################################################################################################
##########################################################################################################################################################################
basemodel = "gpt_nprandom"
model_prefix='chatgpt'
summary_indiv_pca_expvar(basemodel, model_prefix)
summary_indiv_pc_scores(basemodel, model_prefix)
summary_indiv_pca_bootstrap(basemodel, model_prefix)

for subject_idx in range(len(SUBJECTS_ALL)):
    summary_indiv_pca_bootstrap_each(subject_idx, basemodel, model_prefix)


