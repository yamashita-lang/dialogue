import numpy as np
import pandas as pd
import tables
from scipy.stats import pearsonr
import os
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from dialogue.github.modeling.io import save_table_file, get_ready_himalaya, get_ready_statsfile, get_ready_primal
from dialogue.github.modeling.config import SUBJECTS_ALL, TOTAL_RUNS, RESULTS_DIR, NVOXELS, FIG_DIR, LAYERS, CONTEXT_LENGTH

### parameters
figsize=8
figbottom=8
fontsize=5
blank_space=0.05
dpi=150
letter_alpha=0.6
linewidth=0.2
cm=1/2.54
sns.set_context("paper", 0.8, {"lines.linewidth": 0.2})
sns.set_style('ticks')


##########################################################################################################################################################################
##########################################################################################################################################################################
### statistics
##########################################################################################################################################################################
##########################################################################################################################################################################
def utterance_stats():
    postfix='_fulltrs.npy'
    model_name='rinna/japanese-gpt-neox-3.6b-instruction-sft'
    # 0: production, 1: comprehension, 2: production and comprehension
    utterance_labels = np.full((len(CONTEXT_LENGTH), TOTAL_RUNS.max(), N_SCANS, 3, len(SUBJECTS_ALL)), np.nan)
    uttered_scans = np.full((len(CONTEXT_LENGTH), TOTAL_RUNS.max(), 3, len(SUBJECTS_ALL)), np.nan)
    for subject_idx in range(len(SUBJECTS_ALL)):
        subject=SUBJECTS_ALL[subject_idx]
        for cl_idx, cl in enumerate(CONTEXT_LENGTH):
            print("============================================")
            print(f"    {subject}_cl-{cl}")
            print("============================================")
            if cl == 1:
                cl_name = ''
                prefix = '_layer_0'
            else:
                cl_name = f'_cl-{cl}'
                prefix = '_cl-1_layer-0'

            for run_i in range(TOTAL_RUNS[subject_idx]):
                run = run_i+1
                utterance_labels[cl_idx, run_i, :, :, subject_idx] = 1
                prod_data = np.load(os.path.join(FEATURES_DIR, 'CHATGPTNEOX', subject, model_name, f'{subject}_{run:02}_prod{cl_name}', f'chatgptneox_avrg{prefix}{postfix}'))
                prod_noutter_idx = np.all(prod_data==0, axis=1)
                comp_data=np.load(os.path.join(FEATURES_DIR, 'CHATGPTNEOX', subject, model_name, f'{subject}_{run:02}_comp{cl_name}', f'chatgptneox_avrg{prefix}{postfix}'))
                comp_noutter_idx = np.all(comp_data==0, axis=1)
                
                utterance_labels[cl_idx, run_i, prod_noutter_idx, 0, subject_idx] = 0
                utterance_labels[cl_idx, run_i, comp_noutter_idx, 1, subject_idx] = 0
                utterance_labels[cl_idx, run_i, :, 2, subject_idx] = np.where((utterance_labels[cl_idx, run_i, :, 0, subject_idx] != 0) & (utterance_labels[cl_idx, run_i, :, 1, subject_idx] != 0), 1, 0)

                uttered_scans[cl_idx, run_i, 0, subject_idx] = np.sum(utterance_labels[cl_idx, run_i, :, 0, subject_idx])
                uttered_scans[cl_idx, run_i, 1, subject_idx] = np.sum(utterance_labels[cl_idx, run_i, :, 1, subject_idx])
                uttered_scans[cl_idx, run_i, 2, subject_idx] = np.sum(utterance_labels[cl_idx, run_i, :, 2, subject_idx])

                print(f"RUN-{run}: prod: {uttered_scans[cl_idx, run_i, 0, subject_idx]}, comp: {uttered_scans[cl_idx, run_i, 1, subject_idx]}, both: {uttered_scans[cl_idx, run_i, 2, subject_idx]}")
            
    output = f"{RESULTS_DIR}/ALLSUBJ_utterance_stats_cl.hdf"
    save_table_file(output, dict(utterance_labels=utterance_labels, uttered_scans=uttered_scans))



##########################################################################################################################################################################
##########################################################################################################################################################################
### Head Motion model
##########################################################################################################################################################################
##########################################################################################################################################################################
def summary_headmotion_model(sig_idx=0):
    headmotion_model = 'headmotion'
    scores_all=np.full((len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    for subject_idx in range(len(SUBJECTS_ALL)):
        subj_n_voxels=NVOXELS[subject_idx]
        subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file = get_ready_himalaya(subject_idx, 1, model=headmotion_model, basemodel=headmotion_model, layer=-1) 
        with tables.open_file(r_im_file, "r") as MODEL_f:
            if sig_idx == 0:
                tmp_scores_all = MODEL_f.root.raw_r_scores_all.read()
            elif sig_idx == 1:
                tmp_scores_all = MODEL_f.root.sig_r_scores_all.read()

        scores_all[subject_idx,:subj_n_voxels] = tmp_scores_all
    
    if sig_idx == 0:
        output = f"{RESULTS_DIR}/ALLSUBJ_raw_corr_{headmotion_model}{postfix2}.hdf"
    elif sig_idx == 1:
        output = f"{RESULTS_DIR}/ALLSUBJ_sig_corr_{headmotion_model}{postfix2}.hdf"

    print(f"saving {output}")
    save_table_file(output, dict(scores_all=scores_all))



##########################################################################################################################################################################
##########################################################################################################################################################################
### Random embedding model
##########################################################################################################################################################################
##########################################################################################################################################################################
def summary_random_intramodal(sig_idx=0):
    headmotion_model = 'headmotion'
    random_model = "gpt_nprandom"
    scores_all=np.full((len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    scores_split=np.full((2, len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    for subject_idx in range(len(SUBJECTS_ALL)):
        subj_n_voxels=NVOXELS[subject_idx]
        subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file = get_ready_himalaya(subject_idx, 1, random_model, headmotion_model, layer=0, cl=1)
        with tables.open_file(r_im_file, "r") as MODEL_f:
            if sig_idx == 0:
                tmp_scores_all = MODEL_f.root.raw_r_scores_all.read()
                tmp_scores_split = MODEL_f.root.raw_r_scores_split.read()
            elif sig_idx == 1:
                tmp_scores_all = MODEL_f.root.sig_r_scores_all.read()
                tmp_scores_split = MODEL_f.root.sig_r_scores_split.read()

        scores_all[ subject_idx,:subj_n_voxels] = tmp_scores_all
        scores_split[ :, subject_idx, :subj_n_voxels] = tmp_scores_split

    if sig_idx == 0:
        output = f"{RESULTS_DIR}/ALLSUBJ_raw_corr_{headmotion_model}_{random_model}_intramodal.hdf"
    elif sig_idx == 1:
        output = f"{RESULTS_DIR}/ALLSUBJ_sig_corr_{headmotion_model}_{random_model}_intramodal.hdf"

    save_table_file(output, dict(scores_all=scores_all, scores_split=scores_split))


def summary_random_crossmodal(sig_idx=0):
    headmotion_model = 'headmotion'
    random_model = "gpt_nprandom"    
    scores_all=np.empty((len(SUBJECTS_ALL), NVOXELS.max()))
    for subject_idx in range(len(SUBJECTS_ALL)):
        subj_n_voxels=NVOXELS[subject_idx]
        subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file = get_ready_primal(subject_idx, 1, random_model, headmotion_model, layer=0, cl=1)
        with tables.open_file(primal_r_cm_file, "r") as CROSS_f:
            if sig_idx == 0:
                tmp_scores_all = CROSS_f.root.raw_r_scores_all.read()
            elif sig_idx == 1:
                tmp_scores_all = CROSS_f.root.sig_r_scores_all.read()

        scores_all[subject_idx,:subj_n_voxels] = tmp_scores_all

    if sig_idx == 0:
        output = f"{RESULTS_DIR}/ALLSUBJ_raw_corr_{headmotion_model}_{model_name}_crossmodal.hdf"
    elif sig_idx == 1:
        output = f"{RESULTS_DIR}/ALLSUBJ_sig_corr_{headmotion_model}_{model_name}_crossmodal.hdf"

    save_table_file(output, dict(scores_all=scores_all))



##########################################################################################################################################################################
##########################################################################################################################################################################
### Separate model
##########################################################################################################################################################################
##########################################################################################################################################################################
### Figure 2: Separate and Unified Linguistic model * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
def summary_intramodal_context(model, basemodel, model_prefix, sig_idx=0):
    scores_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    scores_split=np.full((len(LAYERS), len(CONTEXT_LENGTH), 2, len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    for subject_idx in range(len(SUBJECTS_ALL)):
        subj_n_voxels=NVOXELS[subject_idx]
        for cl_idx, cl in enumerate(CONTEXT_LENGTH):
            tmp_model = f'{model_prefix}_{cl}'
            for layer_idx, layer in enumerate(LAYERS):
                _, _, _, _, _, _, _, _, r_im_file, _ = get_ready_himalaya(subject_idx, 1, tmp_model, basemodel, layer=layer, cl=cl)
                with tables.open_file(r_im_file, "r") as MODEL_f:
                    if sig_idx == 0:
                        tmp_scores_all = MODEL_f.root.raw_r_scores_all.read()
                        tmp_scores_split = MODEL_f.root.raw_r_scores_split.read()
                    elif sig_idx == 1:
                        tmp_scores_all = MODEL_f.root.sig_r_scores_all.read()
                        tmp_scores_split = MODEL_f.root.sig_r_scores_split.read()

                scores_all[layer_idx, cl_idx, subject_idx,:subj_n_voxels] = tmp_scores_all
                scores_split[layer_idx, cl_idx, :, subject_idx, :subj_n_voxels] = tmp_scores_split

    if sig_idx == 0:
        output = f"{RESULTS_DIR}/ALLSUBJ_raw_corr_{basemodel}_{model}_intramodal.hdf"
    elif sig_idx == 1:
        output = f"{RESULTS_DIR}/ALLSUBJ_sig_corr_{basemodel}_{model}_intramodal.hdf"

    save_table_file(output, dict(scores_all=scores_all, scores_split=scores_split))


def summary_crossmodal_context(model, basemodel, model_prefix, sig_idx=0, svoxel_idx=1):
    if svoxel_idx == 1:
        intramodal_file = f"{RESULTS_DIR}/ALLSUBJ_sig_corr_{basemodel}_{model}_intramodal.hdf"
        with tables.open_file(intramodal_file, "r") as INTRA_f:
            intramodal_scores_all = INTRA_f.root.scores_all.read()
            
    scores_all=np.empty((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()))
    scores_split=np.empty((len(LAYERS), len(CONTEXT_LENGTH), 2, len(SUBJECTS_ALL), NVOXELS.max()))
    for subject_idx in range(len(SUBJECTS_ALL)):
        subj_n_voxels=NVOXELS[subject_idx]
        for cl_idx, cl in enumerate(CONTEXT_LENGTH):
            tmp_model = f'{model_prefix}_{cl}'
            for layer_idx, layer in enumerate(LAYERS):
                _, _, _, _, _, _, primal_r_cm_file, _ = get_ready_primal(subject_idx, 1, tmp_model, basemodel, layer=layer, cl=cl)
                with tables.open_file(primal_r_cm_file, "r") as CROSS_f:
                    if sig_idx == 0:
                        tmp_scores_all = CROSS_f.root.raw_r_scores_all.read()
                    elif sig_idx == 1:
                        tmp_scores_all = CROSS_f.root.sig_r_scores_all.read()

                scores_all[layer_idx, cl_idx, subject_idx,:subj_n_voxels] = tmp_scores_all

    if sig_idx == 0:
        if svoxel_idx == 0:
            output = f"{RESULTS_DIR}/ALLSUBJ_raw_corr_{basemodel}_{model}_crossmodal.hdf"
        elif svoxel_idx == 1:
            output = f"{RESULTS_DIR}/ALLSUBJ_raw_corr_{basemodel}_{model}_crossmodal_svoxels.hdf"
    elif sig_idx == 1:
        if svoxel_idx == 0:
            output = f"{RESULTS_DIR}/ALLSUBJ_sig_corr_{basemodel}_{model}_crossmodal.hdf"
        elif svoxel_idx == 1:
            output = f"{RESULTS_DIR}/ALLSUBJ_sig_corr_{basemodel}_{model}_crossmodal_svoxels.hdf"

    save_table_file(output, dict(scores_all=scores_all))


def summary_intramodal_unified(basemodel, unified_model, unified_model_prefix, sig_idx=0):
    scores_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    for subject_idx in range(len(SUBJECTS_ALL)):
        subj_n_voxels=NVOXELS[subject_idx]
        for cl_idx, cl in enumerate(CONTEXT_LENGTH):
            model = f'{unified_model_prefix}_{cl}'
            for layer_idx, layer in enumerate(LAYERS):
                _, _, _, _, _, _, _, _, r_im_file, _ = get_ready_himalaya(subject_idx, 1, model, basemodel, layer=layer, cl=cl)
                with tables.open_file(r_im_file, "r") as MODEL_f:
                    if sig_idx == 0:
                        tmp_scores_all = MODEL_f.root.raw_r_scores_all.read()
                    elif sig_idx == 1:
                        tmp_scores_all = MODEL_f.root.sig_r_scores_all.read()

                scores_all[layer_idx, cl_idx, subject_idx, :subj_n_voxels] = tmp_scores_all

    if sig_idx == 0:
        output = f"{RESULTS_DIR}/ALLSUBJ_raw_corr_{basemodel}_{unified_model}.hdf"
    elif sig_idx == 1:
        output = f"{RESULTS_DIR}/ALLSUBJ_sig_corr_{basemodel}_{unified_model}.hdf"

    save_table_file(output, dict(scores_all=scores_all))



##########################################################################################################################################################################
##########################################################################################################################################################################
### Variance partitioning
##########################################################################################################################################################################
##########################################################################################################################################################################
def summary_variance_partitioning(model, basemodel, model_prefix):
    RC_intersection_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    RC_prod_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    RC_comp_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    for subject_idx in range(len(SUBJECTS_ALL)):
        for cl_idx, cl in enumerate(CONTEXT_LENGTH):
            model = f'{model_prefix}_{cl}'
            for layer_idx, layer in enumerate(LAYERS):
                postfix = f"_layer-{layer}_cl-{cl}"
                vp_file = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_{basemodel}_{model}{postfix}_vp.hdf"
                print(vp_file)
                with tables.open_file(vp_file, "r") as VP_f:
                    RC_intersection = VP_f.root.RC_intersection.read()
                    RC_prod = VP_f.root.RC_prod.read()
                    RC_comp = VP_f.root.RC_comp.read()

                r_all_file = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_{basemodel}_{model}{postfix}_intramodal.hdf"
                with tables.open_file(r_all_file, "r") as MODEL_ALL_R_f:
                    svoxels = MODEL_ALL_R_f.root.svoxels.read()

                RC_intersection_all[layer_idx, cl_idx, subject_idx, :NVOXELS[subject_idx]] = RC_intersection
                RC_prod_all[layer_idx, cl_idx, subject_idx, :NVOXELS[subject_idx]] = RC_prod
                RC_comp_all[layer_idx, cl_idx, subject_idx, :NVOXELS[subject_idx]] = RC_comp

    output = f"{RESULTS_DIR}/ALLSUBJ_vp_{basemodel}_{model}.hdf"
    save_table_file(output, dict(RC_intersection_all=RC_intersection_all, RC_prod_all=RC_prod_all, RC_comp_all=RC_comp_all))


def summary_best_vp(model, basemodel, model_prefix):
    VRGB_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    for subject_idx in range(len(SUBJECTS_ALL)):
        for cl_idx, cl in enumerate(CONTEXT_LENGTH):
            model = f'{model_prefix}_{cl}'
            for layer_idx, layer in enumerate(LAYERS):
                postfix = f"_layer-{layer}_cl-{cl}"
                vp_file = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_{basemodel}_{model}{postfix}_intramodal_best_vp.hdf"
                tmp = np.full(NVOXELS[subject_idx], np.nan)
                with tables.open_file(vp_file, "r") as VP_f:
                    VRGB = VP_f.root.VRGB.read()

                r_all_file = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_{basemodel}_{model}{postfix}_intramodal.hdf"
                with tables.open_file(r_all_file, "r") as MODEL_ALL_R_f:
                    svoxels = MODEL_ALL_R_f.root.svoxels.read()

                VRGB_all[layer_idx, cl_idx, subject_idx, :NVOXELS[subject_idx]] = VRGB

    output = f"{RESULTS_DIR}/ALLSUBJ_best_vp_{basemodel}_{model}.hdf"
    save_table_file(output, dict(VRGB_all=VRGB_all))


def find_max_context_layers(model, basemodel, unified_model):
    intra_data, cross_data, integrated_data, prod_data, comp_data, intersection_data = get_summary_data(model, basemodel, unified_model)
    sig_name = "sig"
    nsvoxels = np.isnan(intra_data)
    prod_data[nsvoxels] = np.nan
    comp_data[nsvoxels] = np.nan
    intersection_data[nsvoxels] = np.nan 

    data_name = f"Z_Max_context_{sig_name}"
    prod_data = np.arctanh(prod_data)
    comp_data = np.arctanh(comp_data)
    intersection_data = np.arctanh(intersection_data)

    ## 0: production, 1: comprehension, 2: intersectino
    max_cl = np.full((len(LAYERS), len(SUBJECTS_ALL), 3), np.nan)
    for subject_idx in range(len(SUBJECTS_ALL)):
        subject = SUBJECTS_ALL[subject_idx]
        subj_n_voxels=NVOXELS[subject_idx]
        for layer_idx, layer in enumerate(LAYERS):
            mean_prod = np.nanmean(prod_data[layer_idx, :, subject_idx, :], axis=1)
            mean_comp = np.nanmean(comp_data[layer_idx, :, subject_idx, :], axis=1)
            mean_is = np.nanmean(intersection_data[layer_idx, :, subject_idx, :], axis=1)
            max_cl[layer_idx, subject_idx, 0] = CONTEXT_LENGTH[np.argmax(mean_prod)]
            max_cl[layer_idx, subject_idx, 1] = CONTEXT_LENGTH[np.argmax(mean_comp)]
            max_cl[layer_idx, subject_idx, 2] = CONTEXT_LENGTH[np.argmax(mean_is)]

    df = get_index_3data_modality(max_cl)
    
    ## make figure
    figsize=4
    figbottom=5
    color=[ '#192a9a', '#ff3616', '#00b81b']
    fig, ax = plt.subplots(facecolor='white', figsize = (figsize*cm, figbottom*cm))
    sns.lineplot(data=df, x='subject', y='value', hue='modality', palette=color, linewidth=0.5, ax=ax, orient="x")
    plt.xlabel('Participants')
    plt.ylabel('Best context length [s]')
    ax.set_yticks(CONTEXT_LENGTH)
    ax.legend_.remove()
    tmp_FIG_DIR=f"{FIG_DIR}{data_name}/"
    if not os.path.isdir(tmp_FIG_DIR):
        os.makedirs(tmp_FIG_DIR)  

    output = f"{tmp_FIG_DIR}{data_name}_{basemodel}_{model}.pdf"
    plt.savefig(output, format='pdf', bbox_inches='tight')
    plt.close()


def get_index_3data_modality(data):
    layer_idx, subject_idx, modality_idx = np.indices(data.shape)
    layer = layer_idx.flatten()
    layer = (layer)*3
    subject = subject_idx.flatten()
    subject = subject+1
    modality = modality_idx.flatten()
    replace_dict = {0: 'Production', 1: 'Comprehension', 2: 'Intersection'}
    replace_func = np.vectorize(replace_dict.get)
    modality = replace_func(modality)
    values = data.flatten()
    df = pd.DataFrame({'layer': layer, 'modality': modality, 'subject': subject, 'value': values})
    return df


def get_summary_data(model, basemodel, unified_model, sig_idx=0):
    if  sig_idx == 0:
        sig_name = "raw"
    elif sig_idx == 1:
        sig_name = "sig"

    intra_file=f"{RESULTS_DIR}/ALLSUBJ_{sig_name}_corr_{basemodel}_{model}_intramodal.hdf"
    cross_file=f"{RESULTS_DIR}/ALLSUBJ_{sig_name}_corr_{basemodel}_{model}_crossmodal.hdf"
    unified_file=f"{RESULTS_DIR}/ALLSUBJ_{sig_name}_corr_{basemodel}_{unified_model}.hdf"
    vpfile = f"{RESULTS_DIR}/ALLSUBJ_vp_{basemodel}_{model}.hdf"
    with tables.open_file(vpfile, "r") as VP_f:
        prod_data = VP_f.root.RC_prod_all.read()
        comp_data = VP_f.root.RC_comp_all.read()
        intersection_data = VP_f.root.RC_intersection_all.read()

    with tables.open_file(intra_file, "r") as MODEL_f:
        intra_data=MODEL_f.root.scores_all.read()

    with tables.open_file(cross_file, "r") as MODEL_f:
        cross_data=MODEL_f.root.scores_all.read()

    with tables.open_file(unified_file, "r") as MODEL_f:
        integrated_data=MODEL_f.root.scores_all.read()

    return intra_data, cross_data, integrated_data, prod_data, comp_data, intersection_data


def corrcoef_prod_vs_comp_across_context(model, basemodel):
    intra_data, cross_data, integrated_data, prod_data, comp_data, intersection_data = get_summary_data(model, basemodel)
    data_name = "corrcoef_prod_vs_comp_across_contexts"
    df1 = get_index_4data(prod_data)
    df2 = get_index_4data(comp_data)
    R = np.full((len(LAYERS), len(CONTEXT_LENGTH), len(CONTEXT_LENGTH), 8), np.nan)
    for subject_idx in range(lne(SUBJECTS_ALL)):
        subj_df1 = df1[df1['subject'] == subject_idx]
        subj_df2 = df2[df2['subject'] == subject_idx]
        for layer_idx, layer in enumerate(LAYERS):
            layer_cond1 = subj_df1['layer'] == layer
            layer_cond2 = subj_df2['layer'] == layer
            for prod_cl_idx, prod_cl in enumerate(CONTEXT_LENGTH):
                for comp_cl_idx, comp_cl in enumerate(CONTEXT_LENGTH):
                    cl_cond1 = subj_df1['cl'] == prod_cl
                    cl_cond2 = subj_df2['cl'] == comp_cl
                    tmp_df1 = subj_df1[layer_cond1 & cl_cond1]
                    tmp_df2 = subj_df2[layer_cond2 & cl_cond2]
                    scores_1 = tmp_df1['value'].to_numpy()
                    scores_2 = tmp_df2['value'].to_numpy()
                    scores_2[np.isnan(scores_1)] = np.nan
                    scores_1[np.isnan(scores_2)] = np.nan
                    scores_1 = scores_1[~np.isnan(scores_1)]
                    scores_2 = scores_2[~np.isnan(scores_2)]
                    [r, p] = pearsonr(scores_1, scores_2)
                    R[layer_idx, prod_cl_idx, comp_cl_idx, subject_idx] = r

    output =f"{RESULTS_DIR}/ALLSUBJ_{data_name}_{basemodel}_{model}.hdf"
    save_table_file(output, dict(R=R))


def corrcoef_prod_vs_comp_across_layers(model, basemodel):
    intra_data, cross_data, integrated_data, prod_data, comp_data, intersection_data = get_summary_data(model, basemodel)
    data_name = "corrcoef_prod_vs_comp_across_layers"
    df1 = get_index_4data(prod_data)
    df2 = get_index_4data(comp_data)
    R = np.full((len(CONTEXT_LENGTH), len(LAYERS), len(LAYERS), len(SUBJECTS_ALL)), np.nan)
    for subject_idx in range(len(SUBJECTS_ALL)):
        subj_df1 = df1[df1['subject'] == subject_idx]
        subj_df2 = df2[df2['subject'] == subject_idx]
        for cl_idx, cl in enumerate(CONTEXT_LENGTH):
            cl_cond1 = subj_df1['cl'] == cl
            cl_cond2 = subj_df2['cl'] == cl
            for prod_layer_idx, prod_layer in enumerate(LAYERS):
                for comp_layer_idx, comp_layer in enumerate(LAYERS):
                    layer_cond1 = subj_df1['layer'] == prod_layer
                    layer_cond2 = subj_df2['layer'] == comp_layer
                    tmp_df1 = subj_df1[cl_cond1 & layer_cond1]
                    tmp_df2 = subj_df2[cl_cond2 & layer_cond2]
                    scores_1 = tmp_df1['value'].to_numpy()
                    scores_2 = tmp_df2['value'].to_numpy()
                    scores_2[np.isnan(scores_1)] = np.nan
                    scores_1[np.isnan(scores_2)] = np.nan
                    scores_1 = scores_1[~np.isnan(scores_1)]
                    scores_2 = scores_2[~np.isnan(scores_2)]
                    [r, p] = pearsonr(scores_1, scores_2)
                    R[cl_idx, prod_layer_idx, comp_layer_idx, subject_idx] = r

    output =f"{RESULTS_DIR}/ALLSUBJ_{data_name}_{basemodel}_{model}.hdf"
    save_table_file(output, dict(R=R))


def get_index_4data(data):
    layer_idx, cl_idx, subject_idx, voxel_idx = np.indices(data.shape)
    layer = layer_idx.flatten()
    layer = (layer)*3
    cl = cl_idx.flatten()
    cl = np.power(2, cl)
    subject = subject_idx.flatten()
    voxel = voxel_idx.flatten()
    values = data.flatten()
    df = pd.DataFrame({'layer': layer, 'cl': cl, 'subject': subject, 'voxel': voxel, 'value': values})
    return df

def get_index_3data_layer(data):
    layer_idx, subject_idx, voxel_idx = np.indices(data.shape)
    layer = layer_idx.flatten()
    layer = (layer)*3
    subject = subject_idx.flatten()
    voxel = voxel_idx.flatten()
    values = data.flatten()
    df = pd.DataFrame({'layer': layer, 'subject': subject, 'voxel': voxel, 'value': values})
    return df



####################################################################################################################################
####################################################################################################################################
### prediction performance
####################################################################################################################################
####################################################################################################################################
def summary_data_for_lmer(basemodel, model, unified_model):
    ### corrected for RC_prod, RC_comp
    ### using averaged score instead of ses scores
    ### for feature importance
    ## brain score ~ r averaged across sessions
    score_name='r'
    subject_array = np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    layer_array = np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    cl_array = np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_intra =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_cross =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_integrated =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_prod =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_comp =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_is =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_weightcorr =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_weightcorr_cross =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_weightcorr_prod =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_weightcorr_comp =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_weightcorr_is =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_weightcorr_uni_prod =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)
    Y_weightcorr_uni_comp =np.full((len(LAYERS)*len(CONTEXT_LENGTH)*len(SUBJECTS_ALL), 1), np.nan)

    intra_file = f"{RESULTS_DIR}/ALLSUBJ_raw_corr_{basemodel}_{model}_intramodal.hdf"
    intra_sig_file = f"{RESULTS_DIR}/ALLSUBJ_sig_corr_{basemodel}_{model}_intramodal.hdf"
    cross_file = f"{RESULTS_DIR}/ALLSUBJ_raw_corr_{basemodel}_{model}_crossmodal.hdf"
    unified_file = f"{RESULTS_DIR}/ALLSUBJ_raw_corr_{basemodel}_{unified_model}.hdf"
    vp_file = f"{RESULTS_DIR}/ALLSUBJ_vp_{basemodel}_{model}.hdf"
    bestvp_file = f"{RESULTS_DIR}/ALLSUBJ_best_vp_{basemodel}_{model}.hdf"
    wc_file = f"{RESULTS_DIR}/ALLSUBJ_weightcorrs_{basemodel}_{model}_tvoxels.hdf"
    wc_uni_file = f"{RESULTS_DIR}/ALLSUBJ_weightcorrs_{basemodel}_{unified_model}_tvoxels.hdf"

    with tables.open_file(intra_file) as INTRA_f:
        X_intra = INTRA_f.root.scores_all.read()

    with tables.open_file(intra_sig_file) as INTRA_S_f:
        X_sig_intra = INTRA_S_f.root.scores_all.read()

    with tables.open_file(cross_file) as CROSS_f:
        X_cross = CROSS_f.root.scores_all.read()

    with tables.open_file(unified_file) as UNI_f:
        X_unified = UNI_f.root.scores_all.read()

    with tables.open_file(vp_file) as VP_f:
        X_prod = VP_f.root.RC_prod_all.read()
        X_comp = VP_f.root.RC_comp_all.read()
        X_is = VP_f.root.RC_intersection_all.read()
        
    with tables.open_file(bestvp_file) as BESTVP_f:
        X_bestvp = BESTVP_f.root.VRGB_all.read()

    with tables.open_file(wc_file) as WC_f:
        X_wc = WC_f.root.weightcorr_all.read()

    with tables.open_file(wc_uni_file) as WC_UNI_f:
        X_wc_uni_prod = WC_UNI_f.root.prodcorr_all.read()
        X_wc_uni_comp = WC_UNI_f.root.compcorr_all.read()

    X_wc[np.isnan(X_sig_intra)] = np.nan
    X_wc_cross = copy.deepcopy(X_wc)
    X_wc_prod = copy.deepcopy(X_wc)
    X_wc_comp = copy.deepcopy(X_wc)
    X_wc_is = copy.deepcopy(X_wc)

    X_cross[np.isnan(X_sig_intra)] = np.nan
    X_prod[np.isnan(X_sig_intra)] = np.nan
    X_comp[np.isnan(X_sig_intra)] = np.nan
    X_is[np.isnan(X_sig_intra)] = np.nan
    
    X_wc_cross[X_sig_intra<=0.05] = np.nan
    X_wc_cross[X_cross<=0.05] = np.nan

    X_wc_prod[X_bestvp==2] = np.nan
    X_wc_prod[X_bestvp==3] = np.nan

    X_wc_comp[X_bestvp==1] = np.nan
    X_wc_comp[X_bestvp==3] = np.nan

    X_wc_is[X_bestvp==1] = np.nan
    X_wc_is[X_bestvp==2] = np.nan

    data_idx=0
    for subject_idx in range(8):
        subj_n_voxels = NVOXELS[subject_idx]
        print('////////////////////////////////////')
        print(f"    {SUBJECTS_ALL[subject_idx]}")
        print('////////////////////////////////////')
        for cl_idx, cl in enumerate(CONTEXT_LENGTH):
            for layer_idx, layer in enumerate(LAYERS):
                grandmean_brainscore_intra = np.tanh(np.nanmean(np.arctanh(X_intra[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_brainscore_cross = np.tanh(np.nanmean(np.arctanh(X_cross[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_brainscore_integrated = np.tanh(np.nanmean(np.arctanh(X_unified[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_brainscore_prod = np.tanh(np.nanmean(np.arctanh(X_prod[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_brainscore_comp = np.tanh(np.nanmean(np.arctanh(X_comp[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_brainscore_is = np.tanh(np.nanmean(np.arctanh(X_is[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_weightcorrs = np.tanh(np.nanmean(np.arctanh(X_wc[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_weightcorrs_cross = np.tanh(np.nanmean(np.arctanh(X_wc_cross[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_weightcorrs_prod = np.tanh(np.nanmean(np.arctanh(X_wc_prod[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_weightcorrs_comp = np.tanh(np.nanmean(np.arctanh(X_wc_comp[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_weightcorrs_is = np.tanh(np.nanmean(np.arctanh(X_wc_is[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_weightcorrs_uni_prod = np.tanh(np.nanmean(np.arctanh(X_wc_uni_prod[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))
                grandmean_weightcorrs_uni_comp = np.tanh(np.nanmean(np.arctanh(X_wc_uni_comp[layer_idx, cl_idx, subject_idx, :subj_n_voxels])))

                Y_intra[data_idx] = grandmean_brainscore_intra
                Y_cross[data_idx] = grandmean_brainscore_cross
                Y_integrated[data_idx] = grandmean_brainscore_integrated
                Y_prod[data_idx] = grandmean_brainscore_prod
                Y_comp[data_idx] = grandmean_brainscore_comp
                Y_is[data_idx] = grandmean_brainscore_is
                Y_weightcorr[data_idx] = grandmean_weightcorrs
                Y_weightcorr_cross[data_idx] = grandmean_weightcorrs_cross
                Y_weightcorr_prod[data_idx] = grandmean_weightcorrs_prod
                Y_weightcorr_comp[data_idx] = grandmean_weightcorrs_comp
                Y_weightcorr_is[data_idx] = grandmean_weightcorrs_is
                Y_weightcorr_uni_prod[data_idx] = grandmean_weightcorrs_uni_prod
                Y_weightcorr_uni_comp[data_idx] = grandmean_weightcorrs_uni_comp

                subject_array[data_idx] = subject_idx+1
                layer_array[data_idx] = layer
                cl_array[data_idx] = cl
                data_idx = data_idx + 1

    ### save data 
    print(f"subject_array: {subject_array.shape}")
    print(f"layer_array: {layer_array.shape}")
    print(f"cl_array: {cl_array.shape}")
    print(f"Y_intra: {Y_intra.shape}")
    print(f"Y_cross: {Y_cross.shape}")
    print(f"Y_integrated: {Y_integrated.shape}")
    print(f"Y_prod: {Y_prod.shape}")
    print(f"Y_comp: {Y_comp.shape}")
    print(f"Y_is: {Y_is.shape}")
    print(f"Y_weightcorr: {Y_weightcorr.shape}")
    data=np.hstack((subject_array, layer_array, cl_array, Y_intra, Y_cross, Y_integrated, Y_prod, Y_comp, Y_is, Y_weightcorr, Y_weightcorr_cross, Y_weightcorr_prod, Y_weightcorr_comp, Y_weightcorr_is, Y_weightcorr_uni_prod, Y_weightcorr_uni_comp ))
    csvfile = f"{RESULTS_DIR}/ALLSUBJ_raw_mean_{score_name}_scores_{basemodel}_{model}.csv"
    np.savetxt(csvfile, data, delimiter=',')



##########################################################################################################################################################################
##########################################################################################################################################################################
### Weight correlation
##########################################################################################################################################################################
##########################################################################################################################################################################
def summary_weightcorrs_contexts_tvoxels(model_name, basemodel, model_prefix):
    weightcorr_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    sig_weightcorr_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    for subject_idx in range(len(SUBJECTS_ALL)):       
        subj_n_voxels=NVOXELS[subject_idx]
        for layer_idx, layer in enumerate(LAYERS):
            for cl_idx, cl in enumerate(CONTEXT_LENGTH):
                model = f'{model_prefix}_{cl}'
                subject, weight_corr_svoxels, weight_corr_file  = get_ready_statsfile(subject_idx, model, basemodel, layer=layer, cl=cl)
                with tables.open_file(weight_corr_file, "r") as WEIGHT_f:
                    tmp_weightcorr = WEIGHT_f.root.corrs.read()
                    tmp_sig_weightcorr = WEIGHT_f.root.sig_corrs.read()

                weightcorr_all[layer_idx, cl_idx, subject_idx, :subj_n_voxels] = tmp_weightcorr
                sig_weightcorr_all[layer_idx, cl_idx, subject_idx, :subj_n_voxels] = tmp_sig_weightcorr

    output = f"{RESULTS_DIR}/ALLSUBJ_weightcorrs_{basemodel}_{model_name}_tvoxels.hdf"
    print(f"saving: {output}")
    save_table_file(output, dict(weightcorr_all=weightcorr_all, sig_weightcorr_all=sig_weightcorr_all))


def summary_weightcorrs_contexts_tvoxels_unified(basemodel, unified_model):
    prodcorr_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    compcorr_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    sig_prodcorr_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    sig_compcorr_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), NVOXELS.max()), np.nan)
    for subject_idx in range(len(SUBJECTS_ALL)):       
        subj_n_voxels=NVOXELS[subject_idx]
        for layer_idx, layer in enumerate(LAYERS):
            for cl_idx, cl in enumerate(CONTEXT_LENGTH):
                subject, weight_corr_svoxels, weight_corr_file  = get_ready_statsfile(subject_idx, unified_model, basemodel, layer=layer, cl=cl)
                with tables.open_file(weight_corr_file, "r") as WEIGHT_f:
                    tmp_prodcorr = WEIGHT_f.root.prod_corrs.read()
                    tmp_compcorr = WEIGHT_f.root.comp_corrs.read()
                    tmp_sig_prodcorr = WEIGHT_f.root.prod_sig_corrs.read()
                    tmp_sig_compcorr = WEIGHT_f.root.comp_sig_corrs.read()

                prodcorr_all[layer_idx, cl_idx, subject_idx, :subj_n_voxels] = tmp_prodcorr
                compcorr_all[layer_idx, cl_idx, subject_idx, :subj_n_voxels] = tmp_compcorr
                sig_prodcorr_all[layer_idx, cl_idx, subject_idx, :subj_n_voxels] = tmp_sig_prodcorr
                sig_compcorr_all[layer_idx, cl_idx, subject_idx, :subj_n_voxels] = tmp_sig_compcorr

    output = f"{RESULTS_DIR}/ALLSUBJ_weightcorrs_{basemodel}_{unified_model}_tvoxels.hdf"
    print(f"saving: {output}")
    save_table_file(output, dict(prodcorr_all=prodcorr_all, compcorr_all=compcorr_all, sig_prodcorr_all=sig_prodcorr_all, sig_compcorr_all=sig_compcorr_all ))

