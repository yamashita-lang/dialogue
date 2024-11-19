import os
import tables
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import copy

from dialogue.github.modeling.config import TESTSET_RUNS, TOTAL_RUNS, SUBJECTS_ALL, DATA_DIR, FEATURES_DIR, RESULTS_DIR, MODEL_FEATURE_MATRIX, N_SCANS, N_PERM
from dialogue.github.modeling.io import get_ready_himalaya, get_responses_ses, responses_train_test, get_train_test_runs, get_run_onsets, get_data_ready, get_deltas, load_hdf5_array, save_table_file, get_ready_primal


### parameters
# plt.style.use('dark_background')
figsize=5
figbottom=4
fontsize=5
blank_space=0.05
dpi=150
letter_alpha=0.6
LINEWIDTH=0.5
linewidth=LINEWIDTH
CM=1/2.54
cm=CM
# cmap = plt.get_cmap('Set1')
# sns.set(font_scale=0.5)
sns.set_context("paper", 0.8, {"lines.linewidth": 0.2})
sns.set_style('ticks')


##########################################################################################################################################################################
##########################################################################################################################################################################
### statistics
##########################################################################################################################################################################
##########################################################################################################################################################################  
### Supplementary Fig. 1: fMRI data samples * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
def barplot_uttered_scans():
    utter_statfile = f"{RESULTS_DIR}/ALLSUBJ_utterance_stats_cl.hdf"
    output=f"{FIG_DIR}boxplot_uttered_scans_summary_group_cl.pdf"
    with tables.open_file(utter_statfile, "r") as STAT_f:
        uttered_scans=STAT_f.root.uttered_scans.read()

    for modality_idx in [0, 1]:
        X = np.full(8, np.nan)
        for subject_idx in range(8):
            X[subject_idx] = np.nanmean(uttered_scans[0, :, modality_idx, subject_idx])
            print(f"{SUBJECTS_ALL[subject_idx]}: {X[subject_idx]}")

        print(f"group mean, std: {np.nanmean(X)}, {np.nanstd(X)}")      

    cl_idx, run_idx, modality_idx, subject_idx = np.indices(uttered_scans.shape)
    cl = cl_idx.flatten()
    cl = np.power(2, cl)
    run = run_idx.flatten()
    modality = modality_idx.flatten()
    subject = subject_idx.flatten()
    values = uttered_scans.flatten()
    # pandas
    df = pd.DataFrame({'cl': cl, 'run': run, 'modality':modality, 'subject': subject, 'value': values})
    layer_subject_mean = df.groupby(['cl', 'run', 'modality', 'subject'])['value'].mean().reset_index()
    print(layer_subject_mean)

    # make figure
    plt.rcParams['font.family'] = 'Arial'
    figsize=16
    figbottom=10
    fig, axes = plt.subplots(nrows=2, ncols=3, facecolor='white', figsize = (figsize*cm, figbottom*cm))
    err_kws = {'linewidth': 0.5}
    
    plot_name='_boxplot'
    color=['#273c75', '#e84118', '#4cd137']
    for i, ax in enumerate(axes.flat):
        print(f"cl-{CONTEXT_LENGTH[i]}")
        sns.boxplot(data=df, x='subject', y='value', hue='modality', palette=COLORS_3, fliersize=0, linewidth=0.5, medianprops={"color": "w", "linewidth": 0.5}, ax=ax)
        sns.stripplot(data=df, x='subject', y='value', hue='modality', palette=["grey", "grey", "grey"], dodge=True, size=0.8, ax=ax)

        ax.set_ylim(0, 350)
        ax.legend_.remove()
        # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, bbox_transform=plt.gcf().transFigure)
        ax.set_ylabel('Uttered volumes')
        ax.set_xlabel('Participants')

    print(f"saving {output}")
    plt.savefig(output, format='pdf', bbox_inches='tight')
    plt.close()



##########################################################################################################################################################################
##########################################################################################################################################################################
### Separate and Unified Linguistic model
##########################################################################################################################################################################
##########################################################################################################################################################################
### Fig 2b * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
### Supplementary Fig. 8 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
def plot_scores_all_group_layers_contexts(model, basemodel, unified_model, plot_idx, test_idx=0, ymin=0, ymax=0.25):
    if plot_idx == 0:
        plot_name = 'lineplot'
    elif plot_idx == 1:
        plot_name = 'boxplot'

    sig_name = "raw"
    postfix = 'layers_contexts'
    xlabel = 'Context lengths [s]'
    ylabel = 'Mean performance (r)'
    if test_idx == 0:
        data_name = f'Z_R_raw_separate'
        if plot_idx == 0:
            color="#624498"
        elif plot_idx == 1:
            color = mcolors.LinearSegmentedColormap.from_list("white_to_color", ["white", "#624498"]) 
    elif test_idx == 1:
        data_name = f'Z_R_raw_cross_svoxels'
        if plot_idx == 0:
            color="#f8b500"
        elif plot_idx == 1:
            color = mcolors.LinearSegmentedColormap.from_list("white_to_color", ["white", "#f8b500"]) 
    elif test_idx == 2:
        data_name = f'Z_R_raw_unified'
        if plot_idx == 0:
            color="#008DBD"
        elif plot_idx == 1:
            color = mcolors.LinearSegmentedColormap.from_list("white_to_color", ["white", "#008DBD"]) 

    intra_data, cross_data, unified_data, prod_data, comp_data, intersection_data = get_summary_data(model, basemodel, unified_model)
    if test_idx == 0:
        data = intra_data
    elif test_idx == 1:
        intra_data[intra_data<0]=np.nan
        cross_data[np.isnan(intra_data)]=np.nan
        data = cross_data
    elif test_idx == 2:
        intra_data[intra_data<0]=np.nan
        unified_data[np.isnan(intra_data)]=np.nan
        data = unified_data

    ## Fisher z transformation
    data = np.arctanh(data)

    ## combine into pandas dataframe
    df = get_index_4data(data)

    # make figure
    tmp_FIG_DIR=f"{FIG_DIR}{data_name}_{plot_name}/"
    if not os.path.isdir(tmp_FIG_DIR):
        os.makedirs(tmp_FIG_DIR)

    fig_name_subjects = f"{tmp_FIG_DIR}{data_name}_raw_{plot_name}_{model}_subjects.pdf"
    fig_name_layers = f"{tmp_FIG_DIR}{data_name}_raw_{plot_name}_{model}_layers.pdf"
    fig_name_grandmean = f"{tmp_FIG_DIR}{data_name}_raw_{plot_name}_{model}_grandmean.pdf"
    if plot_idx == 0:
        lineplot_each_subject(df, fig_name_subjects, xlabel, ylabel, ymin=ymin, ymax=ymax)
        lineplot_each_layer(df, fig_name_layers, xlabel, ylabel, ymin=ymin, ymax=ymax)
        lineplot_context_X_mean_subject_plus_layer(df, fig_name_grandmean, xlabel, ylabel, color, ymin=ymin, ymax=ymax)
    elif plot_idx == 1:
        boxplot_each_layer(df, fig_name_layers, xlabel, ylabel, color, ymin=ymin, ymax=ymax)



def plot_scores_all_actual_vs_cross_vs_unified(model, basemodel, unified_model, ymin=0, ymax=1):
    plot_name = 'lineplot'
    sig_name='raw'
    data_name = f"R_{sig_name}_score_actual_vs_cross_vs_uni"    
    intra_data, cross_data, unified_data, _, _, _ = get_summary_data(model, basemodel, unified_model, sig_idx=0)
    ## extract significantly predicted voxels (=> linguistic voxels)
    sig_intra_data, _, _, _, _, _ = get_summary_data(model, basemodel, unified_model, sig_idx=1)
    cross_data[np.isnan(sig_intra_data)] = np.nan

    ## Fisher z transformation
    intra_data = np.arctanh(intra_data)
    cross_data = np.arctanh(cross_data)
    unified_data = np.arctanh(unified_data)

    ## combine into pandas dataframe
    df_intra = get_index_4data(intra_data)
    df_cross = get_index_4data(cross_data)
    df_unified = get_index_4data(unified_data)
    df_intra['type'] = 'Actual'
    df_cross['type'] = 'Cross'
    df_unified['type'] = 'Unified'
    df = pd.concat([df_intra, df_cross, df_unified], ignore_index=True)

    # make figure
    tmp_FIG_DIR=f"{FIG_DIR}{plot_name}_{data_name}/"
    if not os.path.isdir(tmp_FIG_DIR):
        os.makedirs(tmp_FIG_DIR)

    fig_name=f"{tmp_FIG_DIR}{plot_name}_{data_name}_{basemodel}_{model}_variables.pdf"
    xlabel = 'Context lengths [s]'
    ylabel = 'Prediction performance (r)'
    color=[ '#624498', '#f8b500', '#008dbd' ]
    lineplot_context_X_mean_subject_plus_layer_variables(df, fig_name, xlabel, ylabel, color=color, ymin=ymin, ymax=ymax)




##########################################################################################################################################################################
##########################################################################################################################################################################
### Variance partitioning
##########################################################################################################################################################################
##########################################################################################################################################################################
### Fig. 3a, 4a * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
def plot_vp_context_X_partitions(model, basemodel, unified_model, plot_idx, ymin=0, ymax=50, sig_idx=1):
    if plot_idx == 0:
        plot_name = 'lineplot'
    elif plot_idx == 1:
        plot_name = 'boxplot'

    xlabel = 'Context lengths [s]'
    ylabel = 'Variance explained (r)'
    color=[ '#e84118', '#ffd32a', '#273c75']
    intra_data, cross_data, unified_data, prod_data, comp_data, intersection_data = get_summary_data(model, basemodel, unified_model, sig_idx=sig_idx)
    if sig_idx == 1:
        nsvoxels = np.isnan(intra_data)
        prod_data[nsvoxels] = np.nan
        comp_data[nsvoxels] = np.nan
        intersection_data[nsvoxels] = np.nan

    sig_name = get_plot_parameters(sig_idx=sig_idx)

    ## Fisher z transformation
    data_name = f"Z_RC_{sig_name}"
    prod_data = np.arctanh(prod_data, z_idx)
    comp_data = np.arctanh(comp_data, z_idx)
    intersection_data = np.arctanh(intersection_data, z_idx)

    ## combine into pandas dataframe
    df_prod = get_index_4data(prod_data)
    df_comp = get_index_4data(comp_data)
    df_is = get_index_4data(intersection_data)
    df_prod['type'] = 'Production'
    df_comp['type'] = 'Comprehension'
    df_is['type'] = 'Intersection'
    merged_df = pd.concat([df_prod, df_comp, df_is], ignore_index=True)
    print(merged_df)

    # make figure
    tmp_FIG_DIR=f"{FIG_DIR}{data_name}_{plot_name}/"
    if not os.path.isdir(tmp_FIG_DIR):
        os.makedirs(tmp_FIG_DIR)

    fig_name_variables=f"{tmp_FIG_DIR}{plot_name}_{data_name}_{basemodel}_{model}_variables.pdf"
    if plot_idx == 0:
        lineplot_context_X_mean_subject_plus_layer_variables(merged_df, fig_name_variables, xlabel, ylabel, color=color, ymin=ymin, ymax=ymax)
    elif plot_idx == 1:
        boxplot_meanXlayers_contexts_variables(subject_df, fig_name, xlabel, ylabel, color=color, ymin=ymin, ymax=ymax)


### ***** Supplementary Fig-9 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
def plot_vp_group_layers_contexts(basemodel, model, unified_model, plot_idx=0, modality_idx=0, ymin=0, ymax=0.06, sig_idx=0):
    if plot_idx == 0:
        plot_name = 'lineplot'
    elif plot_idx == 1:
        plot_name = 'boxplot'

    if sig_idx == 0:
        sig_name='raw'
    elif sig_idx == 1:
        sig_name='sig'

    data, name, color = get_vp_data(basemodel, model, unified_model, modality_idx=modality_idx, sig_idx=sig_idx)
    data = np.arctanh(data)
    df = get_index_4data(data)
    data_name = f"Z_RC_{name}"

    ## make figure
    tmp_FIG_DIR=f"{FIG_DIR}{data_name}_{plot_name}/"
    if not os.path.isdir(tmp_FIG_DIR):
        os.makedirs(tmp_FIG_DIR)

    fig_name_subjects = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_subjects.pdf"
    fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
    fig_name_grandmean = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_variables.pdf"
    xlabel = 'Context lengths [s]'
    ylabel = 'Mean performance (r)'
    if plot_idx == 0:
        lineplot_each_subject(df, fig_name_subjects, xlabel, ylabel, ymin=ymin, ymax=ymax)
        lineplot_each_layer(df, fig_name_layers, xlabel, ylabel, ymin=ymin, ymax=ymax)
        color="#4cd137"
        lineplot_context_X_mean_subject_plus_layer(df, fig_name_grandmean, xlabel, ylabel, color=color, ymin=ymin, ymax=ymax)
    elif plot_idx == 1:
        boxplot_each_layer(df, fig_name_layers, xlabel, ylabel, color, ymin=ymin, ymax=ymax)



##########################################################################################################################################################################
##########################################################################################################################################################################
### Weight correlation
##########################################################################################################################################################################
##########################################################################################################################################################################
### Fig. 4c and Supplementary Fig. 12 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
def plot_weightcorr_context_X_layers_vp_RGB(basemodel, model, plot_idx, ymin=0, ymax=0.4, sig_idx=0):
    if plot_idx == 0:
        plot_name = 'lineplot'
    elif plot_idx == 1:
        plot_name = 'boxplot'

    if sig_idx == 0:
        sig_name = "raw"
    elif sig_idx == 1:
        sig_name = "sig"

    vp_RGB_file = f"{RESULTS_DIR}/ALLSUBJ_best_vp_{basemodel}_{model}.hdf"
    with tables.open_file(vp_RGB_file, "r") as VP_f:
        VRGB_all=VP_f.root.VRGB_all.read()

    wc_whole = get_wc_data(basemodel, model)
    wc_prod = copy.deepcopy(wc_whole)
    wc_comp = copy.deepcopy(wc_whole)
    wc_intersection = copy.deepcopy(wc_whole)
    wc_prod[VRGB_all==2] = np.nan
    wc_prod[VRGB_all==3] = np.nan
    wc_comp[VRGB_all==1] = np.nan
    wc_comp[VRGB_all==3] = np.nan
    wc_intersection[VRGB_all==1] = np.nan
    wc_intersection[VRGB_all==2] = np.nan

    ### Fisher z transformation
    wc_prod = np.arctanh(wc_prod, z_idx)
    wc_comp = np.arctanh(wc_comp, z_idx)
    wc_intersection = np.arctanh(wc_intersection, z_idx)

    ### combine into pandas dataframe
    df_prod = get_index_4data(wc_prod)
    df_comp = get_index_4data(wc_comp)
    df_intersection = get_index_4data(wc_intersection)
    df_prod['type'] = 'Production'
    df_comp['type'] = 'Comprehension'
    df_intersection['type'] = 'Intersection'
    df = pd.concat([df_prod, df_comp, df_intersection], ignore_index=True)

    ## make figure
    data_name = 'Z_WC_vp_all'
    xlabel = 'Context lengths [s]'
    ylabel = 'Mean weight correlation (r)'
    tmp_FIG_DIR=f"{FIG_DIR}{data_name}_{plot_name}/"
    if not os.path.isdir(tmp_FIG_DIR):
        os.makedirs(tmp_FIG_DIR)

    if plot_idx == 0:
        ### Production
        fig_name_subjects = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_subjects_prod.pdf"
        fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers_prod.pdf"
        lineplot_each_subject(df_prod, fig_name_subjects, xlabel, ylabel, ymin=ymin, ymax=ymax)
        lineplot_each_layer(df_prod, fig_name_layers, xlabel, ylabel, ymin=ymin, ymax=ymax)

        ### Comprehension
        fig_name_subjects = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_subjects_comp.pdf"
        fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers_comp.pdf"
        lineplot_each_subject(df_comp, fig_name_subjects, xlabel, ylabel, ymin=ymin, ymax=ymax)
        lineplot_each_layer(df_comp, fig_name_layers, xlabel, ylabel, ymin=ymin, ymax=ymax)

        ### Intersection
        fig_name_subjects = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_subjects_intersection.pdf"
        fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers_intersection.pdf"
        lineplot_each_subject(df_intersection, fig_name_subjects, xlabel, ylabel, ymin=ymin, ymax=ymax)
        lineplot_each_layer(df_intersection, fig_name_layers, xlabel, ylabel, ymin=ymin, ymax=ymax)

        ## comp, is, prod
        color=["#FF3616", "#00B81B",  "#192A9A" ]
        fig_name_grandmean = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_variables.pdf"
        lineplot_context_X_mean_subject_plus_layer_variables(df, fig_name_grandmean, xlabel, ylabel, color, ymin=-0.13, ymax=0.13)
    elif plot_idx == 1:
        data_name = 'Z_WC_vp_prod'
        color = mcolors.LinearSegmentedColormap.from_list("orig", ["white", "#192A9A"]) 
        fig_name_layers=fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
        boxplot_each_layer(df_prod, fig_name_layers, xlabel, ylabel, color, ymin=ymin, ymax=ymax)

        data_name = 'Z_WC_vp_comp'
        color = mcolors.LinearSegmentedColormap.from_list("orig", ["white", "#FF3616"]) 
        fig_name_layers=fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
        boxplot_each_layer(df_comp, fig_name_layers, xlabel, ylabel, color, ymin=ymin, ymax=ymax)

        data_name = 'Z_WC_vp_is'
        color = mcolors.LinearSegmentedColormap.from_list("orig", ["white", "#00B81B"]) 
        fig_name_layers=fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
        boxplot_each_layer(df_intersection, fig_name_layers, xlabel, ylabel, color, ymin=ymin, ymax=ymax)


### Fig. 2d and Supplementary Fig. 10 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
def plot_weightcorr_context_X_layers_intra_cross_svoxels(basemodel, model, unified_model, plot_idx, ymin=0, ymax=0.4, sig_idx=0, z_idx=1):
    if sig_idx == 0:
        sig_name = "raw"
    elif sig_idx == 1:
        sig_name = "sig"

    if plot_idx == 0:
        plot_name = 'lineplot'
    elif plot_idx == 1:
        plot_name = 'boxplot'
    
    xlabel = 'Context lengths [s]'
    ylabel = 'Mean weight correlation (r)'
    tmp_FIG_DIR=f"{FIG_DIR}WC_intra_cross_svoxels_{plot_name}/"
    if not os.path.isdir(tmp_FIG_DIR):
        os.makedirs(tmp_FIG_DIR)

    intra_data, cross_data, unified_data, _, _, _ = get_summary_data(model, basemodel, unified_model, sig_idx=1)    
    wc_whole = get_wc_data(basemodel, model)

    ## Cross-modal voxels (intramodal prediction accuracy > 0.05 and cross-modality prediction accuracy > 0.05)
    thr = 0.05
    wc_whole[np.isnan(intra_data)] = np.nan
    wc_whole[intra_data<thr] = np.nan
    wc_intra = copy.deepcopy(wc_whole)

    wc_whole[np.isnan(cross_data)] = np.nan
    wc_whole[cross_data<thr] = np.nan
    wc_cross = copy.deepcopy(wc_whole)

    ## combine into pandas dataframe
    df_intra = get_index_4data(wc_intra)
    df_cross = get_index_4data(wc_cross)
    df_intra['type'] = 'Actual'
    df_cross['type'] = 'Cross'
    df = pd.concat([df_intra, df_cross], ignore_index=True)
    if plot_idx == 0:
        ### intramodal (linguistic voxels)
        data_name = 'WC_intra'
        data_name = f"Z{data_name}"
        fig_name_subjects = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_subjects.pdf"
        fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
        lineplot_each_subject(df_intra, fig_name_subjects, xlabel, ylabel, ymin=ymin, ymax=ymax)
        lineplot_each_layer(df_intra, fig_name_layers, xlabel, ylabel, ymin=ymin, ymax=ymax)
        
        ### cross-modal voxels
        data_name = 'WC_cross_svoxels'
        data_name = f"Z{data_name}"
        fig_name_subjects = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_subjects.pdf"
        fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
        lineplot_each_subject(df_cross, fig_name_subjects, xlabel, ylabel, ymin=ymin, ymax=ymax)
        lineplot_each_layer(df_cross, fig_name_layers, xlabel, ylabel, ymin=ymin, ymax=ymax)

        ### both
        data_name = 'WC_intra_cross_svoxels'
        data_name = f"Z{data_name}"
        fig_name_grandmean = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_variables.pdf"
        color=[ "#624498", "#f8b500" ]
        lineplot_context_X_mean_subject_plus_layer_variables(df, fig_name_grandmean, xlabel, ylabel, color, ymin=ymin, ymax=ymax)
    elif plot_idx == 1:
        ### intramodal (linguistic voxels)
        data_name = 'WC_intra'
        color = mcolors.LinearSegmentedColormap.from_list("orig", ["white", "#624498"])
        data_name = f"Z{data_name}"
        fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
        boxplot_each_layer(df_intra, fig_name_layers, xlabel, ylabel, color, ymin=ymin, ymax=ymax)

        ### cross-modal voxels
        data_name = 'WC_cross_svoxels'
        color = mcolors.LinearSegmentedColormap.from_list("orig", ["white", "#f8b500"])
        data_name = f"Z{data_name}"
        fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
        boxplot_each_layer(df_cross, fig_name_layers, xlabel, ylabel, color, ymin=ymin, ymax=ymax)

### Fig. 3c and Supplementary Fig. 11 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
def plot_weightcorr_context_X_layers_unified(basemodel, model, unified_model, plot_idx=0, ymin=0, ymax=0.25, sig_idx=0):
    if plot_idx == 0:
        plot_name = 'lineplot'
    elif plot_idx == 1:
        plot_name = 'boxplot'

    if sig_idx == 0:
        sig_name = "raw"
    elif sig_idx == 1:
        sig_name = "sig"
    
    tmp_FIG_DIR=f"{FIG_DIR}{data_name}_{plot_name}/"
    if not os.path.isdir(tmp_FIG_DIR):
        os.makedirs(tmp_FIG_DIR)

    prod_data = get_wc_data_unified(basemodel, unified_model, modality_idx=0)
    comp_data = get_wc_data_unified(basemodel, unified_model, modality_idx=1)
    if sig_idx == 1:
        intra_data, cross_data, unified_data, _, _, _ = get_summary_data(model, basemodel, unified_model, sig_idx=sig_idx)
        nsvoxels = np.isnan(intra_data)
        prod_data[nsvoxels] = np.nan
        comp_data[nsvoxels] = np.nan
        nsvoxels = np.isnan(unified_data)
        prod_data[nsvoxels] = np.nan
        comp_data[nsvoxels] = np.nan

    ## Fisher z transformation
    prod_data = np.arctanh(prod_data, z_idx)
    comp_data = np.arctanh(comp_data, z_idx)

    ## combine into pandas dataframe
    df_1 = get_index_4data(prod_data)
    df_2 = get_index_4data(comp_data)
    df_1['type'] = 'Production'
    df_2['type'] = 'Comprehension'
    df = pd.concat([df_1, df_2], ignore_index=True)

    ## make figure
    xlabel = 'Context lengths [s]'
    ylabel = 'Mean weight correlation (r)'
    if plot_idx == 0:
        ### each 
        data_name = 'Z_WC_prod'
        fig_name_subjects = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_subjects.pdf"
        fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
        lineplot_each_subject(df_1, fig_name_subjects, xlabel, ylabel, ymin=ymin, ymax=ymax)
        lineplot_each_layer(df_1, fig_name_layers, xlabel, ylabel, ymin=ymin, ymax=ymax)
        
        data_name = 'Z_WC_comp'
        fig_name_subjects = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_subjects.pdf"
        fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
        lineplot_each_subject(df_2, fig_name_subjects, xlabel, ylabel, ymin=ymin, ymax=ymax)
        lineplot_each_layer(df_2, fig_name_layers, xlabel, ylabel, ymin=ymin, ymax=ymax)

        ### both
        data_name = 'Z_WC_separate_vs_unified'
        color=[ '#d3381c', '#1e50a2' ]
        fig_name_grandmean = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}.pdf"
        lineplot_context_X_mean_subject_plus_layer_variables(df, fig_name_grandmean, xlabel, ylabel, color, ymin=ymin, ymax=ymax)
    elif plot_idx == 1:
        ### each 
        data_name = 'Z_WC_prod'
        fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
        color = mcolors.LinearSegmentedColormap.from_list("orig", ["white", "#1e50a2"])
        boxplot_each_layer(df_1, fig_name_layers, xlabel, ylabel, color, ymin=ymin, ymax=ymax)

        data_name = 'Z_WC_comp'
        fig_name_layers = f"{tmp_FIG_DIR}{data_name}_{sig_name}_{plot_name}_{model}_layers.pdf"
        color = mcolors.LinearSegmentedColormap.from_list("orig", ["white", "#d3381c"])
        boxplot_each_layer(df_2, fig_name_layers, xlabel, ylabel, color, ymin=ymin, ymax=ymax)



##########################################################################################################################################################################
##########################################################################################################################################################################
### lineplot
##########################################################################################################################################################################
##########################################################################################################################################################################
def lineplot_each_subject(df, fig_name, xlabel, ylabel, ymin=0, ymax=0.1):
    figsize=5.5
    figbottom=7
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(facecolor='white', figsize = (figsize*CM, figbottom*CM))
    err_kws = {'linewidth': LINEWIDTH}
    sns.lineplot(data=df, x='cl', y='value', linewidth=LINEWIDTH, hue='subject', palette='Set1', err_style="bars", err_kws={"elinewidth": LINEWIDTH})
    ax.axhline(y=0, color='k',  linestyle="--", linewidth=0.2)
    ax.set_ylim(ymin, ymax)
    ax.legend_.remove()
    ax.set_xticks(CONTEXT_LENGTH)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    print(f"saving {fig_name}")
    plt.savefig(fig_name, format='pdf', bbox_inches='tight')
    plt.close()

def lineplot_each_layer(df, fig_name, xlabel, ylabel, ymin=0, ymax=0.1):
    figsize=5.5
    figbottom=7
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(facecolor='white', figsize = (figsize*cm, figbottom*cm))
    err_kws = {'linewidth': LINEWIDTH}
    sns.lineplot(data=df, x='cl', y='value', linewidth=LINEWIDTH, hue='layer', palette="viridis", err_style="bars", err_kws={"elinewidth": LINEWIDTH})
    ax.axhline(y=0, color='k', linestyle="--", linewidth=0.2)
    ax.set_ylim(ymin, ymax)
    ax.legend_.remove()
    ax.set_xticks(CONTEXT_LENGTH)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    print(f"saving {fig_name}")
    plt.savefig(fig_name, format='pdf', bbox_inches='tight')
    plt.close()

def lineplot_context_X_mean_subject_plus_layer(df, fig_name, xlabel, ylabel, color, ymin=0, ymax=0.1):
    figsize=5.5
    figbottom=7
    average_df = df.groupby(['cl', 'subject', 'layer'])['value'].mean().reset_index()
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(facecolor='white', figsize = (figsize*cm, figbottom*cm))
    err_kws = {'linewidth': linewidth}
    sns.lineplot(data=average_df, x='cl', y='value', linewidth=linewidth, color=color, errorbar=('sd'))
    ax.axhline(y=0, color='k', linestyle="--", linewidth=0.2)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(CONTEXT_LENGTH)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    print(f"saving {fig_name}")
    plt.savefig(fig_name, format='pdf', bbox_inches='tight')
    plt.close()

def lineplot_context_X_mean_subject_plus_layer_variables(df, fig_name, xlabel, ylabel, color, ymin=0, ymax=0.1):
    average_df = df.groupby(['cl', 'subject', 'layer', 'type'])['value'].mean().reset_index()
    # print(average_df)
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(facecolor='white', figsize = (figsize*cm, figbottom*cm))
    err_kws = {'linewidth': linewidth}
    sns.lineplot(data=average_df, x='cl', y='value', hue="type", palette=color, linewidth=linewidth, errorbar=("sd"))
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.2)
    ax.set_ylim(ymin, ymax)
    ax.legend_.remove()
    ax.set_xticks(CONTEXT_LENGTH)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    print(f"saving {fig_name}")
    plt.tight_layout()
    plt.savefig(fig_name, format='pdf', bbox_inches='tight')
    plt.close()

def lineplot_meanXlayers_contexts_variables(df, fig_name, xlabel, ylabel, color, ymin=0, ymax=50):
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(facecolor='white', figsize = (figsize*cm, figbottom*cm))
    err_kws = {'linewidth': LINEWIDTH}
    sns.lineplot(data=df, x='cl', y='value', hue='type', palette=color, linewidth=LINEWIDTH, err_style="bars", err_kws={"elinewidth": LINEWIDTH})
    # ax.axhline(y=0, color='k', linewidth=1)
    ax.set_ylim(ymin, ymax)
    # ax.legend_.remove()
    ax.set_xticks(CONTEXT_LENGTH)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    print(f"saving {fig_name}")
    # plt.tight_layout()
    plt.savefig(fig_name, format='pdf', bbox_inches='tight')
    plt.close()



##########################################################################################################################################################################
##########################################################################################################################################################################
### boxplot
##########################################################################################################################################################################
##########################################################################################################################################################################
def boxplot_each_layer(df, fig_name, xlabel, ylabel, color, ymin=0, ymax=0.1):
    plt.rcParams['font.family'] = 'Arial'
    figsize=5.5
    figbottom=7
    fig, ax = plt.subplots(facecolor='white', figsize = (figsize*cm, figbottom*cm))
    err_kws = {'linewidth': 0.2}
    sns.boxplot(data=df, x='cl', y='value', hue='layer', palette=color, fliersize=0, linewidth=0.2, medianprops={"color": "r", "linewidth": 0.3})
    sns.stripplot(data=df, x='cl', y='value', hue='layer', palette="viridis", dodge=True, size=1)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(y=0, color='k', linewidth=1)
    plt.legend().remove()
    plt.savefig(fig_name, format='pdf', bbox_inches='tight')
    plt.close()

def boxplot_meanXlayers_contexts_variables(df, fig_name, xlabel, ylabel, color, ymin=0, ymax=50):
    plt.rcParams['font.family'] = 'Arial'
    figsize=6.5
    figbottom=8
    fig, ax = plt.subplots(facecolor='white', figsize = (figsize*cm, figbottom*cm))
    err_kws = {'linewidth': 1}
    sns.boxplot(data=df, x='cl', y='value', hue='type', palette=color, fliersize=0, linewidth=0.2, medianprops={"color": "k", "linewidth": 0.3})
    sns.stripplot(data=df, x='cl', y='value', hue='type', palette=['black']*8, dodge=True, size=1)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    print(f"saving {fig_name}")
    plt.savefig(fig_name, format='pdf', bbox_inches='tight')
    plt.close()



##########################################################################################################################################################################
##########################################################################################################################################################################
### others
##########################################################################################################################################################################
##########################################################################################################################################################################
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
        unified_data=MODEL_f.root.scores_all.read()

    return intra_data, cross_data, unified_data, prod_data, comp_data, intersection_data

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
    df = df.groupby(['layer', 'cl', 'subject'])['value'].mean().reset_index()
    return df

def get_plot_parameters(sig_idx=0):
    if sig_idx == 0:
        sig_name='raw'
    elif sig_idx == 1:
        sig_name='sig'  

    return sig_name

def get_vp_data(basemodel, model, unified_model, modality_idx, sig_idx):
    intra_data, cross_data, unified_data, prod_data, comp_data, intersection_data = get_summary_data(model, basemodel, unified_model, sig_idx=sig_idx)
    if sig_idx == 0:
        sig_name='raw'
    elif sig_idx == 1:
        sig_name='sig'

    if modality_idx == 0:
        name = 'prod'
        data = prod_data
        color = mcolors.LinearSegmentedColormap.from_list("orig", ["white", "#192A9A"]) 
    elif modality_idx == 1:
        name = 'comp'
        data = comp_data
        color = mcolors.LinearSegmentedColormap.from_list("orig", ["white", "#FF3616"]) 
    elif modality_idx == 2:
        name = 'intersection'
        data = intersection_data
        color = "Greens"
        color = mcolors.LinearSegmentedColormap.from_list("orig", ["white", "#00B81B"]) 
            
    return data, name, color

def get_wc_data(basemodel, model, sigwc_idx=0):
    weightcorr_data=f"{RESULTS_DIR}/ALLSUBJ_weightcorrs_{basemodel}_{model}_tvoxels.hdf"
    with tables.open_file(weightcorr_data, "r") as WCORR_f:
        if sigwc_idx == 0:
            data=WCORR_f.root.weightcorr_all.read()
        elif sigwc_idx == 1:
            data=WCORR_f.root.sig_weightcorr_all.read()

    return data

def get_wc_data_unified(basemodel, unified_model, modality_idx, sigwc_idx=0):
    weightcorr_data=f"{RESULTS_DIR}/ALLSUBJ_weightcorrs_{basemodel}_{unified_model}_tvoxels.hdf"
    with tables.open_file(weightcorr_data, "r") as WCORR_f:
        if sigwc_idx == 0:
            if modality_idx == 0:
                data=WCORR_f.root.prodcorr_all.read()
            elif modality_idx == 1:
                data=WCORR_f.root.compcorr_all.read()

        elif sigwc_idx == 1:
            if modality_idx == 0:
                data=WCORR_f.root.sig_prodcorr_all.read()
            elif modality_idx == 1:
                data=WCORR_f.root.sig_compcorr_all.read()

    return data
