import copy
import numpy as np
import os
import tables
from scipy.stats import zscore, pearsonr
from sklearn.decomposition import PCA
from dialogue.io import get_ready_indiv_pcafile, get_features_ses, save_table_file, save_strings_hdf_file, load_strings_hdf_array
from dialogue.config import TOTAL_SESS, SUBJECTS_ALL, FEATURES_DIR, RESULTS_DIR, N_PC, N_TOP_VOXELS, LAYERS, CONTEXT_LENGTH, N_PERM


#######################################################################################################################################################################################
#######################################################################################################################################################################################
### pca_weight_wrapper.py
#######################################################################################################################################################################################
#######################################################################################################################################################################################
def indiv_pca(subject_idx, model, basemodel, modality_idx=0, layer=-1, cl=1):
    indiv_weight_prod_file, indiv_weight_comp_file, pca_prod_file, pca_comp_file, _, _, _, _ = get_ready_indiv_pcafile(SUBJECTS_ALL[subject_idx], model, basemodel, layer=layer, cl=cl, n_top_voxels=None)
    if modality_idx == 0:
        weight_file = indiv_weight_prod_file
        pca_file = pca_prod_file
    elif modality_idx == 1:
        weight_file = indiv_weight_comp_file
        pca_file = pca_comp_file

    if os.path.exists(weight_file):
        with tables.open_file(weight_file, "r") as WEIGHT_f:
            zw = np.nan_to_num(WEIGHT_f.root.zw.read())
    else:
        w_prod, w_comp = get_average_weights(subject_idx, model, basemodel, layer=layer, cl=cl)
        if modality_idx == 0:
            print(f"w_prod: {w_prod.shape}")
            w = w_prod
            weight_file = indiv_weight_prod_file
        elif modality_idx == 1:
            print(f"w_comp: {w_comp.shape}")
            w = w_comp
            weight_file = indiv_weight_comp_file

        zw = zscore(w, axis=1)
        save_table_file(weight_file, dict(zw=zw))
        
    print(f"zw: {zw.shape}")
    assert check_nan_inf(zw), "zw contains NaN or Inf"

    ### group PCA
    pca = PCA(n_components=N_PC)
    pca.fit(zw.T)
    pc_var_explained = pca.explained_variance_ratio_
    pc_coefs = pca.components_
    pc_scores = pca.fit_transform(zw.T).T
    print("PC coef: (n_components, n_features) = ", pc_coefs.shape)
    print("PC scores: (n_components, n_voxels) = ", pc_scores.shape)
    print("PCA explained variance = ", pca.explained_variance_ratio_)
    print(f"saving pca_file: {pca_file}")
    save_table_file(pca_file, dict(pc_coefs=pc_coefs, pc_scores=pc_scores, pc_var_explained=pc_var_explained))

def get_indiv_concatenated_stimlus(subject_idx, modality_idx=0, layer=-1, cl=1, llm='chatgptneox'):
    if modality_idx == 0:
        modality = 'prod'
    elif modality_idx == 1:
        modality = 'comp'
    
    postfix=f"layer-{layer}_cl-{cl}"
    utterance_file = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_utterance_{postfix}_{modality}.hdf"
    embedding_file = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_embedding_{llm}_{postfix}_{modality}.hdf"
    if not os.path.exists(embedding_file):
        print(f"creating {embedding_file}")
        ### 1 Load utterance
        indiv_utt, unique_utterance_idx = get_utterance(subject_idx, modality, cl)
        ### 2 Loading utterance embedding (utterance x LLM hidden units)
        indiv_gpt = get_utterance_embeddings(subject_idx, modality, layer, cl, unique_utterance_idx, llm)
        save_strings_hdf_file(utterance_file, indiv_utt)
        save_table_file(embedding_file, dict(indiv_gpt=indiv_gpt, unique_utterance_idx=unique_utterance_idx))
    else:
        indiv_utt = load_strings_hdf_array(utterance_file)
        with tables.open_file(embedding_file, "r") as f:
            indiv_gpt = f.root.indiv_gpt.read()
            unique_utterance_idx = f.root.unique_utterance_idx.read()

    return indiv_utt, indiv_gpt, unique_utterance_idx

def get_indiv_pc_loadings(subject_idx, model, basemodel, modality_idx=0, layer=-1, cl=1, n_top_voxels=N_TOP_VOXELS):
    if modality_idx == 0:
        modality = 'prod'
    elif modality_idx == 1:
        modality = 'comp'
    
    postfix=f"layer-{layer}_cl-{cl}"
    indiv_utt, indiv_gpt, unique_utterance_idx = get_indiv_concatenated_stimlus(subject_idx, modality_idx=modality_idx, layer=layer, cl=cl)
    ### Load PCA results
    indiv_weight_prod_file, indiv_weight_comp_file, pca_prod_file, pca_comp_file, _, _, pcloads_prod_file, pcloads_comp_file = get_ready_indiv_pcafile(SUBJECTS_ALL[subject_idx], model, basemodel, layer=layer, cl=cl, n_top_voxels=n_top_voxels)
    if modality_idx == 0:
        indiv_weight_file = indiv_weight_prod_file
        pca_file = pca_prod_file
        pcloads_file = pcloads_prod_file
    elif modality_idx == 1:
        indiv_weight_file = indiv_weight_comp_file
        pca_file = pca_comp_file
        pcloads_file = pcloads_comp_file

    with tables.open_file(pca_file, "r") as PCA_f:
        pc_coefs = PCA_f.root.pc_coefs.read()

    ### PC loadings by correlating PC coeff with embeddings
    n_target_pc = 5
    pc_loads = np.full((pc_coefs.shape[0], indiv_gpt.shape[0]), np.nan)
    for ii in range(n_target_pc):
        for jj in range(indiv_gpt.shape[0]):
            pc_loads[ii,jj], _ = pearsonr(pc_coefs[ii,:].T, indiv_gpt[jj,:].T)

    scaled_pc_loads = zscore(pc_loads, axis=0, nan_policy='omit')
    print(f"saving {pcloads_file}")
    save_table_file(pcloads_file, dict(pc_loads=pc_loads, scaled_pc_loads=scaled_pc_loads))

def get_indiv_pcloads_txt(subject_idx, model, basemodel, modality_idx=0, layer=-1, cl=1, n_top_voxels=N_TOP_VOXELS):
    indiv_utt, indiv_gpt, unique_utterance_idx = get_indiv_concatenated_stimlus(subject_idx, modality_idx=modality_idx, layer=layer, cl=cl)
    ### Load PCA results
    indiv_weight_prod_file, indiv_weight_comp_file, pca_prod_file, pca_comp_file, _, _, pcloads_prod_file, pcloads_comp_file = get_ready_indiv_pcafile(SUBJECTS_ALL[subject_idx], model, basemodel, layer=layer, cl=cl, n_top_voxels=n_top_voxels)
    if modality_idx == 0:
        modality = 'prod'
        indiv_weight_file = indiv_weight_prod_file
        pca_file = pca_prod_file
        pcload_file = pcloads_prod_file
    elif modality_idx == 1:
        modality = 'comp'
        indiv_weight_file = indiv_weight_comp_file
        pca_file = pca_comp_file
        pcload_file = pcloads_comp_file
    
    postfix=f"layer-{layer}_cl-{cl}"
    with tables.open_file(pcload_file, "r") as PCL_f:
        pc_loads = PCL_f.root.pc_loads.read()

    with tables.open_file(pca_file, "r") as PCA_f:
        pc_coefs = PCA_f.root.pc_coefs.read()

    ### PC loadings by correlating PC coeff with embeddings
    print(f"pc_coefs: {pc_coefs.shape}")
    print(f"indiv_gpt: {indiv_gpt.shape}")
    n_target_pc = 5
    pc_loads = np.full((pc_coefs.shape[0], indiv_gpt.shape[0]), np.nan)
    for ii in range(n_target_pc):
        for jj in range(indiv_gpt.shape[0]):
            pc_loads[ii,jj], _ = pearsonr(pc_coefs[ii,:].T, indiv_gpt[jj,:].T)

    print(f"pc_loadings: {pc_loads.shape}")
    save_table_file(pcload_file, dict(pc_loads=pc_loads))
    ### Extract most correlative utterances for each PC
    n_extract = 50
    for pc_i in range(n_target_pc):
        print(f"****** PC {pc_i+1}:")
        d_idx = np.argsort(-pc_loads[pc_i, :])
        sorted_pc_loads_high = np.sort(-pc_loads[pc_i,:])
        sorted_indiv_utterance_high = indiv_utt[d_idx]
        print(f"sorted_pc_loads_high: {sorted_pc_loads_high.shape}")
        print(f"sorted_pc_loads_high: {-sorted_pc_loads_high}")
        print(f"sorted_indiv_utterance_high: {sorted_indiv_utterance_high.shape}")
        print(f"sorted_indiv_utterance_high: {sorted_indiv_utterance_high}")
        print(f"indiv_utt on top: {indiv_utt[d_idx[0]]}")
        if not n_top_voxels == None:
            txtfile = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_pcloads_{basemodel}_{model}_{postfix}_{modality}_top{n_top_voxels}_high_PC{pc_i+1}.txt"
        else:
            txtfile = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_pcloads_{basemodel}_{model}_{postfix}_{modality}_high_PC{pc_i+1}.txt"

        for jj in range(n_extract):
            with open(txtfile, "a") as f:
                f.write(f"{jj+1}\t{-sorted_pc_loads_high[jj]}\t{sorted_indiv_utterance_high[jj]}\n")

        a_idx = np.argsort(pc_loads[pc_i, :])
        sorted_pc_loads_low = np.sort(pc_loads[pc_i,:])
        sorted_indiv_utterance_low = indiv_utt[a_idx]
        print(f"sorted_pc_loads_low: {sorted_pc_loads_low.shape}")
        print(f"sorted_pc_loads_low: {sorted_pc_loads_low}")
        print(f"sorted_indiv_utterance_low: {sorted_indiv_utterance_low.shape}")
        print(f"sorted_indiv_utterance_low: {sorted_indiv_utterance_low}")
        print(f"indiv_utt on top: {indiv_utt[d_idx[0]]}")
        if not n_top_voxels == None:
            txtfile = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_pcloads_{basemodel}_{model}_{postfix}_{modality}_top{n_top_voxels}_low_PC{pc_i+1}.txt"
        else:
            txtfile = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_pcloads_{basemodel}_{model}_{postfix}_{modality}_low_PC{pc_i+1}.txt"

        for jj in range(n_extract):
            with open(txtfile, "a") as f:
                f.write(f"{jj+1}\t{sorted_pc_loads_low[jj]}\t{sorted_indiv_utterance_low[jj]}\n")


def run_indiv_pca_bootstrap(subject_idx, model, basemodel, modality_idx=0, layer=-1, cl=1, n_top_voxels=None, llm='chatgptneox'):
    indiv_weight_prod_file, indiv_weight_comp_file, pca_prod_file, pca_comp_file, boot_prod_file, boot_comp_file, _, _ = get_ready_indiv_pcafile(SUBJECTS_ALL[subject_idx], model, basemodel, layer = layer, cl = cl, n_top_voxels=n_top_voxels)
    if modality_idx == 0:
        modality = 'prod'
        boot_file = boot_prod_file
    elif modality_idx == 1:
        modality = 'comp'
        boot_file = boot_comp_file

    postfix=f"layer-{layer}_cl-{cl}"

    ##### 1 Make/load stimuli concatenated across subjects
    if llm == 'chatgptneox':
        indiv_embedding_file = f"{FEATURES_DIR}raw/CHATGPTNEOX/UniqueEmbeddings_{SUBJECTS_ALL[subject_idx]}_concat_{modality}_{postfix}.hdf"
    elif llm == 'gptneox':
        indiv_embedding_file = f"{FEATURES_DIR}raw/GPTNEOX/UniqueEmbeddings_{SUBJECTS_ALL[subject_idx]}_concat_{modality}_{postfix}.hdf"

    if not os.path.exists(indiv_embedding_file):            
        ### 1 Load utterance
        utt, unique_utterance_idx = get_utterance(subject_idx, modality, cl)
        ### 2 Loading utterance embedding (utterance x LLM)
        X = get_utterance_embeddings(subject_idx, modality, layer, cl, unique_utterance_idx)
        ### 3 Normalize and concatenate across subjects
        zX_stim = np.copy(X)
        for ff in range(X.shape[0]):
             zX_stim[ff,:] = zscore(X[ff, :])

        print(f"saving: {indiv_embedding_file}")
        save_table_file(indiv_embedding_file, dict(zX_stim=zX_stim))
    else:
        print(f"loading: {indiv_embedding_file}")
        with tables.open_file(indiv_embedding_file, "r") as STIM_f:
            zX_stim = STIM_f.root.zX_stim.read()

    ##### 2 Load weights 
    if modality_idx == 0:
        weight_file = indiv_weight_prod_file
    elif modality_idx == 1:
        weight_file = indiv_weight_comp_file
        
    with tables.open_file(weight_file, "r") as WEIGHT_f:
        zX_weight = np.nan_to_num(WEIGHT_f.root.zw.read())
        zX_weight = zX_weight.T

    ##### 3 Compare %variance explained between stimulus and weight
    n1 = zX_weight.shape[0]
    n2 = zX_stim.shape[0]
    weight_coeff = np.full([N_PERM, N_PC, N_GPTSIZE], np.nan)
    explained_weight = np.full([N_PERM, N_PC], np.nan)
    stim_coeff = np.full([N_PERM, N_PC, N_GPTSIZE], np.nan)
    explained_stim = np.full([N_PERM, N_PC], np.nan)
    compare_explained = np.full([N_PERM, N_PC], np.nan)

    pca = PCA(n_components=N_PC)
    for rep in range(N_PERM):
        if rep%100==1:
            print(f"permutation : {rep}/{N_PERM}")

        tmp_zX_weight = zX_weight[np.random.choice(n1, n1, replace=True), :];
        tmpzX_stim = zX_stim[np.random.choice(n2, n2, replace=True), :]
        explained_weight[rep, :], weight_coeff[rep, :, :] = get_pccoeff(pca, tmp_zX_weight)
        explained_stim[rep, :], stim_coeff[rep, :, :] = get_pccoeff(pca, tmpzX_stim)

    for rep in range(N_PERM):
        if rep%100 == 1:
            print(f"***** Match and compare {rep}/{N_PERM}");
            
        absC=np.full((N_PC, N_PC), np.nan)
        tmp_a = np.squeeze(weight_coeff[rep,:,:])
        tmp_b = np.squeeze(stim_coeff[rep,:,:])
        for ii in range(tmp_a.shape[0]):
            for jj in range(tmp_b.shape[0]):
                corrtest = pearsonr(tmp_a[ii,:], tmp_b[jj,:])
                absC[ii,jj] = np.abs(corrtest[0])

        weight_pref = np.argsort(absC)[::-1]

        absC=np.full((N_PC,N_PC),np.nan)
        tmp_a = np.squeeze(stim_coeff[rep,:,:])
        tmp_b = np.squeeze(weight_coeff[rep,:,:])
        for ii in range(tmp_a.shape[0]):
            for jj in range(tmp_b.shape[0]):
                corrtest = pearsonr(tmp_a[ii,:], tmp_b[jj,:])
                absC[ii,jj] = np.abs(corrtest[0])

        stim_pref = np.argsort(absC)[::-1]
        stablematch = run_galeshapley(weight_pref, stim_pref)
        compare_explained[rep, :] = explained_weight[rep, :] - explained_stim[rep, stablematch]
        explained_stim[rep, :] = explained_stim[rep, stablematch]

    lose_times = np.full(N_PC, np.nan)
    for nn in range(N_PC):
        lose_times[nn] = np.sum(compare_explained[:, nn] < 0)

    pval = lose_times/N_PERM
    print(f"pval: {pval}")
    save_table_file(boot_file, dict(explained_weight=explained_weight, explained_stim=explained_stim, pval=pval))


def get_average_weights(subject_idx, model, basemodel, layer=-1, cl=1, score_idx=2):
    weights_list=[]
    for ses in range(1, TOTAL_SESS[subject_idx]+1):
        print(f"{SUBJECTS_ALL[subject_idx]}_{basemodel}_{model}_ses-{ses}_layer-{layer}_cl-{cl}")
        subject, primal_ses_im_file, primal_ses_cm_file, primal_r_im_file, primal_r_cm_file, primal_weights_file = get_ready_primal(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
        with tables.open_file(primal_ses_im_file, "r") as WEIGHT_f:
            weights_list.append(WEIGHT_f.root.average_weights.read())

    average_weights = np.nanmean(weights_list, axis=0)
    _, _, n_features_list = get_features_ses(subject_idx, 1, model, layer=layer, cl=cl)
    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices=[ slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:])]
    w_prod=average_weights[slices[-2], :]
    w_comp=average_weights[slices[-1], :]
    return w_prod, w_comp


def get_utterance(subject_idx, modality, cl, unique_idx=1):
    utter_file = f"{ANNT_DIR}/{SUBJECTS_ALL[subject_idx]}/{SUBJECTS_ALL[subject_idx]}_concat_{modality}_cl-{cl}.npy"
    utt = np.load(utter_file)
    if unique_idx == 1:
        print('*** Loading only unique utterances ')
        utt, unique_utt_idx = np.unique(utt, return_index=True)
    else:
        disp('*** Loading all the utterances ');

    return utt, unique_utt_idx


def get_utterance_embeddings(subject_idx, modality, layer, cl, unique_utt_idx, llm='chatgptneox'):
    if llm == 'chatgptneox':
        feature_name = "rinna/japanese-gpt-neox-3.6b-instruction-sft"
        features_path = f"{FEATURES_DIR}raw/CHATGPTNEOX/{SUBJECTS_ALL[subject_idx]}/{feature_name}/{SUBJECTS_ALL[subject_idx]}_concat_{modality}_cl-{cl}/"
        embedding_file = f"{features_path}/chatgptneox_avrg_layer_{layer}.npy"
        chatgptneox = np.load(embedding_file)
        unique_idx = 1
        if unique_idx == 1:
            print('*** Loading only unique utterances embeddings ')
            X = chatgptneox[unique_utt_idx, :]
    elif llm == 'gptneox':
        feature_name = "rinna/japanese-gpt-neox-3.6b"
        features_path = f"{FEATURES_DIR}/raw/GPTNEOX/{SUBJECTS_ALL[subject_idx]}/{feature_name}/{SUBJECTS_ALL[subject_idx]}_concat_{modality}_cl-{cl}/"
        embedding_file = f"{features_path}/gptneox_avrg_layer_{layer}.npy"
        X = np.load(embedding_file)
        unique_idx = 1
        if unique_idx == 1:
            print('*** Loading only unique utterances embeddings ')
            X = X[unique_utt_idx, :]

    return X


def run_galeshapley(brain_pref, stim_pref, n_components=N_PC):
    brain_free = np.zeros(n_components)
    stim_suitor = np.zeros((n_components, n_components))
    stim_partner = np.zeros(n_components)
    rank = np.zeros((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            for k in range(n_components):
                if stim_pref[i, k] == j + 1:
                    rank[i, j] = k + 1

    while min(stim_partner) == 0:
        for i in range(n_components):
            if brain_free[i] == 0:
                next_pref = np.argmax(brain_pref[i] > 0)
                stim_suitor[brain_pref[i, next_pref] - 1, i] = i + 1
                brain_pref[i, next_pref] = 0

        for i in range(n_components):
            for j in range(n_components):
                if stim_suitor[i, j] != 0:
                    if stim_partner[i] == 0:
                        stim_partner[i] = stim_suitor[i, j]
                        brain_free[j] = 1
                    if stim_partner[i] != 0:
                        if rank[i, int(stim_suitor[i, j]) - 1] < rank[i, int(stim_partner[i]) - 1]:
                            brain_free[int(stim_partner[i]) - 1] = 0
                            stim_partner[i] = stim_suitor[i, j]
                            brain_free[j] = 1

    stablematch = stim_partner.astype(int) - 1
    return stablematch



##########################################################################################################################################################################
##########################################################################################################################################################################
### summary
##########################################################################################################################################################################
##########################################################################################################################################################################
def summary_indiv_pca_expvar(basemodel, model_prefix):
    for modality_idx in [0, 1]:
        if modality_idx == 0:
            modality = 'prod'
        elif modality_idx == 1:
            modality = 'comp'

        pc_var_explained=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), N_PC), np.nan)
        for cl_idx, cl in enumerate(CONTEXT_LENGTH):
            for layer_idx, layer in enumerate(LAYERS):
                model = f"{model_prefix}_{cl}"
                for subject_idx in range(len(SUBJECTS_ALL)):
                    subject = SUBJECTS_ALL[subject_idx]
                    _, _, pca_prod_file, pca_comp_file, _, _, _, _ = get_ready_indiv_pcafile(subject, model, basemodel, layer, cl)
                    if modality_idx == 0:
                        pca_file = pca_prod_file
                    elif modality_idx == 1:
                        pca_file = pca_comp_file
               
                    with tables.open_file(pca_file, "r") as PCA_f:
                        pc_var_explained[layer_idx, cl_idx, subject_idx, :] = PCA_f.root.pc_var_explained.read()

        output = f"{RESULTS_DIR}ALLSUBJ_indiv_pca_variance_explained_{basemodel}_{model}_{modality}.hdf"
        save_table_file(output, dict(pc_var_explained=pc_var_explained))


def summary_indiv_pc_scores(basemodel, model_prefix):
    for modality_idx in [0, 1]:
        if modality_idx == 0:
            modality = 'prod'
        elif modality_idx == 1:
            modality = 'comp'

        n_store_pc=6
        pc_scores_all=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), n_store_pc, NVOXELS.max()), np.nan)
        for subject_idx in range(len(SUBJECTS_ALL)):
            subject = SUBJECTS_ALL[subject_idx] 
            subj_n_voxels = NVOXELS[subject_idx]
            for cl_idx, cl in enumerate(CONTEXT_LENGTH):
                for layer_idx, layer in enumerate(LAYERS):
                    model = f"{model_prefix}_{cl}"
                    _, _, pca_prod_file, pca_comp_file, _, _, _, _ = get_ready_indiv_pcafile(subject, model, basemodel, layer, cl)
                    if modality_idx == 0:
                        pca_file = pca_prod_file
                    elif modality_idx == 1:
                        pca_file = pca_comp_file

                    with tables.open_file(pca_file, "r") as PCA_f:
                        tmp_pc_scores = PCA_f.root.pc_scores.read()

                    pc_scores_all[layer_idx, cl_idx, subject_idx, :, :subj_n_voxels] = tmp_pc_scores[:n_store_pc, :subj_n_voxels]

        output = f"{RESULTS_DIR}ALLSUBJ_indiv_pc_scores_{basemodel}_{model}_{modality}.hdf"
        save_table_file(output, dict(pc_scores_all=pc_scores_all))


def summary_indiv_pca_bootstrap(basemodel, model_prefix, alpha=0.001):
    for modality_idx in [0, 1]:
        if modality_idx == 0:
            modality = 'prod'
        elif modality_idx == 1:
            modality = 'comp'

        pvals=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), N_PC), np.nan)
        n_pcs=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL)), np.nan)
        explained_stims=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), N_PERM, N_PC), np.nan)
        explained_weights=np.full((len(LAYERS), len(CONTEXT_LENGTH), len(SUBJECTS_ALL), N_PERM, N_PC), np.nan)
        for layer_idx, layer in LAYERS:
            for cl_idx, cl in CONTEXT_LENGTH:
                model = f"{model_prefix}_{cl}"
                postfix = f"layer-{layer}_cl-{cl}"
                for subject_idx in range(len(SUBJECTS_ALL)):
                    subject = SUBJECTS_ALL[subject_idx]
                    _, _, _, _, boot_prod_file, boot_comp_file, _, _ = get_ready_indiv_pcafile(subject, model, basemodel, layer, cl)
                    if modality_idx == 0:
                        boot_file = boot_prod_file
                    elif modality_idx == 1:
                        boot_file = boot_comp_file
                    
                    with tables.open_file(boot_file, "r") as BOOT_f:
                        pvals[layer_idx, cl_idx, subject_idx, :] = BOOT_f.root.pval.read()
                        explained_stims[layer_idx, cl_idx, subject_idx, :, :] = BOOT_f.root.explained_stim.read()
                        explained_weights[layer_idx, cl_idx, subject_idx, :, :] = BOOT_f.root.explained_weight.read()
                        n_pc = 0
                        for pc_i in range(N_PC):
                            if BOOT_f.root.pval.read()[pc_i] < alpha:
                                n_pc = n_pc + 1
                            else:
                                break

                        n_pcs[layer_idx, cl_idx] = n_pc
                        print(f"    {subject} {postfix}: #PC {n_pc}")

        output = f"{RESULTS_DIR}ALLSUBJ_indiv_pca_bootstrap_{basemodel}_{model}_{modality}_alpha{alpha}.hdf"
        save_table_file(output, dict(n_pcs=n_pcs, explained_stims=explained_stims, explained_weights=explained_weights, pvals=pvals))


def summary_indiv_pca_bootstrap_each(subject_idx, basemodel, model_prefix, alpha=0.001):
    for modality_idx in [0, 1]:
        if modality_idx == 0:
            modality = 'prod'
        elif modality_idx == 1:
            modality = 'comp'

        pvals=np.full((len(LAYERS), len(CONTEXT_LENGTH), N_PC), np.nan)
        n_pcs=np.full((len(LAYERS), len(CONTEXT_LENGTH)), np.nan)
        explained_stims=np.full((len(LAYERS), len(CONTEXT_LENGTH), N_PERM, N_PC), np.nan)
        explained_weights=np.full((len(LAYERS), len(CONTEXT_LENGTH), N_PERM, N_PC), np.nan)
        for layer_idx, layer in enumerate(LAYERS):
            for cl_idx, cl in enumerate(CONTEXT_LENGTH):
                model = f"{model_prefix}_{cl}"
                postfix = f"layer-{layer}_cl-{cl}"
                _, _, _, _, boot_prod_file, boot_comp_file, _, _ = get_ready_indiv_pcafile(SUBJECTS_ALL[subject_idx], model, basemodel, layer, cl)
                if modality_idx == 0:
                    modality = 'prod'
                    boot_file = boot_prod_file
                elif modality_idx == 1:
                    modality = 'comp'
                    boot_file = boot_comp_file
                
                with tables.open_file(boot_file, "r") as BOOT_f:
                    pvals[layer_idx, cl_idx, :] = BOOT_f.root.pval.read()
                    explained_stims[layer_idx, cl_idx, :, :] = BOOT_f.root.explained_stim.read()
                    explained_weights[layer_idx, cl_idx, :, :] = BOOT_f.root.explained_weight.read()
                    n_pc = 0
                    for pc_i in range(n_components):
                        if BOOT_f.root.pval.read()[pc_i] < alpha:
                            n_pc = n_pc + 1
                        else:
                            break

                    n_pcs[layer_idx, cl_idx] = n_pc
                    print(f"    {SUBJECTS_ALL[subject_idx]} {postfix}: #PC {n_pc}")

        output = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_indiv_pca_bootstrap_{basemodel}_{model}_{modality}_alpha{alpha}.hdf"
        save_table_file(output, dict(n_pcs=n_pcs, explained_stims=explained_stims, explained_weights=explained_weights, pvals=pvals))



def summary_stim_pc_var_explained():
    for modality_idx in [0, 1]:
        if modality_idx == 0:
            modality = 'prod'
        elif modality_idx == 1:
            modality = 'comp'

        pc_var_explained_all =np.full((len(LAYERS), len(CONTEXT_LENGTH), N_PC), np.nan)
        layer_idx=0
        for layer_idx, layer in enumerate(LAYERS):
            cl_idx=0
            for cl_idx, cl in enumerate(CONTEXT_LENGTH):
                postfix=f"_layer-{layer}_cl-{cl}"
                stim_pca_file = f"{RESULTS_DIR}group_stim_pca_{modality}{postfix}.hdf"
                with tables.open_file(stim_pca_file, "r") as PCA_f:
                    tmp_pc_var_explained = PCA_f.root.pc_var_explained.read()

                pc_var_explained_all[layer_idx, cl_idx, :] = tmp_pc_var_explained

        output = f"{RESULTS_DIR}ALLSUBJ_stim_pc_var_explained_{modality}.hdf"
        save_table_file(output, dict(pc_var_explained_all=pc_var_explained_all))






