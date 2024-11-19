import copy
import numpy as np
import tables
from scipy.stats import combine_pvalues, false_discovery_control
from dialogue.io import get_ready_himalaya, get_ready_primal, get_responses_ses
from dialogue.config import TOTAL_SESS, SUBJECTS_ALL, RESULTS_DIR


################################################################################################################################################################################
################################################################################################################################################################################
### evaluating_wrapper_eachsubj.py
################################################################################################################################################################################
################################################################################################################################################################################
def score_tests_intramodal(subject_idx, model, basemodel, layer=-1, cl=1, n_perm=1000, alpha=0.05):
    subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file = get_ready_himalaya(subject_idx, 1, model, basemodel, layer=layer, cl=cl, alpha=alpha)
    output=r_im_file

    r_scores_list=[]
    r_scores_split_list=[]
    for ses in range(1, TOTAL_SESS[subject_idx]+1):
        subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file = get_ready_himalaya(subject_idx, ses, model, basemodel, layer=layer, cl=cl, alpha=alpha)
        modeling_file=ses_im_file
        with tables.open_file(modeling_file) as MODEL_f:
            r_score=MODEL_f.root.r_scores.read()
            r_score_split=MODEL_f.root.r_scores_split.read()

        r_scores_list.append(r_score)
        r_scores_split_list.append(r_score_split)

    mean_r_scores=np.tanh(np.nanmean(np.arctanh(r_scores_list), axis=0))
    mean_r_scores_split=np.tanh(np.nanmean(np.arctanh(r_scores_split_list), axis=0))
    raw_r_scores_all = copy.deepcopy(mean_r_scores)
    raw_r_scores_split = copy.deepcopy(mean_r_scores_split)

    negative_scores_mask=mean_r_scores<0
    negative_scores_split_mask=mean_r_scores_split<0
    mean_r_scores[negative_scores_mask]=0
    mean_r_scores_split[negative_scores_split_mask]=0

    pval_list=[]
    for ses in range(1, TOTAL_SESS[subject_idx]+1):
        subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file = get_ready_himalaya(subject_idx, ses, model, basemodel, layer=layer, cl=cl, alpha=alpha)
        perm_file=perm_im_file
        with tables.open_file(perm_file, "r") as PERM_f:
            pval=PERM_f.root.p_corr.read()

        pval_list.append(pval)

    pvals=np.stack(pval_list)
    comb_pval = np.full(pvals.shape[1], np.nan)
    pvals[pvals == 0] = 0.0001
    for v_i in range(pvals.shape[1]):
        result=combine_pvalues(pvals[:,v_i])
        comb_p[v_i]=result.pvalue

    comb_p[negative_scores_mask]=1
    comb_p_fdr=false_discovery_control(comb_p)
    svoxels=comb_p_fdr < alpha
    sig_mean_r_scores=np.full((len(mean_r_scores)), np.nan)
    sig_mean_r_scores_split=np.full(mean_r_scores_split.shape, np.nan)
    sig_mean_r_scores[svoxels]=mean_r_scores[svoxels]
    sig_mean_r_scores_split[:,svoxels]=mean_r_scores_split[:,svoxels]

    print(f"saving file: {output}")
    save_table_file(output, dict(sig_r_scores_all=sig_mean_r_scores, sig_r_scores_split=sig_mean_r_scores_split, svoxels=svoxels, p=comb_p_fdr, n_perm=n_perm, alpha=alpha,
                                    raw_r_scores_all=raw_r_scores_all, raw_r_scores_split=raw_r_scores_split))

def score_tests_crossmodal(subject_idx, model, basemodel, layer=-1, cl=1, n_perm=1000, alpha=0.05):
    _, _, _, _, primal_r_cm_file, _ = get_ready_primal(subject_idx, 1, model, basemodel, layer=layer, cl=cl, alpha=alpha)
    output=primal_r_cm_file

    r_scores_list=[]
    r_scores_split_list=[]
    for ses in range(1, TOTAL_SESS[subject_idx]+1):
        _, _, primal_ses_cm_file, _, _, _ = get_ready_primal(subject_idx, ses, model, basemodel, layer=layer, cl=cl, alpha=alpha)
        with tables.open_file(primal_ses_cm_file) as MODEL_f:
            r_score=MODEL_f.root.r_scores.read()

        r_scores_list.append(r_score)

    mean_r_scores = np.tanh(np.nanmean(np.arctanh(r_scores_list), axis=0))
    raw_r_scores_all = copy.deepcopy(mean_r_score)

    negative_scores_mask = mean_r_scores < 0
    mean_r_scores[negative_scores_mask] = 0

    pval_list=[]
    for ses in range(1, TOTAL_SESS[subject_idx]+1):   
        subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file = get_ready_himalaya(subject_idx, ses, model, basemodel, layer=layer, cl=cl, alpha=alpha)
        perm_file=perm_cm_file
        with tables.open_file(perm_file, "r") as PERM_f:
            pval=PERM_f.root.p_corr.read()

        pval_list.append(pval)

    pvals=np.stack(pval_list)
    comb_pval=np.empty(pvals.shape[1])
    pvals[pvals == 0] = 0.0001
    for v_i in range(pvals.shape[1]):
        result=combine_pvalues(pvals[:,v_i])
        comb_pval[v_i]=result.pvalue

    comb_p[negative_scores_mask]=1
    comb_p_fdr=false_discovery_control(comb_p)
    svoxels=comb_p_fdr<alpha
    sig_mean_r_scores=np.full((len(mean_r_scores)), np.nan)
    sig_mean_r_scores[svoxels]=mean_r_scores[svoxels]

    print(f"saving file: {output}")
    save_table_file(output, dict(sig_r_scores_all=sig_mean_r_scores, raw_r_scores_all=raw_r_scores_all, svoxels=svoxels, p=comb_p_fdr, n_perm=n_perm, alpha=alpha))


def score_tests_primal(subject_idx, model, basemodel, layer=-1, cl=1, n_perm=1000, alpha=0.05, test_idx=1):
    _, _, _, primal_r_im_file, primal_r_cm_file, _ = get_ready_primal(subject_idx, 1, model, basemodel, layer=layer, cl=cl, alpha=alpha)
    if test_idx == 0:
        score_file=primal_r_im_file
    elif test_idx == 1:
        score_file=primal_r_cm_file

    r_scores_list=[]
    r_scores_split_list=[]
    for ses in range(1, TOTAL_SESS[subject_idx]+1):
        if test_idx == 0:
            _, ses_im_file, _, _, _, _, _, _ = get_ready_himalaya(subject_idx, ses, model, basemodel, layer=layer, cl=cl, alpha=alpha)
            modeling_file=ses_im_file
        elif test_idx == 1:
            _, _, primal_ses_cm_file, _, _, _ = get_ready_primal(subject_idx, ses, model, basemodel, layer=layer, cl=cl, alpha=alpha)
            modeling_file=primal_ses_cm_file

        with tables.open_file(modeling_file) as MODEL_f:
            r_score=MODEL_f.root.r_scores.read()
            r_score_split=MODEL_f.root.r_scores_split.read()

        r_scores_list.append(r_score)
        r_scores_split_list.append(r_score_split)

    mean_r_scores = np.tanh(np.nanmean(np.arctanh(r_scores_list), axis=0))
    mean_r_scores_split = np.tanh(np.nanmean(np.arctanh(r_scores_split_list), axis=0))

    raw_r_scores_all = mean_r_scores
    raw_r_scores_split = mean_r_scores_split

    negative_scores_mask = mean_r_scores < 0
    negative_scores_split_mask = mean_r_scores_split < 0
    mean_r_scores[negative_scores_mask] = 0
    mean_r_scores_split[negative_scores_split_mask] = 0

    pval_list=[]
    for ses in range(1, TOTAL_SESS[subject_idx]+1):   
        subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file = get_ready_himalaya(subject_idx, ses, model, basemodel, layer=layer, cl=cl, alpha=alpha)
        if test_idx == 0:
            perm_file=perm_im_file
        elif test_idx == 1:
            perm_file=perm_cm_file

        with tables.open_file(perm_file, "r") as PERM_f:
            pval=PERM_f.root.p_corr.read()

        pval_list.append(pval)

    pvals=np.stack(pval_list)
    comb_pval=np.empty(pvals.shape[1])
    pvals[pvals == 0] = 0.0001
    for v_i in range(pvals.shape[1]):
        result=combine_pvalues(pvals[:,v_i])
        comb_pval[v_i]=result.pvalue

    comb_pval[negative_scores_mask]=1
    comb_pval_fdr=false_discovery_control(comb_pval)
    svoxels=comb_pval_fdr<alpha
    sig_mean_r_scores=np.full((len(mean_r_scores)), np.nan)
    sig_mean_r_scores_split=np.full(mean_r_scores_split.shape, np.nan)
    sig_mean_r_scores[svoxels]=mean_r_scores[svoxels]
    sig_mean_r_scores_split[:,svoxels]=mean_r_scores_split[:,svoxels]

    print(f"saving file: {score_file}")
    save_table_file(score_file, dict(sig_r_scores_all=sig_mean_r_scores, sig_r_scores_split=sig_mean_r_scores_split, svoxels=svoxels, p=comb_p_fdr, n_perm=n_perm, alpha=alpha,
                                        raw_r_scores_all=raw_r_scores_all, raw_scores_split=raw_r_scores_split))



##########################################################################################################################################################################
##########################################################################################################################################################################
### Variance partitioning
##########################################################################################################################################################################
##########################################################################################################################################################################
def varpart(subject_idx, basemodel, model_prefix='gpt_context', layer=-1, cl=1):
    model = f'{model_prefix}_{cl}'
    postfix = f"_layer-{layer}_cl-{cl}"
    output = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_{basemodel}_{model}{postfix}_vp.hdf"
    ########################################################################################################################
    ### Separate Linguistic model (Production U Comprehension), Production model and Comprehension model
    model_prod = f'{model_prefix}_{cl}_prod'
    model_comp = f'{model_prefix}_{cl}_comp'
    resp_list=[]
    presp_all_list=[]
    presp_prod_list=[]
    presp_comp_list=[]
    for ses in range(1, TOTAL_SESS[subject_idx]+1):
        _, Y_test = get_responses_ses(subject_idx, ses, model, basemodel)
        resp_list.append(Y_test)
        ses_all_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}_ses-{ses}{postfix}_intramodal.hdf"
        ses_prod_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model_prod}_ses-{ses}{postfix}_intramodal.hdf"
        ses_comp_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model_comp}_ses-{ses}{postfix}_intramodal.hdf"
        with tables.open_file(ses_all_file, "r") as MODEL_ALL_f:
            Y_test_pred_all = MODEL_ALL_f.root.Y_test_pred_all.read()

        with tables.open_file(ses_prod_file, "r") as MODEL_prod_f:
            Y_test_pred_prod = MODEL_prod_f.root.Y_test_pred_all.read()

        with tables.open_file(ses_comp_file, "r") as MODEL_comp_f:
            Y_test_pred_comp = MODEL_comp_f.root.Y_test_pred_all.read()

        presp_all_list.append(Y_test_pred_all)
        presp_prod_list.append(Y_test_pred_prod)
        presp_comp_list.append(Y_test_pred_comp)

    resp = np.concatenate(resp_list)
    presp_all = np.concatenate(presp_all_list)
    presp_prod = np.concatenate(presp_prod_list)
    presp_comp = np.concatenate(presp_comp_list)
    Tvar = np.sum((resp - np.nanmean(resp, axis=0)) ** 2, axis=0)
    R_all = 1 - np.sum((presp_all - resp) ** 2, axis=0) / Tvar
    R_prod = 1 - np.sum((presp_prod - resp) ** 2, axis=0) / Tvar
    R_comp = 1 - np.sum((presp_comp - resp) ** 2, axis=0) / Tvar

    ########################################################################################################################
    ### Variance partitioning
    R_prod_RC = R_all - R_comp
    R_comp_RC = R_all - R_prod
    R_Intersection_RC = R_prod + R_comp - R_all

    # Transform from R-squared to R
    R_Intersection_RC[R_Intersection_RC < 0] = 0
    R_prod_RC[R_prod_RC < 0] = 0
    R_comp_RC[R_comp_RC < 0] = 0
    RC_intersection = np.sqrt(R_Intersection_RC)
    RC_prod = np.sqrt(R_prod_RC)
    RC_comp = np.sqrt(R_comp_RC)
    save_table_file(output, dict(RC_intersection=RC_intersection, RC_prod=RC_prod, RC_comp=RC_comp))


def best_uniqvarpart_rgb(subject_idx, basemodel, model_prefix='gpt_context', layer=-1, cl=1):
    model = f'{model_prefix}_{cl}'
    postfix = f"_layer-{layer}_cl-{cl}"
    vp_file = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_{basemodel}_{model}{postfix}_vp.hdf"
    with tables.open_file(vp_file, "r") as VP_f:
        A = VP_f.root.RC_prod.read()
        B = VP_f.root.RC_comp.read()
        A_B = VP_f.root.RC_intersection.read()

    # 1:production, 2: comprehension, 3:intersection
    VRGB=np.full(len(A), np.nan)
    VRGB[(A_B>A) & (A_B>B)] = 3
    VRGB[(A>A_B) & (A>B)] = 1
    VRGB[(B>A_B) & (B>A)] = 2
    output = f"{RESULTS_DIR}{SUBJECTS_ALL[subject_idx]}_{basemodel}_{model}{postfix}_intramodal_best_vp.hdf"
    save_table_file(output, dict(VRGB=VRGB))

