import copy
import numpy as np
import tables
from scipy.stats import pearsonr, false_discovery_control
from dialogue.io import get_ready_statsfile, get_ready_primal, get_features_ses
from dialogue.config import TOTAL_SESS


################################################################################################################################################################################
################################################################################################################################################################################
### weight_stats_wrapper.py
################################################################################################################################################################################
################################################################################################################################################################################
def weight_corr(subject_idx, model, basemodel, layer=-1, cl=1):
    subject, weight_corr_svoxels, weight_corr_file = get_ready_statsfile(subject_idx, model, basemodel, layer=layer, cl=cl)
    w_prod, w_comp = get_average_weights(subject_idx, model, basemodel, layer=layer, cl=cl)
    corrs, sig_corrs = weight_corr_each(w_prod, w_comp)
    save_table_file(weight_corr_file, dict(corrs=corrs, sig_corrs=sig_corrs))


def weight_corr_unified(subject_idx, model, basemodel, unified_model, unified_model_prefix, layer=-1, cl=1):
    subject, weight_corr_svoxels, weight_corr_file = get_ready_statsfile(subject_idx, unified_model, basemodel, layer=layer, cl=cl)
    w_prod, w_comp = get_average_weights(subject_idx, model, basemodel, layer=layer, cl=cl)
    w_unified = get_average_weights_unified(subject_idx, basemodel, unified_model_prefix, layer=layer, cl=cl)
    prod_corrs, prod_sig_corrs = weight_corr_each(w_prod, w_unified)
    comp_corrs, comp_sig_corrs = weight_corr_each(w_comp, w_unified)
    save_table_file(weight_corr_file, dict(prod_corrs=prod_corrs, comp_corrs=comp_corrs, prod_sig_corrs=prod_sig_corrs, comp_sig_corrs=comp_sig_corrs))


def get_average_weights(subject_idx, model, basemodel, layer=-1, cl=1):
    weights_list=[]
    for ses in range(1, TOTAL_SESS[subject_idx]+1):
        _, primal_ses_im_file, _, _, _ = get_ready_primal(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
        with tables.open_file(primal_ses_im_file, "r") as WEIGHT_f:
            weights_list.append(WEIGHT_f.root.average_weights.read())

    average_weights = np.nanmean(weights_list, axis=0)
    _, _, n_features_list = get_features_ses(subject_idx, 1, model, layer=layer, cl=cl)
    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices=[ slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:])]
    w_prod=average_weights[slices[-2], :]
    w_comp=average_weights[slices[-1], :]
    return w_prod, w_comp


def get_average_weights_unified(subject_idx, basemodel, unified_model_prefix, layer=-1, cl=1):
    model = f"{unified_model_prefix}_{cl}"
    weights_list=[]
    for ses in range(1, TOTAL_SESS[subject_idx]+1):
        _, primal_ses_im_file, _, _, _ = get_ready_primal(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
        with tables.open_file(primal_ses_im_file, "r") as WEIGHT_f:
            weights_list.append(WEIGHT_f.root.average_weights.read())

    average_weights = np.nanmean(weights_list, axis=0)
    _, _, n_features_list = get_features_ses(subject_idx, 1, model, layer=layer, cl=cl)
    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices=[ slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:])]
    w_unified=average_weights[slices[-1], :]
    return w_unified


def weight_corr_each(w_prod, w_comp, alpha=0.001):
    corrs=np.full(w_prod.shape[1], np.nan)
    pvals = copy.deepcopy(corrs)
    for v_i in range(w_prod.shape[1]):
        tmp_w_prod=w_prod[:, v_i]
        tmp_w_comp=w_comp[:, v_i]
        is_all_nan_prod = np.all(np.isnan(tmp_w_prod))
        is_all_nan_comp = np.all(np.isnan(tmp_w_comp))
        is_all_inf_prod = np.all(np.isinf(tmp_w_prod))
        is_all_inf_comp = np.all(np.isinf(tmp_w_comp))
        is_constant_prod = np.all(tmp_w_prod == w_prod[0, v_i])
        is_constant_comp = np.all(tmp_w_comp == w_comp[0, v_i])
        if is_all_nan_prod == False and is_all_nan_comp == False and is_all_inf_prod == False and is_all_inf_comp == False and is_constant_prod == False and is_constant_comp == False:
            prod_nan_idx = np.where(np.isnan(tmp_w_prod))[0]
            comp_nan_idx = np.where(np.isnan(tmp_w_comp))[0]
            prod_inf_idx = np.where(np.isinf(tmp_w_prod))[0]
            comp_inf_idx = np.where(np.isinf(tmp_w_comp))[0]
            union_nan_idx = np.union1d(prod_nan_idx, comp_nan_idx)
            union_inf_idx = np.union1d(prod_inf_idx, comp_inf_idx)
            union_all_idx = np.union1d(union_nan_idx, union_inf_idx)
            tmp_w_prod = np.delete(tmp_w_prod, union_all_idx)
            tmp_w_comp = np.delete(tmp_w_comp, union_all_idx)
            corrs[v_i], pvals[v_i] = pearsonr(tmp_w_prod, tmp_w_comp)
        else:
            pvals[v_i]=1

    pvals[np.isnan(pvals)] = 1
    pvals_fdr=false_discovery_control(pvals)
    sig_corrs = copy.deepcopy(corrs)
    sig_corrs[ps_fdr>=alpha]=np.nan
    return corrs, sig_corrs
