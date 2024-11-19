import numpy as np
import os
import tables
from himalaya.backend import set_backend
from himalaya.scoring import correlation_score
from dialogue.io import get_ready_himalaya, get_responses_ses, get_train_test_runs, save_table_file, get_ready_primal
from dialogue.config import SUBJECTS_ALL, N_SCANS

backend = set_backend("torch_cuda", on_error="warn")



##########################################################################################################################################################################
##########################################################################################################################################################################
### perm_wrapper_eachsubj.py
##########################################################################################################################################################################
##########################################################################################################################################################################
def run_rscore_perm(subject_idx, ses, model, basemodel, layer, cl, test_idx):
    _, ses_im_file, _, _, perm_im_file, perm_cm_file, _, _ = get_ready_himalaya(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
    if test_idx == 0:
        perm_file=perm_im_file
        model_file=ses_im_file
    elif test_idx == 1:
        _, _, primal_ses_cm_file, _, _, _ = get_ready_primal(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
        perm_file=perm_cm_file
        model_file=primal_ses_cm_file

    run_permtest_ses(subject_idx, ses, model, basemodel, model_file, layer=layer, cl=cl, test_idx=test_idx)


def run_permtest_ses(subject_idx, ses, model, basemodel, layer=-1, cl=1, test_idx=0, n_blocksize=20, n_perm=1000):
    _, ses_im_file, _, _, perm_im_file, perm_cm_file, _, _ = get_ready_himalaya(subject_idx, ses, model=model, basemodel=basemodel, layer=layer, cl=cl)
    if test_idx == 0:
        modeling_file=ses_im_file
        output=perm_im_file
    elif test_idx == 1:
        _, _, primal_ses_cm_file, _, _, _, _, _ = get_ready_primal(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
        modeling_file=primal_ses_cm_file
        output=perm_cm_file

    if os.path.exists(output):
        with tables.open_file(output) as OUTPUT_f:
            print(OUTPUT_f)
            if "/p_corr" in OUTPUT_f.root:
                print(f"Already performed corr_score permutation: {SUBJECTS_ALL[subject_idx]}_ses-{ses}_layer-{layer}_cl-{cl}")
                return
            else:
                print(f"Not performed corr_score permutation yet: {SUBJECTS_ALL[subject_idx]}_ses-{ses}_layer-{layer}_cl-{cl}")
            
    print("")
    print("====== PERMUTATION TEST ======")
    _, Y_test = get_responses_ses(subject_idx, ses, basemodel)
    with tables.open_file(modeling_file) as MODEL_f:
        Y_pred = np.nan_to_num(MODEL_f.root.Y_test_pred_all.read())
        r_scores = np.nan_to_num(MODEL_f.root.r_scores.read())
        
    assert Y_test.shape == Y_pred.shape, print(Y_test.shape, Y_pred.shape)
    _, test_runs = get_train_test_runs(subject_idx, ses)

    perm_r_scores = np.empty((n_perm, Y_test.shape[1]))
    for rep in range(n_perm):
        if rep%100==1:
            print(f"permutation : {rep}/{n_perm}")
        
        perm_r_score=get_permuted_r_scores(Y_test, Y_pred, n_blocksize, test_runs)
        perm_r_scores[rep,:]=perm_r_score

    perm_r_scores=backend.to_numpy(perm_r_scores)
    comparison_matrix = perm_r_scores > np.tile(r_scores, (n_perm, 1))
    p_corr = np.sum(comparison_matrix, axis=0)/n_perm
    save_table_file(output, dict(p_corr=p_corr, r_scores=r_scores, perm_r_scores=perm_r_scores))    


###
def get_permuted_r_scores(Y_true, Y_pred, n_blocksize, test_runs):
    Y_true_part=get_blocked_responses(Y_true, n_blocksize, test_runs)
    Y_perm_part=get_blocked_perm_responses(Y_pred, n_blocksize, test_runs)
    perm_r_scores = np.nan_to_num([backend.to_numpy(correlation_score(Y_true_part, Y_perm_part))])
    return perm_r_scores

def get_blocked_responses(Y, n_blocksize, test_runs):
    n_blocks=int(N_SCANS / n_blocksize)
    Y_i_part=[]
    for r_i in range(len(test_runs)):
        Y_i=Y[r_i*N_SCANS:(r_i+1)*N_SCANS,:]
        Y_i_part.append(Y_i[:n_blocksize*n_blocks])
    Y_part=np.concatenate(Y_i_part, axis=0)
    return Y_part

def get_blocked_perm_responses(Y, n_blocksize, test_runs):
    n_blocks=int(N_SCANS / n_blocksize)
    idx = []
    Y_perm_i_part=[]
    for r_i in range(len(test_runs)):
        Y_i=Y[r_i*N_SCANS:(r_i+1)*N_SCANS,:]
        Y_i_part=Y_i[:n_blocksize*n_blocks]
        block_idx = np.random.choice(range(n_blocks), n_blocks)
        idx = []
        for ii in block_idx:
            start, end = ii*n_blocksize, (ii+1)*n_blocksize
            idx.extend(range(start, end))
        Y_perm_i_part.append(Y_i_part[idx,:])
    Y_perm_part=np.concatenate(Y_perm_i_part, axis=0)
    return Y_perm_part


