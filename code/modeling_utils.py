import numpy as np
import tables
from scipy.stats import zscore
from himalaya.backend import set_backend
from himalaya.kernel_ridge import primal_weights_weighted_kernel_ridge, MultipleKernelRidgeCV, ColumnKernelizer, Kernelizer
from himalaya.scoring import r2_score_split, correlation_score, correlation_score_split
from voxelwise_tutorials.delayer import Delayer
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from modeling.io import get_ready_himalaya, get_responses_ses, get_data_ready, save_table_file, get_ready_primal
from modeling.config import TESTSET_RUNS, TOTAL_RUNS, TOTAL_SESS, SUBJECTS_ALL, RESULTS_DIR, ITER, DELAY, ALPHAS, CV, N_TARGETS_BATCH, ROOT_DIR
backend = set_backend("torch_cuda", on_error="warn")


##########################################################################################################################################################################
##########################################################################################################################################################################
### modeling_wrapper.py
##########################################################################################################################################################################
##########################################################################################################################################################################

####################################################################################
### Step1/3 - Head Motion model 
def run_headmotion_analysis(subject_idx, ses, headmotion_model):
    _, headmotion_ses_file, _, bold_res_headmotion_file, _, _, _, _ = get_ready_himalaya(subject_idx, ses, model=headmotion_model, basemodel=headmotion_model)
    run_himalaya(subject_idx, ses, headmotion_model, headmotion_model, headmotion_ses_file)
    stepwise_1_BOLD(subject_idx, ses, headmotion_model, headmotion_ses_file, bold_res_headmotion_file)


####################################################################################
### Step2/3 - Random Embedding model 
def run_random_embedding_analysis(subject_idx, ses, random_model, headmotion_model, layer=0, cl=1):
    _, random_ses_file, _, bold_res_random_file, _, _, _, _ = get_ready_himalaya(subject_idx, ses, model=random_model, basemodel=headmotion_model, layer=layer, cl=cl)
    run_himalaya(subject_idx, ses, random_model, headmotion_model, random_ses_file, layer=layer, cl=cl)
    primal_intramodal_predict(subject_idx, ses, random_model, headmotion_model, layer=layer, cl=cl)
    primal_crossmodal_predict(subject_idx, ses, random_model, headmotion_model, layer=layer, cl=cl)
    stepwise_2_BOLD(subject_idx, ses, random_model, headmotion_model, random_model_file, bold_res_random_file)


####################################################################################
### Step3/3 - Separate and Unified Linguistic model 
def run_llm_analysis(subject_idx, ses, target_model, headmotion_model, random_model, layer=0, cl=1):
    subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file = get_ready_himalaya(subject_idx, ses, model=target_model, basemodel=random_model, layer=layer)
    random_model_file = f"{RESULTS_DIR}/{SUBJECTS_ALL[subject_idx]}_{headmotion_model}_{random_model}_ses-{ses}_layer-0_cl-1_intramodal.hdf"
    run_himalaya(subject_idx, ses, target_model, random_model, ses_im_file, layer=layer, cl=cl)
    primal_intramodal_predict(subject_idx, ses, target_model, random_model, layer=layer, cl=cl)
    if 'prod' not in target_model and 'comp' not in target_model and 'unified' not in target_model:
        primal_crossmodal_predict(subject_idx, ses, target_model, random_model, layer=layer, cl=cl)


####################################################################################
### stepwise BOLD 
def stepwise_1_BOLD(subject_idx, ses, headmotion_model, headmotion_ses_file, bold_res_headmotion_file):
    Y_train, Y_test = get_responses_ses(subject_idx, ses, headmotion_model, headmotion_model)
    with tables.open_file(headmotion_ses_file, "r") as MODEL_f:
        Y_train_pred = np.nan_to_num(MODEL_f.root.Y_train_pred_all.read())
        Y_test_pred = np.nan_to_num(MODEL_f.root.Y_test_pred_all.read())
    
    Y_train = zscore(Y_train - Y_train_pred, axis=0)
    Y_test = zscore(Y_test - Y_test_pred, axis=0)
    save_table_file(bold_res_headmotion_file, dict(Y_train=Y_train, Y_test=Y_test))


def stepwise_2_BOLD(subject_idx, ses, random_model, headmotion_model, random_model_file, bold_res_random_file):
    Y_train, Y_test = get_responses_ses(subject_idx, ses, random_model, headmotion_model)
    with tables.open_file(random_model_file, "r") as MODEL_f:
        Y_train_pred = np.nan_to_num(MODEL_f.root.Y_train_pred_all.read())
        Y_test_pred = np.nan_to_num(MODEL_f.root.Y_test_pred_all.read())
    
    Y_train = zscore(Y_train - Y_train_pred, axis=0)
    Y_test = zscore(Y_test - Y_test_pred, axis=0)
    save_table_file(bold_res_random_file, dict(Y_train=Y_train, Y_test=Y_test))


def run_himalaya(subject_idx, ses, model, basemodel, ses_im_file, layer=-1, cl=1,
                delays=DELAY, n_iter=ITER, cv=CV, alphas=ALPHAS, n_targets_batch=N_TARGETS_BATCH,
                n_alphas_batch=5, return_weights="dual", diagonalize_method="svd", n_targets_batch_refit=200, random_idx=0):
    subject=SUBJECTS_ALL[subject_idx]
    ###############################################################################
    # Load the data
    # -------------------------
    X_train, X_test, Y_train, Y_test, feature_names, n_features_list, train_runs, run_onsets = get_data_ready(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
    ###############################################################################
    # Define the cross-validation scheme
    # -------------------------
    # print('=== CV setting ===')
    cv=check_cv(cv)
    ###############################################################################
    # Common preprocessing and column_kernelizer
    # -------------------------
    preprocess_pipeline = make_pipeline(StandardScaler(with_mean=True, with_std=True), Delayer(delays=delays), Kernelizer(kernel="linear"))
    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices=[ slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:])]
    kernelizers_tuples =[ (name, preprocess_pipeline, slice_) for name, slice_ in zip(feature_names, slices) ]
    column_kernelizer = ColumnKernelizer(kernelizers_tuples)
    ###############################################################################
    # Define and fit the model using random search
    # -------------------------
    print('=== MultipleKernelRidgeCV, random search ===')
    solver_1_function = MultipleKernelRidgeCV.ALL_SOLVERS["random_search"]
    solver_1_params = dict(n_iter=n_iter, alphas=alphas, n_targets_batch=n_targets_batch, n_alphas_batch=n_alphas_batch, n_targets_batch_refit=n_targets_batch_refit, diagonalize_method=diagonalize_method)
    mkr_model_1 = MultipleKernelRidgeCV(kernels="precomputed", solver="random_search", solver_params=solver_1_params, cv=cv)
    pipeline_1 = make_pipeline(column_kernelizer, mkr_model_1)
    pipe_results_1 = pipeline_1.fit(X_train, Y_train)
    ###############################################################################
    # Define and fit the model using gradient descent
    # -------------------------
    print('=== MultipleKernelRidgeCV, gradient descent ===')
    solver_2_params = dict(max_iter=10, hyper_gradient_method="direct", max_iter_inner_hyper=10)
    solver_2_function = MultipleKernelRidgeCV.ALL_SOLVERS["hyper_gradient"]
    mkr_model_2 = MultipleKernelRidgeCV(kernels="precomputed", solver="hyper_gradient", solver_params=solver_2_params)
    pipeline_2 = make_pipeline(column_kernelizer, mkr_model_2)
    tmp_cv_scores = backend.to_numpy(pipeline_1[-1].cv_scores_)
    best_cv_scores=tmp_cv_scores.max(0)
    perc_top_voxels=60
    mask = best_cv_scores > np.percentile(best_cv_scores, 100 - perc_top_voxels)
    pipeline_2[-1].solver_params['initial_deltas'] = pipeline_1[-1].deltas_[:, mask]
    pipe_results_2 = pipeline_2.fit(X_train, Y_train[:, mask])
    ###############################################################################
    # Postprocessing 
    # -------------------------
    mkr_results_1=pipe_results_1[1]
    mkr_results_2=pipe_results_2[1]
    deltas = backend.to_numpy(backend.copy(mkr_results_1.deltas_))
    deltas[:, mask] = backend.to_numpy(mkr_results_2.deltas_)
    dual_weights = backend.to_numpy(backend.copy(mkr_results_1.dual_coef_))
    dual_weights[:, mask] = backend.to_numpy(mkr_results_2.dual_coef_)
    # r2_score 
    r2_scores_1 = pipeline_1.score(X_test, Y_test)
    r2_scores_2 = backend.copy(r2_scores_1)
    r2_scores_2[mask] = pipeline_2.score(X_test, Y_test[:, mask])
    r2_scores_1 = backend.to_numpy(r2_scores_1)
    r2_scores = backend.to_numpy(r2_scores_2)
    # r2_score for each kernel
    Y_test_pred_split_1 = pipeline_1.predict(X_test, split=True)
    r2_scores_split_1 = r2_score_split(Y_test, Y_test_pred_split_1)
    Y_test_pred_split_2 = backend.copy(Y_test_pred_split_1)
    Y_test_pred_split_2[:,:,mask] = pipeline_2.predict(X_test, split=True)
    r2_scores_split_2 = r2_score_split(Y_test, Y_test_pred_split_2)
    Y_test_pred_split = backend.to_numpy(Y_test_pred_split_2)
    r2_scores_split = backend.to_numpy(r2_scores_split_2)
    Y_test_pred_all_1 = pipeline_1.predict(X_test, split=False)
    r2_scores_all_1 = r2_score_split(Y_test, Y_test_pred_all_1)
    Y_test_pred_all_2 = backend.copy(Y_test_pred_all_1)
    Y_test_pred_all_2[:,mask] = pipeline_2.predict(X_test, split=False)
    r2_scores_all_2 = r2_score_split(Y_test, Y_test_pred_all_2)
    Y_test_pred_all = backend.to_numpy(Y_test_pred_all_2)
    print("(n_samples_test, n_voxels) =", Y_test_pred_all.shape)
    Y_train_pred_all_1 = pipeline_1.predict(X_train, split=False)
    Y_train_pred_all_2 = backend.copy(Y_train_pred_all_1)
    Y_train_pred_all_2[:,mask] = pipeline_2.predict(X_train, split=False)
    Y_train_pred_all = backend.to_numpy(Y_train_pred_all_2)
    print("(n_samples_train, n_voxels) =", Y_train_pred_all.shape)
    # r score
    r_scores = backend.to_numpy(correlation_score(Y_test, Y_test_pred_all))
    r_scores_split = backend.to_numpy(correlation_score_split(Y_test, Y_test_pred_split))
    ###############################################################################
    # Save
    # -------------------------
    save_table_file(ses_im_file, dict(deltas = deltas, dual_weights = dual_weights, r2_scores=r2_scores, r2_scores_split=r2_scores_split, r_scores=r_scores, r_scores_split=r_scores_split,
                                Y_test_pred_all = Y_test_pred_all, Y_test_pred_split = Y_test_pred_split, Y_train_pred_all = Y_train_pred_all))


####################################################################################
### same-modality prediciton
def primal_intramodal_predict(subject_idx, ses, model, basemodel, layer=-1, cl=1, delays=DELAY, random_idx=0):
    subject, primal_ses_im_file, primal_ses_cm_file, primal_r_im_file, primal_r_cm_file, primal_weights_file = get_ready_primal(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
    Xs_train_delays, Xs_test_delays, Y_test = get_data_prep_with_delay_list(subject_idx, ses, model, basemodel, layer=layer, cl=cl, delays=delays)
    subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file = get_ready_himalaya(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
    with tables.open_file(ses_im_file, "r") as MODEL_f:
        dual_weights = backend.asarray(MODEL_f.root.dual_weights.read(), dtype=backend.float32)
        deltas = backend.asarray(MODEL_f.root.deltas.read(), dtype=backend.float32)
    
    primal_weights_list=[]
    primal_weights_llm_list=[]
    Y_train_pred_list=[]
    Y_test_pred_list=[]
    Y_test_pred_llm_list=[]
    for delay_i in range(len(delays)):
        # primal_weights_i: list of torch.Size([n_features, n_targets]) for each feature
        primal_weights_i = primal_weights_weighted_kernel_ridge(dual_weights, deltas, Xs_train_delays[delay_i])
        primal_weights_llm_i=[]
        for f_i in range(len(primal_weights_i)):
            tmp_weights=primal_weights_i[f_i]
            if 'gpt' in model:
                primal_weights_llm_i.append(primal_weights_i[f_i])

        primal_weights_list.append(np.concatenate(primal_weights_i))
        primal_weights_llm_list.append(np.concatenate(primal_weights_llm_i))
        Y_test_pred_i = backend.to_numpy(backend.stack([X @ backend.asarray(w) for X, w in zip(Xs_test_delays[delay_i], primal_weights_i)]).sum(0))
        Y_test_pred_list.append(Y_test_pred_i)
        Y_test_pred_llm_i = backend.to_numpy(backend.stack([X @ backend.asarray(w) for X, w in zip(Xs_test_delays[delay_i], primal_weights_llm_i)]).sum(0))
        Y_test_pred_llm_list.append(Y_test_pred_llm_i)

    primal_weights=np.concatenate(primal_weights_list)
    Y_test_pred=np.sum(Y_test_pred_list, axis=0)
    Y_test_pred_llm=np.sum(Y_test_pred_llm_list, axis=0)
    r2_scores = get_r2(Y_test, Y_test_pred)
    r2_scores_llm = get_r2(Y_test, Y_test_pred_llm)

    # weights 
    primal_weights /= np.linalg.norm(primal_weights, axis=0)[None]
    delayer = Delayer(delays=delays)
    primal_weights_per_delay = delayer.reshape_by_delays(primal_weights, axis=0)
    average_weights = np.nanmean(primal_weights_per_delay, axis=0)
    print(f"saving {primal_ses_im_file}")
    save_table_file(primal_ses_im_file, dict(primal_weights=primal_weights, average_weights=average_weights, Y_test_pred_llm=Y_test_pred_llm, r2_scores_llm=r2_scores_llm ))


####################################################################################
### cross-modality prediciton
def primal_crossmodal_predict(subject_idx, ses, model, basemodel, layer=-1, cl=1, delays=DELAY):
    subject, primal_ses_im_file, primal_ses_cm_file, primal_r_im_file, primal_r_cm_file, _ = get_ready_primal(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
    Xs_train_delays, Xs_test_delays, Y_test = get_data_prep_with_delay_list(subject_idx, ses, model, basemodel, layer=layer, cl=cl, delays=delays)
    subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file= get_ready_himalaya(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
    with tables.open_file(ses_im_file, "r") as MODEL_f:
        dual_weights = backend.asarray(MODEL_f.root.dual_weights.read(), dtype=backend.float32)
        deltas = backend.asarray(MODEL_f.root.deltas.read(), dtype=backend.float32)
    
    swap_primal_weights_list=[]
    swap_primal_weights_llm_list=[]
    Y_test_cmpred_split_list=[]
    Y_test_cmpred_all_list=[]
    Y_test_cmpred_llm_list=[]

    swap_idx=[1,0]
    print(f"swap_idx: {swap_idx}")
    for delay_i in range(len(delays)):
        primal_weights_i = primal_weights_weighted_kernel_ridge(dual_weights, deltas, Xs_train_delays[delay_i])
        swap_primal_weights_i=[]
        swap_primal_weights_llm_i=[]
        for f_i in range(len(primal_weights_i)):
            swap_primal_weights_i.append(primal_weights_i[swap_idx[f_i]])

            tmp_weights=primal_weights_i[f_i]
            if 'gpt' in model:
                if f_i == 0:
                    swap_primal_weights_llm_i.append(primal_weights_i[1])
                elif f_i == 1:
                    swap_primal_weights_llm_i.append(primal_weights_i[0])

        swap_primal_weights_list.append(np.concatenate(swap_primal_weights_i))
        swap_primal_weights_llm_list.append(np.concatenate(swap_primal_weights_llm_i))
        Y_test_cmpred_i_split = backend.to_numpy(backend.stack([X @ backend.asarray(w) for X, w in zip(Xs_test_delays[delay_i], swap_primal_weights_i)]))
        Y_test_cmpred_i_all = np.sum(Y_test_cmpred_i_split, axis=0)
        Y_test_cmpred_split_list.append(Y_test_cmpred_i_split)
        Y_test_cmpred_all_list.append(Y_test_cmpred_i_all)

        Y_test_cmpred_i_llm = backend.to_numpy(backend.stack([X @ backend.asarray(w) for X, w in zip(Xs_test_delays[delay_i], swap_primal_weights_llm_i)])).sum(0)
        Y_test_cmpred_llm_list.append(Y_test_cmpred_i_llm)

    swap_primal_weights_list=np.concatenate(swap_primal_weights_list)
    Y_test_cmpred_split=np.sum(Y_test_cmpred_split_list, axis=0)
    Y_test_cmpred_all=np.sum(Y_test_cmpred_all_list, axis=0)
    Y_test_cmpred_llm=np.sum(Y_test_cmpred_llm_list, axis=0)

    r2_scores = get_r2(Y_test, Y_test_cmpred_all)
    r2_scores_split=np.empty((Y_test_cmpred_split.shape[0], Y_test_cmpred_all.shape[1]))
    for ii in range(Y_test_cmpred_split.shape[0]):
        r2_scores_split[ii,:]=get_r2(Y_test, Y_test_cmpred_split[ii,:,:])

    r2_scores_llm = get_r2(Y_test, Y_test_cmpred_llm)
    assert Y_test.shape == Y_test_cmpred_all.shape, print(Y_test.shape, Y_test_cmpred_all.shape)

    r_scores = backend.to_numpy(correlation_score(Y_test, Y_test_cmpred_all))
    r_scores_split = backend.to_numpy(correlation_score_split(Y_test, Y_test_cmpred_all))
    save_table_file(primal_ses_cm_file, dict(r2_scores=r2_scores, r2_scores_split=r2_scores_split, r2_scores_llm=r2_scores_llm, r_scores=r_scores, r_scores_split=r_scores_split, 
                                            Y_test_pred_all=Y_test_cmpred_all, Y_test_pred_split=Y_test_cmpred_split))


def get_data_prep_with_delay_list(subject_idx, ses, model, basemodel, layer=-1, cl=1, delays=DELAY):
    X_train, X_test, Y_train, Y_test, feature_names, n_features_list, train_runs, run_onsets = get_data_ready(subject_idx, ses, model, basemodel, layer=layer, cl=cl)
    # preprocess
    prep_pipeline = make_pipeline(StandardScaler(with_mean=True, with_std=True), Delayer(delays=delays))
    prep_pipeline.fit(X_train)
    X_train_delays = prep_pipeline.transform(X_train)
    X_test_delays = prep_pipeline.transform(X_test)
    # make list of each delay
    X_train_delays_3d = np.stack(np.split(X_train_delays, len(delays), axis=1))
    X_test_delays_3d = np.stack(np.split(X_test_delays, len(delays), axis=1))
    X_train_delays = backend.asarray(X_train_delays, dtype=backend.float32)
    X_test_delays = backend.asarray(X_test_delays, dtype=backend.float32)
    X_train_delays_3d = backend.asarray(X_train_delays_3d, dtype=backend.float32)
    X_test_delays_3d = backend.asarray(X_test_delays_3d, dtype=backend.float32)
    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices=[ slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:]) ]
    Xs_train_delays=[]
    Xs_test_delays=[]
    for delay_i in range(len(delays)):
        Xs_train=[]
        Xs_test=[]
        for slice_ in slices:
            Xs_train.append(X_train_delays_3d[delay_i, :, slice_])
            Xs_test.append(X_test_delays_3d[delay_i, :, slice_])

        Xs_train_delays.append(Xs_train)
        Xs_test_delays.append(Xs_test)
        
    return Xs_train_delays, Xs_test_delays, Y_test


def get_r2(Y_true_mat, Y_pred_mat):
    sst = np.sum((Y_true_mat - np.mean(Y_true_mat))**2, axis=0)  
    ssr = np.sum((Y_true_mat - Y_pred_mat)**2, axis=0) 
    r2 = 1 - ssr / sst
    return r2

