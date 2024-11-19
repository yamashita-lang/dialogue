import numpy as np
import tables
import h5py
from dialogue.github.modeling.config import TESTSET_RUNS, TOTAL_RUNS, SUBJECTS_ALL, DATA_DIR, FEATURES_DIR, RESULTS_DIR, MODEL_FEATURE_MATRIX, N_SCANS, N_PERM


################################################################################################################################################################################
def get_ready_himalaya(subject_idx, ses, model, basemodel, layer=-1, cl=1):
    subject=SUBJECTS_ALL[subject_idx]
    if layer < 0:
        ### Head Motion model
        postfix=''
    else:
        ### Separate and Unified Linguistic model, Random Embedding model
        postfix=f"_layer-{layer}_cl-{cl}"

    if model == basemodel:
        ### Head Motion model
        ses_im_file = f"{RESULTS_DIR}{subject}_{basemodel}_ses-{ses}{postfix}_intramodal.hdf"
        ses_cm_file = f"{RESULTS_DIR}{subject}_{basemodel}_ses-{ses}{postfix}_crossmodal.hdf"
        bold_res_file = f"{DATA_DIR}{subject}/{subject}_ses-{ses}_task-dialogue_bold_res_{basemodel}.hdf"
        perm_im_file = f"{RESULTS_DIR}{subject}_{basemodel}_ses-{ses}{postfix}_intramodal_perm_{N_PERM}.hdf"
        perm_cm_file = f"{RESULTS_DIR}{subject}_{basemodel}_ses-{ses}{postfix}_crossmodal_perm_{N_PERM}.hdf"
        r_im_file = f"{RESULTS_DIR}{subject}_{basemodel}{postfix}_intramodal.hdf"
        r_cm_file = f"{RESULTS_DIR}{subject}_{basemodel}{postfix}_crossmodal.hdf"
    else:
        ### Separate and Unified Linguistic model, Random Embedding model
        ses_im_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}_ses-{ses}{postfix}_intramodal.hdf"
        ses_cm_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}_ses-{ses}{postfix}_crossmodal.hdf"
        bold_res_file = f"{DATA_DIR}{subject}/{subject}_ses-{ses}_task-dialogue_bold_res_{basemodel}_{model}.hdf"
        perm_im_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}_ses-{ses}{postfix}_intramodal_perm_{N_PERM}.hdf"
        perm_cm_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}_ses-{ses}{postfix}_crossmodal_perm_{N_PERM}.hdf"
        r_im_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}{postfix}_intramodal.hdf"
        r_cm_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}{postfix}_crossmodal.hdf"

    return subject, ses_im_file, ses_cm_file, bold_res_file, perm_im_file, perm_cm_file, r_im_file, r_cm_file


def get_ready_primal(subject_idx, ses, model, basemodel, layer=-1, cl=1, alpha=0.05):
    subject=SUBJECTS_ALL[subject_idx]
    if layer < 0:
        postfix=''
    else:
        postfix=f"_layer-{layer}_cl-{cl}"

    primal_ses_im_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}_ses-{ses}{postfix}_primal_intramodal.hdf"
    primal_ses_cm_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}_ses-{ses}{postfix}_primal_crossmodal.hdf"
    primal_r_im_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}{postfix}_primal_intramodal.hdf"
    primal_r_cm_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}{postfix}_primal_crossmodal.hdf"
    primal_weights_file = f"{RESULTS_DIR}{subject}_{basemodel}_{model}{postfix}_primal_weights.hdf"
    return subject, primal_ses_im_file, primal_ses_cm_file, primal_r_im_file, primal_r_cm_file, primal_weights_file



def get_data_ready(subject_idx, ses, model, basemodel, layer=-1, cl=1):
    print(f'     === {SUBJECTS_ALL[subject_idx]}, basemodel/model = {basemodel}/{model}, ses {ses}, layer {layer}, cl {cl}')
    Y_train, Y_test = get_responses_ses(subject_idx, ses, model, basemodel)
    feature_names = MODEL_FEATURE_MATRIX[model]
    X_train, X_test, n_features_list = get_features_ses(subject_idx, ses, model, layer, cl)
    print("(Y_train: n_samples_train, n_voxels) = ", Y_train.shape)
    print("(Y_test : n_samples_test , n_voxels) = ", Y_test.shape)
    print("(X_train: n_samples_train, n_features_total) = ", X_train.shape)
    print("(X_test : n_samples_test , n_features_total) = ", X_test.shape)
    print("[n_features, ...] = ", n_features_list)
    Y_train = Y_train.astype("float32")
    Y_test = Y_test.astype("float32")
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    assert np.all(Y_test.mean(0) < 1e-5)
    assert np.all(Y_test.std(0) - 1 < 1e-5)
    assert np.all(Y_train.mean(0) < 1e-5)
    assert np.all(Y_train.std(0) - 1 < 1e-5)
    train_runs, _ = get_train_test_runs(subject_idx, ses)
    run_onsets = get_run_onsets(train_runs)
    return X_train, X_test, Y_train, Y_test, feature_names, n_features_list, train_runs, run_onsets


def get_responses_ses(subject_idx, ses, model, basemodel):
    if model=='headmotion':
        Y_train, Y_test = responses_train_test(subject_idx, ses)
    elif model=='gpt_nprandom':
        bold_res_file=f"{DATA_DIR}{SUBJECTS_ALL[subject_idx]}/{SUBJECTS_ALL[subject_idx]}_ses-{ses}_task-dialogue_bold_res_headmotion.hdf"
        Y_train=load_hdf5_array(bold_res_file, key='Y_train')
        Y_test=load_hdf5_array(bold_res_file, key='Y_test')
    else:
        bold_res_file=f"{DATA_DIR}{SUBJECTS_ALL[subject_idx]}/{SUBJECTS_ALL[subject_idx]}_ses-{ses}_task-dialogue_bold_res_headmotion_gpt_nprandom.hdf"
        Y_train=load_hdf5_array(bold_res_file, key='Y_train')
        Y_test=load_hdf5_array(bold_res_file, key='Y_test')
    
    train_runs, test_runs = get_train_test_runs(subject_idx, ses)
    assert Y_test.shape[0] == N_SCANS*len(test_runs), print(Y_test.shape)
    return Y_train, Y_test


def get_features_ses(subject_idx, ses, model, layer=-1, cl=1):
    subject=SUBJECTS_ALL[subject_idx]
    tmp_FEATURES_DIR=f"{FEATURES_DIR}{subject}/"
    train_runs, test_runs=get_train_test_runs(subject_idx, ses)
    print(f"model: {model}")
    if 'gpt_nprandom' in model:
        X=load_hdf5_array(f"{tmp_FEATURES_DIR}/{subject}_gpt_nprandom.hdf")
    elif 'chatgpt_unified' in model:
        X=load_hdf5_array(f"{tmp_FEATURES_DIR}/{subject}_chatgptneox_unified_cl-{cl}.hdf")
    elif 'gpt_unified' in model:
        X=load_hdf5_array(f"{tmp_FEATURES_DIR}/{subject}_gptneox_unified_cl-{cl}.hdf")
    elif 'chatgpt' in model:
        X=load_hdf5_array(f"{tmp_FEATURES_DIR}/{subject}_chatgptneox_cl-{cl}.hdf")
    elif 'gpt' in model:
        X=load_hdf5_array(f"{tmp_FEATURES_DIR}/{subject}_gptneox_cl-{cl}.hdf")
    else:
        X=load_hdf5_array(f"{tmp_FEATURES_DIR}/{subject}_headmotion.hdf")

    features=MODEL_FEATURE_MATRIX[model]
    n_features_list=[]
    Xs_train = []
    Xs_test = []
    for feature in features:
        if layer > -1 and "gpt" in feature:
            feature=f"{feature}_layer-{layer}"

        X_train_3d=X[feature][train_runs-1]
        X_test_3d=X[feature][test_runs-1]
        Xi_train_list=[]
        Xi_test_list=[]
        for run_i in range(len(train_runs)):
            Xi_train_list.append(X_train_3d[run_i])

        Xi_train=np.concatenate(Xi_train_list,0)
        for run_i in range(len(test_runs)):
            Xi_test_list.append(X_test_3d[run_i])

        Xi_test=np.concatenate(Xi_test_list,0)
        Xs_train.append(Xi_train)
        Xs_test.append(Xi_test)
        n_features_list.append(X[feature].shape[2])

    X_train = np.concatenate(Xs_train, 1)
    X_test = np.concatenate(Xs_test, 1)
    return X_train, X_test, n_features_list


def responses_train_test(subject_idx, ses):
    train_runs, test_runs = get_train_test_runs(subject_idx, ses)
    Y_all=load_hdf5_array(f"{DATA_DIR}{SUBJECTS_ALL[subject_idx]}/{SUBJECTS_ALL[subject_idx]}_task-dialogue_bold.hdf", key='bold')
    Y_train_3d=Y_all[train_runs-1]
    Y_test_3d=Y_all[test_runs-1]
    Y_train_list=[]
    Y_test_list=[]
    for run_i in range(len(train_runs)):
        Y_train_list.append(Y_train_3d[run_i])

    Y_train=np.concatenate(Y_train_list,0)
    for run_i in range(len(test_runs)):
        Y_test_list.append(Y_test_3d[run_i])

    Y_test=np.concatenate(Y_test_list,0)
    return Y_train, Y_test


def get_train_test_runs(subject_idx, ses):
    test_runs=TESTSET_RUNS[subject_idx][ses]
    train_runs=np.setdiff1d(np.arange(1, TOTAL_RUNS[subject_idx]+1), test_runs)
    return train_runs, test_runs


def get_ready_statsfile(subject_idx, model, basemodel, layer=-1, cl=1):
    subject=SUBJECTS_ALL[subject_idx]
    if layer < 0:
        postfix=''
    else:
        postfix=f"_layer-{layer}_cl-{cl}"

    name = '_r_score'
    weight_corr_svoxels = f"{RESULTS_DIR}{subject}_{basemodel}_{model}{postfix}_weight_corr{name}_svoxels.hdf"
    weight_corr_tvoxels = f"{RESULTS_DIR}{subject}_{basemodel}_{model}{postfix}_weight_corr{name}_tvoxels.hdf"
    return subject, weight_corr_svoxels, weight_corr_tvoxels

def get_ready_statsfile_context(subject_idx, basemodel, prod_cl, comp_cl, layer=-1):
    subject=SUBJECTS_ALL[subject_idx]
    if layer < 0:
        postfix=''
    else:
        postfix=f"_layer-{layer}_prod_cl-{prod_cl}_vs_comp_cl-{comp_cl}"

    name = '_r_score'
    weight_corr_svoxels = f"{RESULTS_DIR}{subject}_{basemodel}_{postfix}_weight_corr{name}_svoxels.hdf"
    weight_corr_tvoxels = f"{RESULTS_DIR}{subject}_{basemodel}_{postfix}_weight_corr{name}_tvoxels.hdf"
    return subject, weight_corr_svoxels, weight_corr_tvoxels

def get_ready_statsfile_layer(subject_idx, basemodel, prod_layer, comp_layer, cl=1):
    subject=SUBJECTS_ALL[subject_idx]
    postfix=f"_prod_layer-{prod_layer}_vs_comp_layer-{comp_layer}_cl-{cl}"

    name = '_r_score'
    weight_corr_svoxels = f"{RESULTS_DIR}{subject}_{basemodel}_{postfix}_weight_corr{name}_svoxels.hdf"
    weight_corr_tvoxels = f"{RESULTS_DIR}{subject}_{basemodel}_{postfix}_weight_corr{name}_tvoxels.hdf"
    return subject, weight_corr_svoxels, weight_corr_tvoxels






##########################################################################################################################################################################
##########################################################################################################################################################################
### individual pca
##########################################################################################################################################################################
##########################################################################################################################################################################
def get_ready_indiv_pcafile(subject, model, basemodel, layer, cl):
    if layer < 0:
        postfix=''
    else:
        postfix=f"_layer-{layer}_cl-{cl}"

    indiv_weight_prod_file = f"{RESULTS_DIR}{subject}_weights_{basemodel}_{model}{postfix}_prod.hdf"
    indiv_weight_comp_file = f"{RESULTS_DIR}{subject}_weights_{basemodel}_{model}{postfix}_comp.hdf"
    pca_prod_file = f"{RESULTS_DIR}{subject}_pca_{basemodel}_{model}{postfix}_prod.hdf"
    pca_comp_file = f"{RESULTS_DIR}{subject}_pca_{basemodel}_{model}{postfix}_comp.hdf"
    boot_prod_file = f"{RESULTS_DIR}{subject}_pca_bootstrap_{basemodel}_{model}{postfix}_prod.hdf"
    boot_comp_file = f"{RESULTS_DIR}{subject}_pca_bootstrap_{basemodel}_{model}{postfix}_comp.hdf"
    pcloads_prod_file = f"{RESULTS_DIR}{subject}_pcloads_{basemodel}_{model}{postfix}_prod.hdf"
    pcloads_comp_file = f"{RESULTS_DIR}{subject}_pcloads_{basemodel}_{model}{postfix}_comp.hdf"
    return indiv_weight_prod_file, indiv_weight_comp_file, pca_prod_file, pca_comp_file, boot_prod_file, boot_comp_file, pcloads_prod_file, pcloads_comp_file

def get_features_list_ses(subject_idx, ses, model, layer=-1):
    subject=SUBJECTS_ALL[subject_idx]
    tmp_FEATURES_DIR=f"{FEATURES_DIR}{subject}/"
    train_runs, test_runs=get_train_test_runs(subject_idx, ses)
    X=load_hdf5_array(f"{tmp_FEATURES_DIR}{subject}_allrun.hdf")
    features=MODEL_FEATURE_MATRIX[model]
    n_features_list=[]
    Xs_train = []
    Xs_test = []
    for feature in features:
        if layer > -1 and "gptneox" in feature:
            feature=f"{feature}_layer-{layer}"

        X_train_3d=X[feature][train_runs-1]
        X_test_3d=X[feature][test_runs-1]
        Xi_train_list=[]
        Xi_test_list=[]
        for run_i in range(len(train_runs)):
            Xi_train_list.append(X_train_3d[run_i])

        Xi_train=np.concatenate(Xi_train_list,0)
        for run_i in range(len(test_runs)):
            Xi_test_list.append(X_test_3d[run_i])

        Xi_test=np.concatenate(Xi_test_list,0)
        Xs_train.append(Xi_train)
        Xs_test.append(Xi_test)
    return Xs_train, Xs_test, n_features_list


def get_run_onsets(train_runs):
    run_onsets=[]
    for rr in range(len(train_runs)):
        run_onsets.append(N_SCANS*rr)
    return run_onsets



def get_deltas(subject_idx, ses, model, layer=-1):
    subject = SUBJECTS_ALL[subject_idx]
    if layer < 0:
        output = f"{RESULTS_DIR}{subject}_{basemodel}_{model}_ses-{ses}_himalaya.hdf"
    else:
        output = f"{RESULTS_DIR}{subject}_{basemodel}_{model}_ses-{ses}_layer-{layer}_himalaya.hdf"
    deltas=load_hdf5_array(output, key="deltas")
    return deltas

def load_hdf5_array(file_name, key=None, slice=slice(0, None)):
    with h5py.File(file_name, mode='r') as hf:
        if key is None:
            data = dict()
            for k in hf.keys():
                data[k] = hf[k][slice]
            return data
        else:
            return hf[key][slice]

def save_table_file(filename, filedict):
    hf = tables.open_file(filename, mode="w", title="save_file")
    for vname, var in filedict.items():
        hf.create_array("/", vname, var)
    hf.close()


def load_strings_hdf_array(file_name, key=None, slice=slice(0, None)):
    with h5py.File(file_name, mode='r') as hf:
        strings_array = hf['data'][:]
        strings_array = np.array([x.decode('utf-8') for x in strings_array])

    return strings_array

def save_strings_hdf_file(filename, string_data):
    """Saves the variables in a hdf5 file at [filename].
    """
    with h5py.File(filename, "w") as file:
        file.create_dataset("data", data=string_data.astype('O'))
