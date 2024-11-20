import cortex
import numpy as np
import tables
import h5py
import sys
# add path to config.py
sys.path.append('/home/projects/dialogue/code/')
from config import FSROI_DIR, DATASIZE, NVOXELS

#############################################################
#############################################################
#############################################################
#### get_volume
#############################################################
#############################################################
#############################################################
def get_volume_neg2nan_ALLSUBJ_base(subject_idx, subject, data_file, key='scores', idx=None, cmap='hot', vmin=0, vmax=0.05):
    data = get_neg2nan_ALLSUBJ_base(data_file, subject_idx, get_tvoxels(subject), key=key, idx=idx)
    volume = cortex.Volume(data, subject, subject, cmap=cmap, vmin=vmin, vmax=vmax)
    return volume

def get_volume_context_neg2nan_ALLSUBJ(subject_idx, subject, layer_idx, cl_idx, test_idx, data_file, key='scores', idx=None, cmap='hot', vmin=0, vmax=0.25, thr=0):
    data = get_context_neg2nan_ALLSUBJ(data_file, subject_idx, layer_idx, cl_idx, test_idx, get_tvoxels(subject), key=key, idx=idx)
    data[data<thr]=np.nan
    volume = cortex.Volume(data, subject, subject, cmap=cmap, vmin=vmin, vmax=vmax)
    return volume

def get_volume_cross_context_neg2nan_ALLSUBJ(subject_idx, subject, layer_idx, cl_idx, data_file, model_file, key='scores', idx=None, cmap='hot', vmin=0, vmax=0.25, thr=0):
    test_idx=2
    data = get_context_neg2nan_ALLSUBJ(data_file, subject_idx, layer_idx, cl_idx, test_idx, get_tvoxels(subject))
    data[data<thr]=np.nan

    test_idx=1
    intra_data = get_context_neg2nan_ALLSUBJ(model_file, subject_idx, layer_idx, cl_idx, test_idx, get_tvoxels(subject))
    data[np.isnan(intra_data)]=np.nan
    volume = cortex.Volume(data, subject, subject, cmap=cmap, vmin=vmin, vmax=vmax)
    return volume

def get_volume2D_bestvp_neg2nan_ALLSUBJ(subject_idx, subject, layer_idx, cl_idx, data_file, model_file, cmap='BROYG_2D', vmin=0.05, vmax=0.2, thr=0.05):
    data = get_bestvp_neg2nan_ALLSUBJ(data_file, subject_idx, layer_idx, cl_idx, get_tvoxels(subject))
    test_idx=1
    score_data = get_context_neg2nan_ALLSUBJ(model_file, subject_idx, layer_idx, cl_idx, test_idx, get_tvoxels(subject))
    data[np.isnan(score_data)]=np.nan
    data[data==3]=5
    volume = cortex.Volume2D(data, score_data, subject, subject, cmap=cmap, vmin=0.5, vmax=5.5, vmin2=vmin, vmax2=vmax)
    return volume

def get_volume2D_context_weightcorr_bothsignificant(subject, subject_idx, layer_idx, cl_idx, weight_file, intra_file, cross_file, cmap='BuWtRd_black_2D', vmin=-0.4, vmax=0.4, score_min=0.05, score_max=0.2):
    data1 = get_weightcorr_context_fullrange(subject, subject_idx, layer_idx, cl_idx, weight_file, get_tvoxels(subject))
    test_idx=1
    intra_data = get_context_neg2nan_ALLSUBJ(intra_file, subject_idx, layer_idx, cl_idx, test_idx, get_tvoxels(subject), key='scores')
    test_idx=2
    cross_data = get_context_neg2nan_ALLSUBJ(cross_file, subject_idx, layer_idx, cl_idx, test_idx, get_tvoxels(subject), key='scores')
    data1[intra_data < score_min] = np.nan
    data1[cross_data < score_min] = np.nan
    volume = cortex.Volume2D(data1, cross_data, subject, subject, vmin=vmin, vmax=vmax, vmin2=score_min, vmax2=score_max, cmap=cmap)
    return volume

def get_volume_weightcorr_VP_mask(subject, subject_idx, layer_idx, cl_idx, modality_idx, wc_file, vp_file, model_file, cmap='BrBG', vmin=-0.6, vmax=0.6):
    wc_data = get_weightcorr_context_fullrange(subject, subject_idx, layer_idx, cl_idx, wc_file, get_tvoxels(subject))
    test_idx=1
    score_data = get_context_neg2nan_ALLSUBJ(model_file, subject_idx, layer_idx, cl_idx, test_idx, get_tvoxels(subject), key='scores')
    wc_data[score_data < 0.05] = np.nan

    vp_data = get_bestvp_neg2nan_ALLSUBJ(vp_file, subject_idx, layer_idx, cl_idx, get_tvoxels(subject))
    indices_prod = np.where(vp_data==1)
    indices_comp = np.where(vp_data==2)
    indices_bimodal = np.where(vp_data==3)
    print(f"# of prod, comp, bimodal voxels: {len(indices_prod[0]), len(indices_comp[0]), len(indices_bimodal[0])}")
    tmp_wc_data = np.full(wc_data.shape, np.nan)
    if modality_idx == 1:
        tmp_wc_data[indices_prod] = wc_data[indices_prod]
    elif modality_idx == 2:
        tmp_wc_data[indices_comp] = wc_data[indices_comp]
    elif modality_idx == 3:
        tmp_wc_data[indices_bimodal] = wc_data[indices_bimodal]
    
    volume = cortex.Volume2D(tmp_wc_data, score_data, subject, subject, vmin=vmin, vmax=vmax, vmin2=0.05, vmax2=0.2, cmap=cmap)
    return volume

def get_volume2D_context_all_weightcorr_significant_unified_diff(subject, subject_idx, layer_idx, cl_idx, weight_file, separate_file, unified_file, cmap='BuWtRd_black_2D', vmin=-0.4, vmax=0.4, score_min=0.05, score_max=0.3):
    test_idx=1
    data1_prod = get_weightcorr_context_significant_unified(subject, subject_idx, layer_idx, cl_idx, 0, weight_file, get_tvoxels(subject))
    data1_comp = get_weightcorr_context_significant_unified(subject, subject_idx, layer_idx, cl_idx, 1, weight_file, get_tvoxels(subject))
    data1 = np.arctanh(data1_comp) - np.arctanh(data1_prod)
    separate_data = get_context_neg2nan_ALLSUBJ(separate_file, subject_idx, layer_idx, cl_idx, test_idx, get_tvoxels(subject), key='scores')
    unified_data = get_context_neg2nan_ALLSUBJ(unified_file, subject_idx, layer_idx, cl_idx, test_idx, get_tvoxels(subject), key='scores')
    data1[np.isnan(separate_data)] = np.nan
    data1[np.isnan(unified_data)] = np.nan
    unified_data[unified_data < score_min] = np.nan
    volume = cortex.Volume2D(data1, unified_data, subject, subject, vmin=vmin, vmax=vmax, vmin2=score_min, vmax2=score_max, cmap=cmap)
    return volume

#############################################################
#############################################################
#############################################################
#### get_data
#############################################################
#############################################################
#############################################################
def get_neg2nan_ALLSUBJ_base(hdf_file, subject_idx, tvoxels, key='scores', idx=None):
    with tables.open_file(hdf_file, "r") as hf:
        data=np.nan_to_num(hf.root.scores_all.read())
        
    tmp_data=data[subject_idx, :]
    Y=np.empty(np.prod(DATASIZE))
    Y[:]=np.nan
    for vv in range(tvoxels.shape[0]):
        tmpv=int(tvoxels[vv])
        Y[tmpv-1]=tmp_data[vv]
        
    Y=np.where(Y<= 0, np.nan, Y)
    Y2=np.reshape(Y, DATASIZE, order="F")
    Y2=np.transpose(Y2, (2,1,0))
    return Y2

def get_context_neg2nan_ALLSUBJ(hdf_file, subject_idx, layer_idx, cl_idx, test_idx, tvoxels, key='scores', idx=None):
    if key=='scores':
        with tables.open_file(hdf_file, "r") as hf:
            if test_idx == 1:
                data=np.nan_to_num(hf.root.scores_all.read())
            elif test_idx == 2:
                data=np.nan_to_num(hf.root.scores_all.read())

        data=np.squeeze(data[layer_idx, cl_idx, subject_idx, :])
    elif key=='scores_split':
        with tables.open_file(hdf_file, "r") as hf:
            if test_idx == 1:
                data=np.nan_to_num(hf.root.scores_split.read())
            elif test_idx == 2:
                data=np.nan_to_num(hf.root.scores_cm_split.read())
        
        data=np.squeeze(data[layer_idx, cl_idx, idx, subject_idx, :])

    Y=np.empty(np.prod(DATASIZE))
    Y[:]=np.nan
    for vv in range(tvoxels.shape[0]):
        tmpv=int(tvoxels[vv])
        Y[tmpv-1]=data[vv]
        
    Y=np.where(Y<= 0, np.nan, Y)
    Y2=np.reshape(Y, DATASIZE, order="F")
    Y2=np.transpose(Y2, (2,1,0))
    return Y2

def get_bestvp_neg2nan_ALLSUBJ(hdf_file, subject_idx, layer_idx, cl_idx, tvoxels):
    with tables.open_file(hdf_file, "r") as hf:
        data=np.nan_to_num(hf.root.VRGB_all.read())
        data=np.squeeze(data[layer_idx, cl_idx, subject_idx, :])
    
    Y=np.empty(np.prod(DATASIZE))
    Y[:]=np.nan
    for vv in range(tvoxels.shape[0]):
        tmpv=int(tvoxels[vv])
        Y[tmpv-1]=data[vv]
        
    Y=np.where(Y < 1, np.nan, Y)
    Y2=np.reshape(Y, DATASIZE, order="F")
    Y2=np.transpose(Y2, (2,1,0))
    return Y2

def get_neg2nan_ALLSUBJ_base_contextual(hdf_file, subject_idx, cl_idx, tvoxels, key='scores_all', idx=None):
    with tables.open_file(hdf_file, "r") as hf:
        data=np.nan_to_num(hf.root.scores_all.read())

    tmp_data=np.squeeze(data[cl_idx, subject_idx, :])
    Y=np.empty(np.prod(DATASIZE))
    Y[:]=np.nan
    for vv in range(tvoxels.shape[0]):
        tmpv=int(tvoxels[vv])
        Y[tmpv-1]=tmp_data[vv]
        
    Y=np.where(Y<= 0, np.nan, Y)
    print(f"sig. voxels: {np.sum(~np.isnan(Y))}")
    Y2=np.reshape(Y, DATASIZE, order="F")
    Y2=np.transpose(Y2, (2,1,0))
    return Y2

def get_weightcorr_context_fullrange(subject, subject_idx, layer_idx, cl_idx, hdf_file, tvoxels):
    with tables.open_file(hdf_file, "r") as hf:
        data=hf.root.weightcorr_all.read()

    data=np.squeeze(data[layer_idx, cl_idx, subject_idx, :])
    Y=np.full(np.prod(DATASIZE), np.nan)
    for vv in range(tvoxels.shape[0]):
        tmpv=int(tvoxels[vv])
        Y[tmpv-1]=data[vv]

    Y2=np.reshape(Y, DATASIZE, order="F")
    Y2=np.transpose(Y2, (2,1,0))
    return Y2

def get_weightcorr_context_significant_unified(subject, subject_idx, layer_idx, cl_idx, modality_idx, hdf_file, tvoxels):
    with tables.open_file(hdf_file, "r") as hf:
        if modality_idx == 0:
            data=hf.root.sig_prodcorr_all.read()
        elif modality_idx == 1:
            data=hf.root.sig_compcorr_all.read()

    data=np.squeeze(data[layer_idx, cl_idx, subject_idx, :])
    Y=np.full(np.prod(DATASIZE), np.nan)
    for vv in range(tvoxels.shape[0]):
        tmpv=int(tvoxels[vv])
        Y[tmpv-1]=data[vv]

    Y2=np.reshape(Y, DATASIZE, order="F")
    Y2=np.transpose(Y2, (2,1,0))
    return Y2

#############################################################
#### others
#############################################################
def load_hdf5_array(file_name, key=None, slice=slice(0, None)):
    with h5py.File(file_name, mode='r') as hf:
        if key is None:
            data = dict()
            for k in hf.keys():
                data[k] = hf[k][slice]
            return data
        else:
            return hf[key][slice]

def get_tvoxels(subject):
    tvoxels_file=f"{FSROI_DIR}{subject}_tvoxels.mat"
    tvoxels=load_hdf5_array(tvoxels_file, 'tvoxels')
    tvoxels=tvoxels.T
    return tvoxels

