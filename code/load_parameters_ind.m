function IND = load_parameters_ind(ID, PRM)
% load_parameters_ind.m
% 
% Inputs:	ID		: subject ID (e.g. 'sub-OSU01')
%
%			PRM		: parameters of project


%%
file_prefix=['RidgeResults_' PRM.LLM '_' ID];

DataDir = [ PRM.ProjectDir '/derivative/preprocessed_data/' ID '/'];
ROIDir = [ PRM.ProjectDir '/derivative/fsROI/' ID '/' ];


% Load target voxels in cerebral cortex estimated by freesurfer
load([ ROIDir '/vset_info.mat' ]);
ROI = vset_info.IDs;
vset = ROI(1);
datasize = [96 96 72];
voxelSetForm = [ ROIDir '/vset_%03d.mat'];
load(sprintf(voxelSetForm,vset),'tvoxels');
nvoxels = vset_info.nvoxels(1);


% individual EPI space
RefEPI = [DataDir 'target_' ID '.nii'];


% The number of fMRI sessions performed
if strcmp(ID, 'sub-OSU02') || strcmp(ID, 'sub-OSU03') || strcmp(ID, 'sub-OSU05')  
    ses_num=3;
else
    ses_num=4;
end


IND.file_prefix=file_prefix;
IND.DataDir=DataDir;
IND.ROIDir=ROIDir;
IND.ROI=ROI;
IND.vset=vset;
IND.datasize=datasize;
IND.tvoxels=tvoxels;
IND.nvoxels=nvoxels;
IND.RefEPI=RefEPI;
IND.ses_num=ses_num;

end

