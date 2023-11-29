function ridge_session_average_mainfeatures(ID)
% ridge_session_average_mainfeatures.m
% 
% Average prediction performance across sessions
%
% Inputs: ID     : subject ID (e.g. 'sub-OSU01')
%
% Outputs:  'RidgeResults_CHATGPTNEOX_sub-OSU01_MainFeatures_FDRcorr_mean.mat'
%			'RidgeResults_CHATGPTNEOX_sub-OSU01_MainFeatures_FDRcorr_mean.nii'
%


%%
PRM=load_parameters_proj;
IND=load_parameters_ind(ID, PRM);


postfix='MainFeatures';


% Load results from each cross-validation
for ses = 1:IND.ses_num    
	load([PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '_' postfix '.mat'], 'Result');        
    ccs(ses,:) = Result.ccs_fdr; fdrthr(ses) = Result.fdrthr;
end


% Average prediction performance after Fisher Z transformation
z_ccs = atanh(ccs); mean_zccs = nanmean(z_ccs); mean_ccs = tanh(mean_zccs);


% Median FDR threshold value 
med_fdrthr = median(fdrthr);
mean_ccs(find(mean_ccs < med_fdrthr)) = 0;
svoxels = find(mean_ccs > med_fdrthr);


% Mapping from 1d Data to 3d .nii data
mapidx=1;
if mapidx
    Y = NaN(prod(IND.datasize),1);
    for ii=1:length(IND.tvoxels)
        Y(IND.tvoxels(ii))= mean_ccs(ii);
    end
    vol = reshape(Y, IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm;
    MRIwrite(V, [PRM.SaveDir IND.file_prefix '_' postfix '_FDRcorr_mean.nii']);    
end


mean_Result.med_fdrthr = med_fdrthr;
mean_Result.mean_ccs = mean_ccs;
mean_Result.svoxels = svoxels;
 save([PRM.SaveDir IND.file_prefix '_' postfix '_FDRcorr_mean.mat' ], 'mean_Result', '-v7.3');

end
