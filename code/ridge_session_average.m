function ridge_session_average(ID)
% ridge_session_average.m
%
% Average prediction performance across sessions
%
% Inputs: ID     : subject ID (e.g. 'sub-OSU01')
%
% Outputs:  'RidgeResults_sub-OSU01_same_FDRcorr_mean.nii'
%           'RidgeResults_sub-OSU01_cross_FDRcorr_mean.nii'
%           'RidgeResults_sub-OSU01_FDRcorr_mean.mat'


%%
PRM=load_parameters_proj;
IND=load_parameters_ind(ID, PRM);

for modality = 1:2
    if modality == 1
        name = 'same';
    else
        name = 'cross';
    end


    % Extract results from each session
    for ses = 1:IND.ses_num    
    	load([PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '.mat'], 'Result');
        if modality == 1
            ccs(ses,:) = Result.ccs_same_fdr; fdrthr(ses) = Result.same_fdrthr;
        else
            ccs(ses,:) = Result.ccs_cross_fdr;  fdrthr(ses) = Result.cross_fdrthr;
        end
    end


    % Average prediction performance after Fisher Z transformation
    z_ccs = atanh(ccs); mean_zccs = nanmean(z_ccs); mean_ccs = tanh(mean_zccs);


    % Calculate median FDR threshold value across sessions
    fdrthr_med = median(fdrthr);


    % Extract significant prediction performance and the voxels
    mean_ccs(find(mean_ccs < fdrthr_med)) = 0;
    svoxels = find(mean_ccs > fdrthr_med);


    % Mapping from 1d Data to 3d .nii data
    mapidx=1;
    if mapidx
        Y = NaN(prod(IND.datasize),1);
        for ii=1:length(IND.tvoxels)
            Y(IND.tvoxels(ii))= mean_ccs(ii);
        end
        vol = reshape(Y, IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm;
        MRIwrite(V, [PRM.SaveDir IND.file_prefix '_' name '_FDRcorr_mean.nii']);    
    end

    if modality == 1
        mean_Result.med_same_fdrthr = fdrthr_med;
        mean_Result.mean_ccs_same = mean_ccs;
        mean_Result.svoxels = svoxels;
    else
        mean_Result.med_cross_fdrthr = fdrthr_med;
        mean_Result.mean_ccs_cross = mean_ccs;
        mean_Result.svoxels_cross = svoxels;
    end
end
 
save([PRM.SaveDir IND.file_prefix '_FDRcorr_mean.mat' ], 'mean_Result', '-v7.3');

end
