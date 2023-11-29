function ridge_fdrcorr(ID, modality)
% ridge_fdrcorr.m
%
% FDR correction for multiple comparison (Benjamini & Hochberg, 1995)
%
% Inputs: ID      : subject ID (e.g. 'sub-OSU01')
%
%         modality: 1: same, 2: cross
%
% Outputs:  'RidgeResults_sub-OSU01_ses-1.mat'
%


%%
PRM=load_parameters_proj;
IND=load_parameters_ind(ID, PRM);


if modality == 1
    name = 'same';
else
    name = 'cross';
end


% Random seed
rng(1234, 'twister');


% FDR correction for each session
for ses = 1:IND.ses_num
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    disp(['%%%%%  Cross-validation: ' num2str(ses) '/' num2str(IND.ses_num) ]);


    % Load cross-validation result file
    load([PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '.mat'], 'Result');


    % Load prediction performance
    if modality == 1
        ccs = Result.ccs_same;
    else
        ccs = Result.ccs_cross;
    end


    % Make random correlation coefficient histogram
    for ii = 1:IND.nvoxels;
    	A = normrnd(0,1,size(Result.resp,1),1); B = normrnd(0,1,size(Result.resp,1),1);
    	rccs(ii) = corr2(A,B);
    end


    % Calculate P values for each voxel
    PX = [];
    for ii = 1:length(ccs)
    	x = find(rccs>ccs(ii)); px = length(x)/length(ccs); PX(ii) = px;
    end


    % Perform FDR correction
    [PXsorted PXind] = sort(PX, 'ascend');
    FDRthr = PRM.Q*[1:length(ccs)]/length(ccs);
    Diff = PXsorted - FDRthr;
    thrInd = PXind(max(find(Diff<0))); thrP = PX(thrInd); thrCCs = ccs(thrInd);
    disp(['Threshold ccs = ' num2str(thrCCs)])


    % Make prediction performance 0 under threshold
    ccs(find(ccs < thrCCs)) = 0;


    % Mapping from 1d Data to 3d .nii data 
    mapidx=0;
    if mapidx
        Y = zeros(prod(IND.datasize),1);
        for ii=1:length(IND.tvoxels)
            Y(IND.tvoxels(ii))= ccs(ii);
        end
        vol = reshape(Y,IND.datasize);  vol_perm = permute(vol, [2,1,3]);  V = MRIread(IND.RefEPI); V.vol = vol_perm;
        MRIwrite(V, [PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '_' name '_FDRcorr.nii']);
    end


    % Save results
    if modality == 1
        Result.ccs_same_fdr = ccs; Result.same_fdrthr = thrCCs;
    else
        Result.ccs_cross_fdr = ccs; Result.cross_fdrthr = thrCCs;
    end
    save([PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '.mat'], 'Result', '-v7.3'); 
        
end

