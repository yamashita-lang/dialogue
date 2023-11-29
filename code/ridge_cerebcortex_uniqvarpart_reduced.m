function ridge_cerebcortex_uniqvarpart_reduced(ID)
% ridge_cerebcortex_uniqvarpart_reduced.m
% 
% Variance partitioning for LLM production/comprehension
%
% Inputs: ID     : subject ID (e.g. 'sub-OSU01')
%
% Outputs:  'RidgeResults_CHATGPTNEOX_sub-OSU01_SingleFeature_Intersection.nii'
%           'RidgeResults_CHATGPTNEOX_sub-OSU01_SingleFeature_Prod_RC.nii'
%           'RidgeResults_CHATGPTNEOX_sub-OSU01_SingleFeature_Comp_RC.nii'
%           'RidgeResults_CHATGPTNEOX_sub-OSU01_SingleFeature_VarPart.mat'
%


%%
PRM=load_parameters_proj;
IND=load_parameters_ind(ID, PRM);

main_idx=1;


Features_used=[];
ii = 0;
for modality = [1 2]
    for mm = PRM.Features_index
        ii = ii + 1;
        if mm == length(PRM.Features) && modality == 2
            ; % skipping repetition for MOT
        else
            Method=char(PRM.Features{mm});
            Features_used=[Features_used char(num2str(ii))];
        end    
    end
end


%% A U B
disp('Data from union (production, comprehension)')

% Concatenate LLM production comprehension results across sessions
resp=[]; presp=[];
for ses = 1:IND.ses_num
    full_result=[ PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '.mat' ]; load(full_result); tmp_resp=Result.resp; clear Result;

	ses_result=[ PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '_MainFeatures.mat' ]; 
	disp(['loading ' ses_result ]); load(ses_result);
	
	if ses == 1
		resp = tmp_resp; prespU = Result.presp;
	else
		resp = [resp; tmp_resp ]; prespU = [prespU; Result.presp ];
	end
end
clear Result;


% Significantly predicted voxels from mean results across sessions
load([ PRM.SaveDir IND.file_prefix '_MainFeatures_FDRcorr_mean.mat' ]); ;
resp = resp(:, mean_Result.svoxels); 
prespU = prespU(:, mean_Result.svoxels);


% Variance partitioning
Tvar = sum(( resp -  mean(resp)).^2);
RU = 1 - sum((prespU - resp).^2) ./ Tvar;


% Each model
disp('Data from each production, comprehension')

	
% Concatenate LLM production comprehension results across sessions
for ses = 1:IND.ses_num
    disp('===================================================================')
    disp(['     Cross-validation: ' num2str(ses) '/' num2str(IND.ses_num) ]);
    disp('===================================================================')
    load([PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '_SingleFeature_1.mat']);
    if ses == 1
		prespProd = Result.presp(:, mean_Result.svoxels);
	else
		prespProd = [prespProd; Result.presp(:, mean_Result.svoxels) ];
	end
end
clear Result;

RProd = 1 - sum((prespProd - resp).^2) ./ Tvar;


for ses = 1:IND.ses_num
    disp('===================================================================')
    disp(['     Cross-validation: ' num2str(ses) '/' num2str(IND.ses_num) ]);
    disp('===================================================================')
    load([PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '_SingleFeature_7.mat']);
    if ses == 1
		prespComp = Result.presp(:, mean_Result.svoxels);
	else
		prespComp = [prespComp; Result.presp(:, mean_Result.svoxels) ];
	end
end
clear Reslt;

RComp = 1 - sum((prespComp - resp).^2) ./ Tvar;

% Relative complement
R_Prod_RC = RU - RComp;
R_Comp_RC = RU - RProd;


% Intersection
R_Intersection = RProd + RComp - RU;


% Calculate significance
PV_Intersection = R_Intersection;
PV_Prod_RC = R_Prod_RC;
PV_Comp_RC = R_Comp_RC;


% Transform from R-squared to R
R_Intersection = sqrt(PV_Intersection); Result.R_Intersection = R_Intersection;
R_Prod_RC = sqrt(PV_Prod_RC); Result.R_Prod_RC = R_Prod_RC;
R_Comp_RC = sqrt(PV_Comp_RC); Result.R_Comp_RC = R_Comp_RC;

disp(['Mean R_Intersection: ' num2str(nanmean(R_Intersection)) ]);
disp(['Min R_Intersection: ' num2str(min(R_Intersection)) ]);
disp(['Max R_Intersection: ' num2str(max(R_Intersection)) ]);

disp(['Mean R_Prod_RC: ' num2str(nanmean(R_Prod_RC)) ]);
disp(['Min R_Prod_RC: ' num2str(min(R_Prod_RC)) ]);
disp(['Max R_Prod_RC: ' num2str(max(R_Prod_RC)) ]);

disp(['Mean R_Comp_RC: ' num2str(nanmean(R_Comp_RC)) ]);
disp(['Min R_Comp_RC: ' num2str(min(R_Comp_RC)) ]);
disp(['Max R_Comp_RC: ' num2str(max(R_Comp_RC)) ]);


% Mapping from 1d Data to 3d .nii data 

% Result for intersection
image_result_file=[PRM.SaveDir IND.file_prefix '_SingleFeature_Intersection.nii' ];
R_Intersection_ori = NaN(size(IND.tvoxels));
for vv = 1:length(R_Intersection)
    R_Intersection_ori(mean_Result.svoxels(vv)) = R_Intersection(vv);
end
Result.R_Intersection_ori=R_Intersection_ori;


Y = NaN(prod(IND.datasize),1);
for ii=1:length(IND.tvoxels)
    Y(IND.tvoxels(ii))= R_Intersection_ori(ii);
end
vol = reshape(Y, IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm;
MRIwrite(V, image_result_file);


% Result for production
image_result_file=[PRM.SaveDir IND.file_prefix '_SingleFeature_Prod_RC.nii' ]; 
R_Prod_RC_ori = NaN(size(IND.tvoxels));
for vv = 1:length(R_Prod_RC)
    R_Prod_RC_ori(mean_Result.svoxels(vv)) = R_Prod_RC(vv);
end

Result.R_Prod_RC_ori=R_Prod_RC_ori;
Y = NaN(prod(IND.datasize),1);
for ii=1:length(IND.tvoxels)
    Y(IND.tvoxels(ii))= R_Prod_RC_ori(ii);
end
vol = reshape(Y, IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm;
MRIwrite(V, image_result_file);


% Result for comprehension
image_result_file=[PRM.SaveDir IND.file_prefix '_SingleFeature_Comp_RC.nii' ]; 
R_Comp_RC_ori = NaN(size(IND.tvoxels));
for vv = 1:length(R_Comp_RC)
    R_Comp_RC_ori(mean_Result.svoxels(vv)) = R_Comp_RC(vv);
end

Result.R_Comp_RC_ori=R_Comp_RC_ori;
Y = NaN(prod(IND.datasize),1);
for ii=1:length(IND.tvoxels)
    Y(IND.tvoxels(ii))= R_Comp_RC_ori(ii);
end
vol = reshape(Y, IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm;
MRIwrite(V, image_result_file);


save([ PRM.SaveDir IND.file_prefix '_SingleFeature_VarPart.mat'], 'Result', '-v7.3');


end
