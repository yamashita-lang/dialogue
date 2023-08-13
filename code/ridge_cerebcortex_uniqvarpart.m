function ridge_cerebcortex_uniqvarpart(ID)
% ridge_cerebcortex_uniqvarpart.m
% 
% Unique variance partitioning
%
% Inputs: ID     : subject ID (e.g. 'sub-OSU01')
%


%%
PRM=load_parameters_proj;
IND=load_parameters_ind(ID, PRM);


% Concatenate results across sessions
resp=[]; presp=[];
for ses = 1:IND.ses_num
	ses_result=[ PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '_' PRM.FullModel_name '.mat' ];
	disp(['loading ' ses_result ]); load(ses_result);
	
	if ses == 1
		resp = Result.resp; prespU = Result.presp;
	else
		resp = [resp; Result.resp ]; prespU = [prespU; Result.presp ];
	end
end


% Significantly predicted voxels from mean results across sessions
load([ PRM.SaveDir IND.file_prefix '_FDRcorr_mean.mat' ]);
resp = resp(:, mean_Result.svoxels); 
prespU = prespU(:, mean_Result.svoxels);


% Calculate varaince
Tvar = sum(( resp -  mean(resp)).^2);
RU = 1 - sum((prespU - resp).^2) ./ Tvar;


kk = 0;
for modality = [1 2]
	if modality == 1
		result_fileName='Prod';
	elseif modality == 2
		result_fileName='Comp';
	end

	for mm = 1:length(PRM.Features)
		if mm == length(PRM.Features) && modality == 2
            ; %  skipping repetition for MOT
        else
			kk = kk + 1;

			disp('--------------------------------------------------------');
			disp(['     Unique variance explained by ' result_fileName '_' char(PRM.Features{mm}) ]);
			disp('--------------------------------------------------------');
			Features_index=1:length(PRM.Features)*2-1;
			Features_index(kk) = [];
			subFeatures_name = [];
			for jj = 1:length(PRM.Features)*2-2
				subFeatures_name = [ subFeatures_name char(num2str(PRM.Features_index(jj))) ];
			end

			image_result_file=[PRM.SaveDir IND.file_prefix '_' char(PRM.Features{mm}) '_' result_fileName '_UniqVP.nii' ];
			if exist(image_result_file)
				disp('- - - Previously analyzed - - -');
			else
				for ses = 1:IND.ses_num
				    disp('===================================================================')
				    disp(['     Cross-validation: ' num2str(ses) '/' num2str(IND.ses_num) ]);
				    disp('===================================================================')
				    load([PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '_' subFeatures_name '.mat']);
				    if ses == 1
						prespSU = Result.presp(:, mean_Result.svoxels);
					else
						prespSU = [prespSU; Result.presp(:, mean_Result.svoxels) ];
					end
				end
				
				% sub full union model
			    RSU = 1 - sum((prespSU - resp).^2) ./ Tvar;				


				% Unique variance explained
				Ruve = RU - RSU; Ruve(find(Ruve<0)) = 0;


		        % Transform from R^2 to R
		        R = sqrt(Ruve); Result.R_RC = R;

		        disp(['Mean R: ' num2str(nanmean(R)) ]);
		        disp(['Min R: ' num2str(min(R)) ]);
		        disp(['Max R: ' num2str(max(R)) ]);


		        mapidx=1
			    if mapidx
			        % Mapping from 1d Data to 3d .nii data 
			        R_ori = NaN(size(IND.tvoxels));
		            for vv = 1:length(R)
		                R_ori(mean_Result.svoxels(vv)) = R(vv);
		            end
				    Result.R_ori=R_ori;
				   	
				    Y = NaN(prod(IND.datasize),1);
				    for ii=1:length(IND.tvoxels)
				        Y(IND.tvoxels(ii))= R_ori(ii);
				    end
				    vol = reshape(Y, IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm;
				    MRIwrite(V, image_result_file);
				end


			    save([ PRM.SaveDir IND.file_prefix '_UniqVP_' char(PRM.Features{mm}) '_' result_fileName '.mat'], 'Result', '-v7.3');
			end
		end
	end
end

end
