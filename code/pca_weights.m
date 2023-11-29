function pca_weights(md_idx, sig_pc)
% pca_weights.m
% 
% Determine vest variance partition for each voxel
%
% Inputs: md_idx : modality index (1: production, 2: comprehension)
%
%         sig_pc : number of significant PCs
%
% Outputs:  'RidgeResults_CHATGPTNEOX_sub-OSU01_meanweight.mat'
%           'Weight_CHATGPTNEOX_sub-OSU01_Prod.mat'
%           'GroupWeight_CHATGPTNEOX_Prod.mat'
%           'PCA_Results_CHATGPTNEOX_Prod.mat'
%           'PCA_Results_CHATGPTNEOX_sub-OSU01_Prod_RGBmap_R.nii'
%


%%
if nargin<2,
    sig_pc=10;
end

PRM=load_parameters_proj;

switch md_idx
    case 1
        name='Prod';
    case 2
        name='Comp';
end


groupweight_file=[ PRM.SaveDir 'GroupWeight_' PRM.LLM '_' name '.mat'];
pca_result_file=[ PRM.SaveDir 'PCA_Results_' PRM.LLM '_' name '.mat'];


if exist(pca_result_file)
    disp(['Loading ' pca_result_file ]); load(pca_result_file);
else
    disp(['Performing PCA for ' name]);
    if ~exist(groupweight_file) 
        disp(['===== Concatenating weights across participants ====='])
        for ss = 1:length(PRM.SubjectsList)
            ID = ['sub-OSU0' char(PRM.SubjectsList(ss)) ];
            disp(['===== Subject ' ID '  ====='])
            IND=load_parameters_ind(ID, PRM);

            % 1 Load mean weight across sessions
            meanweight_file = [PRM.SaveDir IND.file_prefix '_meanweight.mat' ];
            if ~exist(meanweight_file)
                disp(['make and save ' meanweight_file])
                
                for ses = 1:IND.ses_num
                    load([PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '.mat' ]);
                    w(ses,:,:) = Result.w;
                end
                meanw = squeeze(nanmean(w)); clear w;
                save(meanweight_file, 'meanw', '-v7.3');
            else
                load(meanweight_file)
            end 
            

            % 2 Average weights across time delays
            switch md_idx
                case 1
                    w_topn = meanw(PRM.Prod_index,:);
                case 2
                    w_topn = meanw(PRM.Comp_index,:);
            end

            nFeature = size(w_topn,1)/PRM.nDelay; disp(['Feature: ' num2str(nFeature) ' dimensions']);
            mw_topn = reshape(w_topn, nFeature, PRM.nDelay, []);
            mw_topn = nanmean(mw_topn,2); mw_topn = squeeze(mw_topn);

            pcaweight_file = [PRM.SaveDir 'Weight_' PRM.LLM '_' ID '_' name '.mat' ];
            w=mw_topn; save(pcaweight_file, 'w', '-v7.3');


            % 3 Select pca_svoxels for weight_prod and weight_comp
            % Get top PRM.PCANvoxel showing largest mean ccs
            load([PRM.SaveDir IND.file_prefix '_FDRcorr_mean.mat' ]);
            [sv si] = sort(mean_Result.mean_ccs_same,'descend');
            pca_svoxels = si(1:PRM.PCANvoxel);   
            mw_topn = mw_topn(:, pca_svoxels);


            % 4 Normalize weights for PCA
            zmw_topn = [];
            for ff = 1:size(mw_topn,1)
                [zmw_topn(ff,:) norm_mu(ff) norm_sigma(ff)] = zscore(mw_topn(ff,:));
            end


            % 5 Concatenate weights across participants
            if ss == 1
                zX = zmw_topn;
            else
                zX = [zX zmw_topn];
            end

            clear zmw_topn;
        end
                    
        zX = zX'; %'
        disp(['Weight: ' num2str(size(zX, 1)) ' x ' num2str(size(zX, 2))]);
        save(groupweight_file, 'zX', '-v7.3'); 
    else
         load(groupweight_file)
    end

    disp('****************************************');
    disp(['*** Applying PCA for weight ...' num2str(size(zX, 1)) ' x ' num2str(size(zX, 2)) ]) ;
    [coeff, score, latent, tsquared, explained, mu] = pca(zX);

    pcaResult.coeff = coeff; pcaResult.score = score;
    pcaResult.latent = latent; pcaResult.tsquared = tsquared;
    pcaResult.explained = explained; pcaResult.mu = mu;
    pcaResult

    save(pca_result_file, 'pcaResult', '-v7.3');
end



% Save PCA score map
for ss = 1:length(PRM.SubjectsList)
    ID = ['sub-OSU0' char(PRM.SubjectsList(ss)) ];
    disp(['===== Subject ' ID '  ====='])
    IND=load_parameters_ind(ID, PRM);
    

    % 1 Load mean weight across time delays
    pcaweight_file = [PRM.SaveDir 'Weight_' PRM.LLM '_' ID '_' name '.mat' ];
    load(pcaweight_file);
    tmp_w = w;


    % Write down voxel values of the Nth PC into .nii
    load([PRM.SaveDir IND.file_prefix '_FDRcorr_mean.mat' ]);
    [sv si] = sort(mean_Result.mean_ccs_same,'descend');
    pca_svoxels = si(1:PRM.PCANvoxel);


    % 2 Calculate PC scores 
    zw_pcav = []; w_pcav=w(:, pca_svoxels);
    for ff = 1:size(w_pcav,1)
        [zw_pcav(ff,:) norm_mu(ff) norm_sigma(ff)] = zscore(w_pcav(ff,:));
    end

    zw = [];
    for ff = 1:size(tmp_w,1)
        zw(ff,:) = (tmp_w(ff,:) - repmat(norm_mu(ff), 1, size(tmp_w,2)))/norm_sigma(ff);
    end
    tmp = zw'; %'

    load([ PRM.SaveDir 'PCA_Result_' PRM.LLM '_' name '.mat']);
    new_score = (tmp - pcaResult.mu) * pcaResult.coeff(:, 1:sig_pc); 

    % Load significantly predicted voxels
    load([ PRM.SaveDir IND.file_prefix '_FDRcorr_mean.mat' ]);
    disp([ 'svoxles from session average: ' num2str(length(mean_Result.svoxels)) ]);

    
    for N = 1:sig_pc
        disp([' - PC ' num2str(N) ]);
        score_svoxels=new_score(mean_Result.svoxels, N);
        PCAdata = zscore(score_svoxels);

        disp([ 'PCAdata, max: ' num2str(max(PCAdata)) ]);
        disp([ 'PCAdata, min: ' num2str(min(PCAdata)) ]);

        X = NaN(size(IND.tvoxels));
        for kk = 1:length(mean_Result.svoxels)
            X(mean_Result.svoxels(kk)) = PCAdata(kk);
        end
        
        Y = NaN(prod(IND.datasize),1);
        for vv=1:length(IND.tvoxels)
           Y(IND.tvoxels(vv))= X(vv);
        end

        vol = reshape(Y,IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm;
        MRIwrite(V, [ PRM.SaveDir 'PCA_Result_' PRM.LLM '_' ID '_' name '_PC' num2str(N) '.nii' ]);
        
        % RGB map
        % Normalize into the [0 1]
        if N == 1 || N == 2 || N == 3
            switch N
                case 1
                    data = PCAdata; data = data + abs(min(data)); data = data / max(data);
                    cName = 'R';
                case 2
                    data = PCAdata; data = data + abs(min(data)); data = data / max(data);
                    cName = 'G';            
                case 3
                    data = PCAdata; data = data + abs(min(data)); data = data / max(data);
                    cName = 'B';            
            end
            X = NaN(size(IND.tvoxels));
            for kk = 1:length(mean_Result.svoxels)
                X(mean_Result.svoxels(kk)) = data(kk);
            end
            
            Y = NaN(prod(IND.datasize),1);
            for vv=1:length(IND.tvoxels)
               Y(IND.tvoxels(vv))= X(vv);
            end

            vol = reshape(Y,IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm;
            MRIwrite(V, [ PRM.SaveDir 'PCA_Results_' PRM.LLM '_' ID '_' name '_RGBmap_'  cName '.nii']);
        end     
    end

    disp(['===== Subject ' ID '  done ====='])

    clear tmp_w; clear w_prod; clear w_comp; clear w_topn;
    clear meanw; clear meanw_prod; clear meanw_comp;
    clear zw; 
    clear data; clear PCAdata; 
end

end
