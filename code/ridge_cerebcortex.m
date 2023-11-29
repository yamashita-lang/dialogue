function ridge_cerebcortex(ID)
% ridge_cerebcortex.m
% 
% Ridege regression for same/cross-modal prediction
%   of BOLD responses for each voxel
%
% Inputs: ID     : subject ID (e.g. 'sub-OSU01')
%
% Outputs:  'RidgeResults_sub-OSU01_ses-1.mat'
%


%%
PRM=load_parameters_proj;
IND=load_parameters_ind(ID, PRM);


% Perform leave-session-out cross-validation
for ses = 1:IND.ses_num
    disp('===================================================================')
    disp(['     Cross-validation: ' num2str(ses) '/' num2str(IND.ses_num) ]);
    disp('===================================================================')


    % Load BOLD response [Time x Voxels], separately for training and testing 
    load([IND.DataDir '/RespData_' ID '_ses-' num2str(ses) '_' PRM.taskname '_bold' ]);
    respTrn_ROI = RespData.respTrn; respTest_ROI = RespData.respTest;
    

    % Load stimulus [Time x Stimuli], separately for training and testing
    load([IND.DataDir '/StimData_' PRM.LLM '_' ID '_ses-' num2str(ses) '_' PRM.taskname ]); 
    
    
    % Make delayed stimulus matrix
    stimTrn = cat(2, circshift(stimTrn,2), circshift(stimTrn,3), circshift(stimTrn,4), circshift(stimTrn,5), circshift(stimTrn,6), circshift(stimTrn,7));
    stimTest = cat(2, circshift(stimTest,2), circshift(stimTest,3), circshift(stimTest,4), circshift(stimTest,5), circshift(stimTest,6), circshift(stimTest,7));


    % 10 fold cross validation
    ssinds = getResampInd10fold(size(respTrn_ROI,1));
    tccs = zeros(1,18);

    for dd = 1:10
        % Model fitting
        ws = ridgemulti(stimTrn(ssinds(dd).trnInd,:), respTrn_ROI(ssinds(dd).trnInd,:), PRM.ridge_as);

        % Validation
        ccs=zeros(length(IND.tvoxels),length(PRM.ridge_as));
        for ii=1:length(PRM.ridge_as)
            presp = stimTrn(ssinds(dd).regInd,:)*ws(:,:,ii); % predicted responses
            ccs(:,ii) = mvn_corr(presp, respTrn_ROI(ssinds(dd).regInd,:));
            fprintf('alpha=%12.2f, ccs = %.3f\n', PRM.ridge_as(ii), nanmean(ccs(:,ii)));
        end
        tccs = tccs + nanmean(ccs);
    end
    tccs = tccs/10;


    % Select the best a from repetitions
    [baccs, baind] = max(tccs);
    fprintf('The best alpha is %12.2f, ccs=%.3f.\n', PRM.ridge_as(baind), baccs);


    % Calculate the final model and performance using test data
    disp('calculating test performance...');
    w = ridgemulti(stimTrn, respTrn_ROI, PRM.ridge_as(baind));


    % Extract LLM stimuli and weights

    stimTest_LLM_prod = horzcat( stimTest(:, 1+PRM.ALLdim*0:PRM.LLMdim+PRM.ALLdim*0),  ...
                                 stimTest(:, 1+PRM.ALLdim*1:PRM.LLMdim+PRM.ALLdim*1), ...
                                 stimTest(:, 1+PRM.ALLdim*2:PRM.LLMdim+PRM.ALLdim*2), ...
                                 stimTest(:, 1+PRM.ALLdim*3:PRM.LLMdim+PRM.ALLdim*3), ...
                                 stimTest(:, 1+PRM.ALLdim*4:PRM.LLMdim+PRM.ALLdim*4), ...
                                 stimTest(:, 1+PRM.ALLdim*5:PRM.LLMdim+PRM.ALLdim*5) );

    stimTest_LLM_comp = horzcat( stimTest(:, PRM.PRODdim+1+PRM.ALLdim*0:PRM.PRODdim+PRM.ALLdim*0+PRM.LLMdim), ...
                                 stimTest(:, PRM.PRODdim+1+PRM.ALLdim*1:PRM.PRODdim+PRM.ALLdim*1+PRM.LLMdim), ...
                                 stimTest(:, PRM.PRODdim+1+PRM.ALLdim*2:PRM.PRODdim+PRM.ALLdim*2+PRM.LLMdim), ...
                                 stimTest(:, PRM.PRODdim+1+PRM.ALLdim*3:PRM.PRODdim+PRM.ALLdim*3+PRM.LLMdim), ...
                                 stimTest(:, PRM.PRODdim+1+PRM.ALLdim*4:PRM.PRODdim+PRM.ALLdim*4+PRM.LLMdim), ...
                                 stimTest(:, PRM.PRODdim+1+PRM.ALLdim*5:PRM.PRODdim+PRM.ALLdim*5+PRM.LLMdim) );


    w_LLM_prod = vertcat(   w(1+PRM.ALLdim*0:PRM.LLMdim+PRM.ALLdim*0,:), ...
                            w(1+PRM.ALLdim*1:PRM.LLMdim+PRM.ALLdim*1,:), ...
                            w(1+PRM.ALLdim*2:PRM.LLMdim+PRM.ALLdim*2,:), ...
                            w(1+PRM.ALLdim*3:PRM.LLMdim+PRM.ALLdim*3,:), ...
                            w(1+PRM.ALLdim*4:PRM.LLMdim+PRM.ALLdim*4,:), ...
                            w(1+PRM.ALLdim*5:PRM.LLMdim+PRM.ALLdim*5,:) );

    w_LLM_comp = vertcat(   w(PRM.PRODdim+1+PRM.ALLdim*0:PRM.PRODdim+PRM.ALLdim*0+PRM.LLMdim, :), ...
                            w(PRM.PRODdim+1+PRM.ALLdim*1:PRM.PRODdim+PRM.ALLdim*1+PRM.LLMdim, :), ...
                            w(PRM.PRODdim+1+PRM.ALLdim*2:PRM.PRODdim+PRM.ALLdim*2+PRM.LLMdim, :), ...
                            w(PRM.PRODdim+1+PRM.ALLdim*3:PRM.PRODdim+PRM.ALLdim*3+PRM.LLMdim, :), ...
                            w(PRM.PRODdim+1+PRM.ALLdim*4:PRM.PRODdim+PRM.ALLdim*4+PRM.LLMdim, :), ...
                            w(PRM.PRODdim+1+PRM.ALLdim*5:PRM.PRODdim+PRM.ALLdim*5+PRM.LLMdim, :) );


    % same-modality
    presp_same = stimTest_LLM_prod*w_LLM_prod+stimTest_LLM_comp*w_LLM_comp;
    ccs_same = mvn_corr(presp_same, respTest_ROI); % test predictions
    fprintf('mean ccs_same = %.3f\n', nanmean(ccs_same));

    % cross-modality
    presp_cross = stimTest_LLM_prod*w_LLM_comp+stimTest_LLM_comp*w_LLM_prod;
    ccs_cross = mvn_corr(presp_cross, respTest_ROI); % test predictions
    fprintf('mean ccs_cross = %.3f\n', nanmean(ccs_cross));

    Result.session = ses; 
    Result.best_a = PRM.ridge_as(baind);
    Result.w = w;
    Result.ccs_same = ccs_same;
    Result.ccs_cross = ccs_cross; 
    Result.mean_ccs_same = nanmean(ccs_same);
    Result.mean_ccs_cross = nanmean(ccs_cross);
    Result.resp = respTest_ROI;
    Result.presp_same = presp_same;
    Result.presp_cross = presp_cross;

    mapidx=0
    if mapidx
        % Mapping from 1d Data to 3d .nii data 
        Y = zeros(prod(IND.datasize),1);
        for ii=1:length(IND.tvoxels)
            Y(IND.tvoxels(ii))= ccs_same(ii);
        end
        vol = reshape(Y, IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm;
        MRIwrite(V,[ PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '_same.nii']);

        Y = zeros(prod(IND.datasize),1);
        for ii=1:length(IND.tvoxels)
            Y(IND.tvoxels(ii))= ccs_cross(ii);
        end
        vol = reshape(Y, IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm;
        MRIwrite(V,[ PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '_cross.nii']);
    end

    % Save Result 
    save([ PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '.mat' ], 'Result', '-v7.3');
  
end

end        
