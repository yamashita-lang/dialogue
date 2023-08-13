function ridge_cerebcortex_submodel(ID, rd_idx)
% ridge_cerebcortex_submodel.m
% 
% Ridege regression reducing one feature space from the full model
%
% Inputs: ID     : subject ID (e.g. 'sub-OSU01')
%
%         rd_idx : index for reduced model (1: prod_LLM, 2: prod_ccg, ..., 10: comp_mrpl, 11: comp_pos)


%%
PRM=load_parameters_proj;
IND=load_parameters_ind(ID, PRM);


Features_used=[];
ii = 0;
for modality = [1 2]
    if modality == 1
        name='prod';
    elseif modality == 2
        name='comp';
    end

    for mm = PRM.Features_index
        ii = ii + 1;
        if ii == rd_idx
            disp(['Reducing ' name '_' PRM.Features{mm}  ' from the full model']); 
        else
            if mm == length(PRM.Features) && modality == 2
                ; % skipping repetition for MOT
            else
                Method=char(PRM.Features{mm});
                Features_used=[Features_used char(num2str(ii))];
            end
        end        
    end
end

disp(['Current features used: ' Features_used ]);


% Perform leave-session-out cross-validation
for ses = 1:IND.ses_num
    disp('===================================================================')
    disp(['     Cross-validation: ' num2str(ses) '/' num2str(IND.ses_num) ]);
    disp('===================================================================')

    % Load BOLD response [Time x Voxels], separately for training and testing 
    load([DataDir '/RespData_' ID '_ses-' num2str(ses) '_' PRM.taskname '_bold' ]);
    respTrn_ROI = RespData.respTrn; respTest_ROI = RespData.respTest;
    
    
    % Load stimulus [Time x Stimuli], separately for training and testing
    load([DataDir '/StimData_' PRM.LLM '_' ID '_ses-' num2str(ses) '_' PRM.taskname ]); 
    disp(['full stimuli size: ' num2str(size(stimTrn,1)) ' x ' num2str(size(stimTrn,2))]);

    % Extract features used in the current model
    ii = 0; stim_index=0;
    target_stimTrn=[]; target_stimTest=[];
    for modality = [1 2]
        if modality == 1
            name='prod';
        elseif modality == 2
            name='comp';
        end
        for mm = PRM.Features_index
            ii = ii + 1;
            switch mm
                case 1 % LLM
                    tmp_index=PRM.LLMdim;
                case 2 % ccg
                    tmp_index=PRM.CCGdim;
                case 3 % syllable count
                    tmp_index=PRM.SYLdim; 
                case 4 % morpheme count
                    tmp_index=PRM.MRPLdim;
                case 5 % part-of-speech
                    tmp_index=PRM.POSdim;
                case 6 % hear motion 
                    tmp_index=PRM.MOTdim;
            end
            stim_index=max(stim_index)+1:max(stim_index)+tmp_index;
            
            if ii == rd_idx
                ; % disp(['Reduce ' name '_' PRM.Features{mm}  ' from the full model']);;
            else
                if mm == length(PRM.Features) && modality == 2
                    ; % skipping repetition for MOT
                else
                    target_stimTrn = horzcat(target_stimTrn, stimTrn(:,stim_index));
                    target_stimTest = horzcat(target_stimTest, stimTest(:,stim_index));
                end
            end        
        end
    end
    disp(['reduced stimuli size: ' num2str(size(target_stimTrn,1)) ' x ' num2str(size(target_stimTrn,2))]);

    clear stimTrn; clear stimTest;
    stimTrn=target_stimTrn; stimTest=target_stimTest;
    

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
        ccs=zeros(length(tvoxels),length(PRM.ridge_as));
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
    presp = stimTest*w;
    ccs = mvn_corr(presp, respTest_ROI); % test predictions
    fprintf('mean ccs = %.3f\n', nanmean(ccs));


    Result.session = ses;
    Result.best_a = PRM.ridge_as(baind);
    Result.w = w;
    Result.ccs = ccs; 
    Result.mean_ccs = nanmean(ccs);
    Result.resp = respTest_ROI;
    Result.presp = presp;

    mpaidx=0;
    if mpaidx 
        % Mapping from 1d Data to 3d .nii data 
        Y = zeros(prod(IND.datasize),1);
        for ii=1:length(IND.tvoxels)
            Y(IND.tvoxels(ii))= ccs(ii);
        end
        vol = reshape(Y, IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm;
        MRIwrite(V,[ PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '_' Features_used '.nii']);
    end


    % Save Result 
    save([ PRM.SaveDir IND.file_prefix '_ses-' num2str(ses) '_' Features_used '.mat' ], 'Result', '-v7.3');
  
end


end        