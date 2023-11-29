function pca_bootstrap(md_idx, mode)
% pca_bootstrap.m
% 
% PCA for bootstrap resampled data 
%
% Note: utterance_file ('Stim_Utterance_Prod_sub-OSU01.mat') includes
%            sensitive information about participants and only avilable upon reasonable request.
%
% Inputs: md_idx : modality index (1: production, 2: comprehension)
%
%         mode   : 1: bootstrap, 2: pvalue
%
% Outputs:  'PCA_bootstrap_CHATGPTNEOX_Prod.mat'
%           'PCA_bootstrap_CHATGPTNEOX_Prod_pval.csv'
%           'PCA_bootstrap_CHATGPTNEOX_Prod.csv'
%


%%
PRM=load_parameters_proj;

switch md_idx
    case 1
        name='Prod';
    case 2
        name='Comp';
end


% Bootstrap parameter and randomization
rng(9, 'twister');


% Stimulus concatenated across participants
groupstim_file=[ PRM.SaveDir 'GroupUniqEmbedding_' PRM.LLM '_' name '.mat'];


switch mode
    case 1

        % 1 Make or load stimuli concatenated across subjects

        if ~exist(groupstim_file) 
            for ss = 1:length(PRM.SubjectsList)
                ID = ['sub-OSU0' char(SubjectsList(ss)) ];
                disp(['===== Subject ' ID '  ====='])
                IND=load_parameters_ind(ID, PRM);

                % 1 Load utterance
                utterance_file = [ IND.DataDir 'Stim_Utterance_' ID '_' name '.mat' ];
                load(utterance_file); % utt
                unique_idx=1;
                if unique_idx
                    disp('*** Loading only unique utterances ');
                    [ utt, ia, ic ] = unique(utt, 'rows');
                else
                    disp('*** Loading all the utterances ');
                end

                % 2 Loading utterance embedding (utterance x LLM)
                load([ IND.DataDir 'Embedding_' PRM.LLM '_' ID '_' name '.mat' ]);
                if unique_idx
                    disp('**** Loading LLM embedding for only unique utterances');
                    X = chatgptneox(ia,:); clear chatgptneox;
                else
                    disp('**** Loading LLM embedding for all the utterances');
                    X = chatgptneox; clear chatgptneox;
                end
                save( [ IND.DataDir 'Uniq_Embedding_' PRM.LLM '_' ID '_' name '.mat' ], 'X', '-v7.3');


                % 3 Normalize and concatenate across subjects
                for ff = 1:size(X,1)
                    zX(ff,:) = zscore(X(ff,:));
                end

                Nutt = size(X,1); clear X;
                tmp_zX_stim = zX; clear zX;

                if ss == 1
                    zX_stim = tmp_zX_stim; %
                else
                    zX_stim = [zX_stim; tmp_zX_stim];
                end

                clear tmp_zX_stim;
            end

            zX_stim = zX_stim'; %'
            disp(['LLM embeddings: ' num2str(size(zX_stim, 1)) ' x ' num2str(size(zX_stim, 2))]);
            save(groupstim_file, 'zX_stim', '-v7.3');
        else
            load(groupstim_file)
        end


        % 2 Load concatenated weights across participants
        groupweight_file=[ PRM.SaveDir 'GroupWeight_' PRM.LLM '_' name '.mat'];
        load(groupweight_file); zX_weight = zX; clear zX;


        % 3 Compare % variance explained between stimulus and weight
        zX_stim = zX_stim'; %' 
        n1=size(zX_weight,1); n2=size(zX_stim,1);

        disp(['bootstrap weight: ' num2str(n1)]);
        disp(['bootstrap stimuli: ' num2str(n2)]);

        for bb = 1:PRM.nboot
            if rem(bb, 100) == 0
                disp(['***** BOOTSTRAP ' num2str(bb) '/' num2str(PRM.nboot) ]);
            end

            % 1 Weight PCA
            tmp_weight = randsample(n1, n1, true);
            tmpzX_weight = zX_weight(tmp_weight,:);
            [coeff, ~, ~, ~, explained] = pca(tmpzX_weight);

            weight_coeff(bb,:,1:PRM.topNcomp) = coeff(:, 1:PRM.topNcomp);
            explained_weight(bb,:) = explained(1:PRM.topNcomp); 
            clear coeff; clear explained;

            % 2 Stimulus PCA
            tmp_stim = randsample(n2, n2, true);
            tmpzX_stim = zX_stim(tmp_stim,:);
            [coeff, ~, ~, ~, explained] = pca(tmpzX_stim);

            stim_coeff(bb,:,1:PRM.topNcomp) = coeff(:, 1:PRM.topNcomp);
            explained_stim(bb,:) = explained(1:PRM.topNcomp); 

            clear coeff; clear explained;
        end


        % Match the most comparable PCs using Gale-Shapley algorithm
        for bb = 1:PRM.nboot
            if rem(bb, 100) == 0
                disp(['***** Match and compare ' num2str(bb) '/' num2str(PRM.nboot) ]);
            end

            absC = abs(corr(squeeze(weight_coeff(bb,:,:)), squeeze(stim_coeff(bb,:,:))));
            [B weight_pref] = sort(absC, 'descend');

            absC = abs(corr(squeeze(stim_coeff(bb,:,:)), squeeze(weight_coeff(bb,:,:))));
            [B stim_pref] = sort(absC, 'descend');

            stablematch = galeshapley(PRM.topNcomp, weight_pref, stim_pref);

            compare_explained(bb,:) = explained_weight(bb,:) - explained_stim(bb,stablematch);
            bootResult.explained_stim(bb,:) = explained_stim(bb,stablematch);

        end

        for nn = 1:PRM.topNcomp
            lose_times(nn) = length(find(compare_explained(:,nn) < 0));
        end

        lose_times/PRM.nboot

        bootResult.explained_weight = explained_weight;
        bootResult.pval = lose_times/PRM.nboot;
            
        tmp_fileName = [ PRM.SaveDir 'PCA_bootstrap_' PRM.LLM '_' name ];
        save(tmp_fileName, 'bootResult',  '-v7.3');
        writematrix(bootResult.pval, [tmp_fileName '_pval.csv']);


    case 2
        tmp_fileName = [ PRM.SaveDir 'PCA_bootstrap_' PRM.LLM '_' name ];
        load(tmp_fileName);

        disp('confidence intervals for each PC')
        alpha = 0.01;

        SEM = std(bootResult.explained_weight) / sqrt(PRM.nboot);
        ts = tinv([alpha/2 1-alpha/2], PRM.nboot-1);
        M = mean(bootResult.explained_weight);
        CI_high = M - ts(1).*SEM;
        CI_low = M - ts(2).*SEM;
        CI_value = ts(2).*SEM;

        X = [M; CI_value];

        SEM = std(bootResult.explained_stim) / sqrt(PRM.nboot);
        ts = tinv([alpha/2 1-alpha/2], PRM.nboot-1);
        M = mean(bootResult.explained_stim);
        CI_high = M - ts(1).*SEM;
        CI_low = M - ts(2).*SEM;
        CI_value = ts(2).*SEM;

        X = [X; M; CI_value ];

        writematrix(X, [tmp_fileName '.csv'])
end

