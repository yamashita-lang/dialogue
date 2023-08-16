function pca_stats(md_idx, sig_pc, mode)
% pca_stats.m
% 
% Interpretation of PCA results  
%
% Inputs: md_idx : modality index (1: production, 2: comprehension)
%
%         sig_pc : number of significant PCs
%
%         mode   : 1: PC loading for each utterance (Extended Data Tables)
%                  2: Correlation with low-level features (Figure 5 b,e)
%                  3: Correlation with part-of-speech weights (Figure 5 c,f)
%                  4: RGB color corresponding PC1/2/3 for each utterance (Figure 5 g,h)


%%
PRM=load_parameters_proj;

switch md_idx
    case 1
        name='Prod';
    case 2
        name='Comp';
end


switch mode

    case 1 % PC loadings by correlating PC coeff and embeddings
        % 1 Concatenate stimuli across participants
        groupstim_file=[ SaveDir 'GroupStim_' PRM.LLM '_' name '.mat'];


        for ss = 1:length(PRM.SubjectsList)
            ID = ['sub-OSU0' char(PRM.SubjectsList(ss)) ];
            disp(['===== Subject ' ID '  ====='])
            IND=load_parameters_ind(ID, PRM);


            % 1 Load stimuli (utterance)
            utterance_file = [ IND.DataDir 'Stim_Utterance_' name '_' ID '.mat' ]; load(utterance_file);
            disp(['# utterances: ' num2str(length(utt)) ]);
            unique_idx=1;
            if unique_idx % Loading only unique utterances
                [ utt, ia, ic ] = unique(utt, 'rows');
            else
                % Loading all the utterances
            end

            disp([ '# unique utterances: ' num2str(length(utt)) ]);
            for cc = 1:length(utt) 
                utt_str(cc) = convertCharsToStrings(utt(cc,:));
            end


            % 2 Loading embedding (utterance x LLM hidden units)
            embfile=[ IND.DataDir 'Embedding_' PRM.LLM '_' name '_' ID '.mat' ]; load(embfile);

            if unique_idx
                % Loading LLM embedding for only unique utterances
                X = chatgptneox(ia,:); clear chatgptneox;
            else
                % Loading LLM embedding for all the utterances
                X = chatgptneox; clear chatgptneox;
            end
            save( [ IND.DataDir 'Uniq_embedding_' PRM.LLM '_' name '_' ID '.mat' ], 'X', '-v7.3');

            % 3 Concatenate across the participants

            if ss == 1
                groupX = X'; %'
                U = utt_str;
            else
                groupX = [groupX X']; %'
                U = cat(2, U, utt_str);
            end

            clear utt_str; clear utt; clear X; clear ia; clear ic;
        end
        

        % 2 Extract unique utterances
        disp(['U: ' num2str(size(U,1))  ' x ' num2str(size(U,2))]);
        disp(['groupX: ' num2str(size(groupX,1))  ' x ' num2str(size(groupX,2))]);
        disp(['Extract unique utterances ... ']);

        U = strtrim(U); clear ia; clear ic; 
        [ U, ia, ic ] = unique(U); groupX = groupX(:, ia);
        group_emb_uniq = groupX; save([ PRM.SaveDir 'GroupEmb_' name '_uniq.mat' ], 'group_emb_uniq', '-v7.3');
        clear group_emb_uniq;


        % 3 Load PCA results
        load([PRM.SaveDir 'PCA_Result_' PRM.LLM '_' name '.mat' ]); pcaResult
        

        % 4 Calculate PC loadings by correlating PC coeff with embeddings
        [PCloading, P] = corr(pcaResult.coeff(:,1:sig_pc), groupX);


        % 5 Extract most correlative utterances for each PC
        tmp_fileName = [PRM.SaveDir 'PCA_Result_' PRM.LLM '_' name ]
        n_extract=50;

        for pc = 1:sig_pc
            disp([ name '****** PC ' num2str(pc) ]);
            [sorted_PCloading d_idx] = sort(PCloading(pc, :), 'descend');
            
            txtfile = [ tmp_fileName '_high_PC' num2str(pc) '.txt' ];
            fileID = fopen(txtfile,'w');
            for ii = 1:n_extract
                disp(['Max corr ' num2str(ii) num2str(sorted_PCloading(ii)) U(d_idx(ii)) ]);
                fprintf(fileID, '%d\t%f\t%s\n', ii, sorted_PCloading(ii), U(d_idx(ii)) );
            end
            fclose(fileID);

            txtfile = [ tmp_fileName '_low_PC' num2str(pc) '.txt' ];
            fileID = fopen(txtfile,'w');
            [sorted_PCloading a_idx] = sort(PCloading(pc, :), 'ascend');
            for ii = 1:n_extract
                disp(['Min corr ' num2str(ii) num2str(sorted_PCloading(ii)) U(a_idx(ii)) ]);
                fprintf(fileID, '%d\t%f\t%s\n', ii, sorted_PCloading(ii), U(a_idx(ii)) );
            end
            fclose(fileID);
        end
        



    case 2 % Correlation between PC loadings and low-level features (Figure 5 b,e)

        % 1 Concatenate stimuli and low-level features across participants

        for ss = 1:length(PRM.SubjectsList)
            ID = ['sub-OSU0' char(PRM.SubjectsList(ss)) ];
            disp(['===== Subject ' ID '  ====='])
            IND=load_parameters_ind(ID, PRM);

            % 1 Load stimuli (utterance)
            utterance_file = [ IND.DataDir 'Stim_Utterance_' name '_' ID '.mat' ]; load(utterance_file);
            disp(['# utterances: ' num2str(length(utt)) ]);
            unique_idx=1;
            if unique_idx % Loading only unique utterances
                [ utt, ia, ic ] = unique(utt, 'rows');
            else
                % Loading all the utterances
            end

            disp([ '# unique utterances: ' num2str(length(utt)) ]);
            for cc = 1:length(utt) 
                utt_str(cc) = convertCharsToStrings(utt(cc,:));
            end


            % 2 Loading embedding (utterance x LLM hidden units)
            load( [ IND.DataDir 'Uniq_embedding_' PRM.LLM '_' name '_' ID '.mat' ]);
            

            % 3 Load morphem and syllable counts for each utterance
            lowlevel_file=[ IND.DataDir 'Lowlevel_' name '_' ID ]; load(lowlevel_file);
            
            if unique_idx
                mrph = mrph_rate(ia); clear mrph_rate;
                syllable = syllable(ia); % clear syllable;
            else
                % Loading all the utterances
            end

            
            % 4 Concatenate across the participants
            X2 = [ mrph' syllable'];
            if ss == 1
                groupX = X; U = utt_str; X_lowlevel = X2'; %'
            else
                groupX = [groupX; X]; U = cat(2, U, utt_str); X_lowlevel = [X_lowlevel X2']; %'
            end

            clear utt_str; clear utt; clear ia; clear ic;
        end
               

        % 2 Extract unique utterances 
        disp(['Extract unique utterances ... ']);
        U = strtrim(U); clear ia; clear ic;
        [ U, ia, ic ] = unique(U); groupX = groupX(ia,:); X_lowlevel = X_lowlevel(:,ia);


        % 3 Load PCA results
        load([PRM.SaveDir 'PCA_Result_' PRM.LLM '_' name '.mat' ]); pcaResult


        % 4 Calculate PC loadings by correlating PC coeff with embeddings
        PCloading = corr(pcaResult.coeff(:,1:sig_pc), groupX'); %'
        PCloading = PCloading'; %'
        X_lowlevel = X_lowlevel'; %'
        


        % 5 Calculate partial Pearson correlation between PC loading and morpheme controlling syllabel, and vice versa
        tmp_fileName = [PRM.SaveDir 'PCA_Result_' PRM.LLM '_' name '_lowlevel' ];
        R_all = [];
        for ff = 1:2
            switch ff
                case 1
                    name='mrph';
                case 2
                    name='syllable';
            end

            tmpX2 = X_lowlevel; tmpX2(:,ff) = [];
            [R, P] = partialcorr(PCloading, X_lowlevel(:,ff), tmpX2);

            % Perform FDR correction
            Q=0.05;
            [Psorted Pind] = sort(P, 'ascend');
            FDRthr = Q*[1:length(P)]/length(P);
            Diff = Psorted - FDRthr'; %'
            thrInd = Pind(max(find(Diff<0))); 
            thrP = P(thrInd);
            disp(['Threshold P = ' num2str(thrP)])

            for pp = 1:sig_pc
                disp(['PC' num2str(pp) ': ' num2str(R(pp)) ', P=' num2str(P(pp)) ]);
            end
            
            R_all = [R_all R];
            
        end

        csvwrite([ tmp_fileName '_PC_vs_LowLevelFeatures.csv' ], R_all)



    case 3 %%% Correlation between part-of-speech and PCA (Figure c, f)
        % 0 Load PCA results
        pca_result=[ PRM.SaveDir 'PCA_Result_' PRM.LLM '_' name '.mat']; load(pca_result);

        % 1 Calculate Pearson correaltion for each participant

        for ss = 1:length(SubjectsList)
            ID = ['sub-OSU0' char(PRM.SubjectsList(ss)) ];
            disp(['===== Subject ' ID '  ====='])
            IND=load_parameters_ind(ID, PRM);

            % 0 load weight including part-of-speech weight
            weight_file = [PRM.SaveDir 'Weight_' PRM.LLM '_' name '_' ID '.mat' ]; load(weight_file);
        

            % 1 Load meanweight file of LLM
            meanweight_file = [PRM.SaveDir 'RidgeResult_' PRM.LLM '_meanweight_' ID '.mat' ];    
            load(meanweight_file);


            % 2 Extract part-of-speech weight averaged across time delays
            switch md_idx
                case 1
                    w_pos = meanw(PRM.POS_Prod_index,:);
                case 2
                    w_pos = meanw(PRM.POS_Comp_index,:);
            end

            % average for time delays
            nFeature = size(w_pos,1)/nDelay; disp(['Feature: ' num2str(nFeature) ' dimensions']);
            mw_pos = reshape(w_pos, nFeature, nDelay, []); mw_pos = nanmean(mw_pos,2); mw_pos = squeeze(mw_pos);


            % 3 Select pca_svoxels for weight_prod and weight_comp
            % Get top PRM.PCANvoxel showing largest mean ccs
            load([PRM.SaveDir IND.file_prefix '_FDRcorr_mean.mat' ]);
            [sv si] = sort(mean_Result.mean_ccs_same,'descend');
            pca_svoxels = si(1:PRM.PCANvoxel); mw_pos_pcasvoxels = mw_pos(:, pca_svoxels);


            % 4 correlation PC score vs. POS weight in topNvoxels
            [r p] = corr(pcaResult.score(1+(ss-1)*PRM.PCANvoxel:ss*PRM.PCANvoxel, 1:sig_pc), mw_pos_pcasvoxels'); %'
            outname=[ PRM.SaveDir 'Corr_PCA_vs_POS_' PRM.LLM '_' name '_' ID '.mat' ];
            save(outname, 'r', '-v7.3' );

            R(ss,:,:) = r;
        end

        % 3 Calculate mean correlation
        group_meanR = squeeze(tanh(nanmean(atanh(R))));
        outname=[ PRM.SaveDir 'GroupMean_Corr_PCA_vs_POS_' PRM.LLM '_' name '.mat' ]
        save(outname, 'group_meanR', '-v7.3' );



    case 4 % Output RGB color for each utterance based on r(PCcoef, LLM) (Figure 5 g,h)

    	for ss = 1:length(SubjectsList)
            ID = ['sub-OSU0' char(PRM.SubjectsList(ss)) ];
            disp(['===== Subject ' ID '  ====='])
            IND=load_parameters_ind(ID, PRM);

            % 1 Load stimuli (utterance)
            utterance_file = [ IND.DataDir 'Stim_Utterance_' name '_' ID '.mat' ]; load(utterance_file);
            disp(['# utterances: ' num2str(length(utt)) ]);
            unique_idx=1;
            if unique_idx % Loading only unique utterances
                [ utt, ia, ic ] = unique(utt, 'rows');
            else
                % Loading all the utterances
            end

            disp([ '# unique utterances: ' num2str(length(utt)) ]);
            for cc = 1:length(utt) 
                utt_str(cc) = convertCharsToStrings(utt(cc,:));
            end


            % 2 Loading embedding (utterance x LLM hidden units)
            load( [ IND.DataDir 'Uniq_embedding_' PRM.LLM '_' name '_' ID '.mat' ]);

            if ss == 1
                groupX = X;
                U = utt_str;
            else
                groupX = [groupX; X];
                U = cat(2, U, utt_str);
            end

            clear utt_str; clear utt; clear ia; clear ic;
        end

        clear X; X=groupX'; %'

        % 2 Extract unique utterances
        disp(['U: ' num2str(size(U,1))  ' x ' num2str(size(U,2))]);
		disp(['X: ' num2str(size(X,1))  ' x ' num2str(size(X,2))]);
		disp(['Extract unique utterances ... ']);

		U = strtrim(U); clear ia; clear ic;
		[ U, ia, ic ] = unique(U);
		X = X(:, ia);

        tmp_fileName = [PRM.SaveDir 'Unique_utterances_' name '.txt' ]
        fileID = fopen(tmp_fileName,'w');
        for tt = 1:length(U)
            fprintf(fileID,'%s\n',U(tt));
        end
        fclose(fileID);


		% 3 Calculate PC loading for each utterance and save
		load([PRM.SaveDir 'PCA_Result_' PRM.LLM '_' name '.mat' ]);
		utt_pc = corr(pcaResult.coeff(:,1:sig_pc), X);

		tmp_fileName = [PRM.SaveDir 'PCload_' PRM.LLM '_' name '.mat' ]; save(tmp_fileName, 'utt_pc',  '-v7.3');
		tmp_fileName = [PRM.SaveDir 'PCload_' PRM.LLM '_' name '.txt' ];
		fileID = fopen(tmp_fileName,'w');
		for tt = 1:length(U)
			fprintf(fileID,'%s\n',U(tt));
		end
		fclose(fileID);

	
        % 4 save for each dimension
		for pc = 1:3
			data = zscore(utt_pc(pc,:)); data = data + abs(min(data)); data = data / max(data);
			clr(pc,:) = data;  clear data;

			switch pc
				case 1
					cName ='R';
				case 2
					cName ='G';
				case 3
					cName ='B';
			end
			tmp_fileName = [PRM.SaveDir 'PCload_' PRM.LLM '_' name '_RGB_' cName '.txt' ]; fileID = fopen(tmp_fileName,'w');
			for tt = 1:size(clr,2)
				fprintf(fileID,'%f\n',clr(pc,tt));
			end
			fclose(fileID);
		end
		clear utt_pc; clear clr;


end


end