function do_individual_analysis(ss)
%
% do_individual_analysis.m
%
% Input: ss     : subject number (1 to 8) 
%

%% 
ID = [ 'sub-OSU0' num2str(ss) ];


% Prediction of BOLD responses (Figure 1 and 2)
ridge_cerebcortex(ID)
ridge_fdrcorr(ID, 1)
ridge_fdrcorr(ID, 2)
ridge_session_average(ID)


% Unique variance partitioning

% Full model (Extended Data Figure 3)
% sub model removing one set of feature (e.g. production_cochleagram)
for rd_idx = 0:11
  	ridge_cerebcortex_submodel(ID, rd_idx);
end
ridge_cerebcortex_uniqvarpart(ID)


% Reduced model (Figure 3)
ridge_cerebcortex_mainfeatures(ID)
ridge_cerebcortex_singlefeature(ID, 1)
ridge_cerebcortex_singlefeature(ID, 2)
ridge_fdrcorr_mainfeatures(ID)
ridge_session_average_mainfeatures(ID)
ridge_cerebcortex_uniqvarpart_reduced(ID)
best_uniqvarpart_rgb(ID)


% Weight correlation (Figure 4)
weight_corr(ID)


end
