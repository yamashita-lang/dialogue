function do_group_pca
%
% do_group_pca.m
%

%% 


% Group PCA (Figure 5)
mode=1; % production
pca_weights(mode);

for idx=1:2 
	pca_bootstrap(mode, idx);
end

sig_pc=4; for production

for idx = 1:4
	pca_stats(mode, sig_pc, idx)
end


mode=2; % comprehension
pca_weights(mode);

for idx=1:2 
	pca_bootstrap(mode, idx);
end

sig_pc=6; for comprehension

for idx = 1:4
	pca_stats(mode, sig_pc, idx)
end

end
