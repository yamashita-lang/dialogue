function best_uniqvarpart_rgb(ID)
% best_uniqvarpart_rgb.m
% 
% Determine vest variance partition for each voxel
%
% Inputs: ID     : subject ID (e.g. 'sub-OSU01')
%
% Outputs:  'RidgeResults_CHATGPTNEOX_sub-OSU01_VarPart_ccs.mat'
%			'RidgeResults_CHATGPTNEOX_sub-OSU01_VarPart_RGB.nii'
%


%%
PRM=load_parameters_proj;
IND=load_parameters_ind(ID, PRM);


% Load variance partitioning results
load([ PRM.SaveDir IND.file_prefix '_SingleFeature_VarPart.mat' ]); 
ccs.prod=Result.R_Prod_RC; 
ccs.comp=Result.R_Comp_RC; 
ccs.intsct=Result.R_Intersection; clear Result;


disp(['***** production vs. comprehension *****']);
A = ccs.prod; B = ccs.comp; A_B = ccs.intsct; ccs

VRGB=NaN(length(A),1);
% B:production, R: comprehension, G:intersection
VG=NaN(length(A),1); tmp1=find(A_B>A & A_B>B);  VG(tmp1)=1; VRGB(tmp1)=3;
VB=NaN(length(A),1); tmp1=find(A>A_B & A>B);	VB(tmp1)=1; VRGB(tmp1)=1;
VR=NaN(length(A),1); tmp1=find(B>A_B & B>A);    VR(tmp1)=1; VRGB(tmp1)=2;
group.VR=VR; group.VB=VB; group.VG=VG; 

disp([' Prodution voxels: ' num2str(length(find(VB==1))) ', ' num2str(length(find(VB==1))/length(A)*100) ' percent' ]);
disp([' Comprehension voxels: ' num2str(length(find(VR==1))) ', ' num2str(length(find(VR==1))/length(A)*100) ' percent' ]);
disp([' Intersection voxels: ' num2str(length(find(VG==1))) ', ' num2str(length(find(VG==1))/length(A)*100) ' percent' ]);


% Save file
save_file = [ PRM.SaveDir IND.file_prefix '_VarPart_ccs.mat'];
save(save_file, 'ccs')


% Significantly predicted voxels
load([ PRM.SaveDir IND.file_prefix '_MainFeatures_FDRcorr_mean.mat' ]);
disp(['svoxels: ' num2str(length(mean_Result.svoxels)) ' voxels']);


% Data for IND.tvoxels
NR = zeros(size(IND.tvoxels)); NG = zeros(size(IND.tvoxels)); NB = zeros(size(IND.tvoxels)); NRGB = zeros(size(IND.tvoxels));
for vv = 1:length(A)
	NR(mean_Result.svoxels(vv)) = VR(vv); NG(mean_Result.svoxels(vv)) = VG(vv); NB(mean_Result.svoxels(vv)) = VB(vv); NRGB(mean_Result.svoxels(vv)) = VRGB(vv);
end


% Mapping from 1d Data to 3d .nii data
YR = NaN(prod(IND.datasize),1); YG = NaN(prod(IND.datasize),1); YB = NaN(prod(IND.datasize),1); % YX = NaN(prod(IND.datasize),1); 
YRGB = NaN(prod(IND.datasize),1);

for ii=1:length(IND.tvoxels)
	YR(IND.tvoxels(ii))= NR(ii); YG(IND.tvoxels(ii))= NG(ii); YB(IND.tvoxels(ii))= NB(ii); % YX(IND.tvoxels(ii))= NX(ii); 
	YRGB(IND.tvoxels(ii))= NRGB(ii);
end


niifile = [ PRM.SaveDir IND.file_prefix '_VarPart' ];
% vol = reshape(YR,IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm; MRIwrite(V,[niifile '_R.nii']);
% vol = reshape(YB,IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm; MRIwrite(V,[niifile '_B.nii']);
% vol = reshape(YG,IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm; MRIwrite(V,[niifile '_G.nii']);
vol = reshape(YRGB,IND.datasize); vol_perm = permute(vol, [2,1,3]); V = MRIread(IND.RefEPI); V.vol = vol_perm; MRIwrite(V,[niifile '_RGB.nii']);

end
