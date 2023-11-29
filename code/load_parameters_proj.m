function PRM = load_parameters_proj
% load_parameters_proj.m
% 
% Edit:	ProjectDir	: project directory
%
%		LLM			: large language model used
%
%		LLMdim		: hidden units of LLM


%%
SubjectsList = { '01', '02', '03', '04', '05', '06', '07', '08' };
group_name = '01_02_03_04_05_06_07_08';


ProjectDir = '/home/open/dialogue/';

data_from_openneuro = [ ProjectDir '/derivative/' ];
code_from_github = [ ProjectDir '/code/' ];
SaveDir = [ data_from_openneuro '/results/'];

if ~exist(SaveDir)
    mkdir(SaveDir)
end

addpath(code_from_github);
addpath([code_from_github 'util']);


%%
LLM='CHATGPTNEOX';
LLMdim=2816;
taskname='task-dialogue';


% Features used in the full model
%   CCG: cochleagram 
%	SYL: syllable count
%	MRPL: morpheme count
%	POS: part-of-speech
%	MOT: head motion
Features={ LLM, 'CCG', 'SYL', 'MRPL', 'POS', 'MOT' };
FullModel_name=[LLM '_CCG_SYL_MRPL_POS_MOT'];
Features_index = 1:length(Features);


% number of features
CCGdim=79;
SYLdim=1;
MRPLdim=1;
POSdim=11;
MOTdim=6;
PRODdim=LLMdim+POSdim+SYLdim+MRPLdim+CCGdim+MOTdim;
COMPdim=LLMdim+POSdim+SYLdim+MRPLdim+CCGdim;
ALLdim=PRODdim+COMPdim;


Prod_index = [  ALLdim*0+1:ALLdim*0+LLMdim 
                ALLdim*1+1:ALLdim*1+LLMdim 
                ALLdim*2+1:ALLdim*2+LLMdim 
                ALLdim*3+1:ALLdim*3+LLMdim 
                ALLdim*4+1:ALLdim*4+LLMdim 
                ALLdim*5+1:ALLdim*5+LLMdim  ];

Comp_index = [  PRODdim+1+ALLdim*0:PRODdim+LLMdim+ALLdim*0
                PRODdim+1+ALLdim*1:PRODdim+LLMdim+ALLdim*1 
                PRODdim+1+ALLdim*2:PRODdim+LLMdim+ALLdim*2 
                PRODdim+1+ALLdim*3:PRODdim+LLMdim+ALLdim*3 
                PRODdim+1+ALLdim*4:PRODdim+LLMdim+ALLdim*4 
                PRODdim+1+ALLdim*5:PRODdim+LLMdim+ALLdim*5  ];  


% (liguistic, spectrogram, morpheme, syllable, part-of-speech)
POS_Prod_index = [  ALLdim*0+LLMdim+CCGdim+SYLdim+MRPLdim+1:ALLdim*0+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim 
                    ALLdim*1+LLMdim+CCGdim+SYLdim+MRPLdim+1:ALLdim*1+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim 
                    ALLdim*2+LLMdim+CCGdim+SYLdim+MRPLdim+1:ALLdim*2+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim 
                    ALLdim*3+LLMdim+CCGdim+SYLdim+MRPLdim+1:ALLdim*3+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim 
                    ALLdim*4+LLMdim+CCGdim+SYLdim+MRPLdim+1:ALLdim*4+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim 
                    ALLdim*5+LLMdim+CCGdim+SYLdim+MRPLdim+1:ALLdim*5+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim ];

POS_Comp_index = [  ALLdim*0+PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+1:PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim+ALLdim*0
                    ALLdim*1+PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+1:PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim+ALLdim*1 
                    ALLdim*2+PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+1:PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim+ALLdim*2 
                    ALLdim*3+PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+1:PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim+ALLdim*3 
                    ALLdim*4+PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+1:PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim+ALLdim*4 
                    ALLdim*5+PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+1:PRODdim+LLMdim+CCGdim+SYLdim+MRPLdim+POSdim+ALLdim*5 ];                  


Features_used=[];
ii = 0;
for modality = [1 2]
    for mm = Features_index
        ii = ii + 1;
        if mm == length(Features) && modality == 2
            ; % skipping repetition for MOT
        else
            Method=char(Features{mm});
            Features_used=[Features_used char(num2str(ii))];
        end    
    end
end


%% regularization parameters for ridge regression
Abase = 10;
ridge_as = Abase*(2.^(0:17));


% FDR q value
Q = 0.05; 


% Use top Nvoxel for PCA
PCANvoxel=10000;


% FIR model delay
nDelay=6;


% PCA bootstrap
nboot=1000;
topNcomp=20;


PRM.SubjectsList=SubjectsList;
PRM.group_name=group_name;
PRM.LLM=LLM;
PRM.ProjectDir=ProjectDir;
PRM.SaveDir=SaveDir;
PRM.LLMdim=LLMdim;
PRM.taskname=taskname;
PRM.Features=Features;
PRM.FullModel_name=FullModel_name;
PRM.Features_index=Features_index;
PRM.Features_used=Features_used;
PRM.CCGdim=CCGdim;
PRM.SYLdim=SYLdim;
PRM.MRPLdim=MRPLdim;
PRM.POSdim=POSdim;
PRM.MOTdim=MOTdim;
PRM.PRODdim=PRODdim;
PRM.COMPdim=COMPdim;
PRM.ALLdim=ALLdim;
PRM.Prod_index=Prod_index;
PRM.Comp_index=Comp_index;
PRM.POS_Prod_index=POS_Prod_index;
PRM.POS_Comp_index=POS_Comp_index;
PRM.ridge_as=ridge_as;
PRM.Q=Q;
PRM.PCANvoxel=PCANvoxel;
PRM.nDelay=nDelay;
PRM.nboot=nboot;
PRM.topNcomp=topNcomp;

end
