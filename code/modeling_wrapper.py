import sys
import argparse
from modeling.modeling_utils import run_headmotion_analysis, run_random_embedding_analysis, run_llm_analysis
from config import SUBJECTS_ALL, MODELS_ALL, TOTAL_SESS, LAYERS, CONTEXT_LENGTH, ROOT_DIR
parser=argparse.ArgumentParser(description='Run himalaya for each subject')
parser.add_argument('subject_idx', type=int, help='subject_idx (e.g., 1 for sub-OSU01)')
args=parser.parse_args()
subj_id=args.subject_idx-1


headmotion_model="headmotion"
random_model="gpt_nprandom"
model_base='chatgpt'

####################################################################################
### Step1/3 - Head Motion model 
for ses in range(1, TOTAL_SESS[subj_id]+1):
    run_headmotion_analysis(subj_id, ses, headmotion_model)

####################################################################################
### Step2/3 - Random Embedding model 
for ses in range(1, TOTAL_SESS[subj_id]+1):
    run_random_embedding_analysis(subj_id, ses, random_model, headmotion_model)

####################################################################################
### Step3/3 - Separate and Unified Linguistic model 
for cl in CONTEXT_LENGTH:
    for layer in LAYERS:
        for ses in range(1, TOTAL_SESS[subj_id]+1):
            ####################################################################################
            ### Separate Linguistic model
            model=f"{model_base}_{cl}"
            run_llm_analysis(subj_id, ses, model, headmotion_model, random_model, layer=layer, cl=cl)

            ####################################################################################
            ### Production model
            model=f"{model_base}_{cl}_prod"
            run_llm_analysis(subj_id, ses, model, headmotion_model, random_model, layer=layer, cl=cl)
            
            ####################################################################################
            ### Comprehension model
            model=f"{model_base}_{cl}_comp"
            run_llm_analysis(subj_id, ses, model, headmotion_model, random_model, layer=layer, cl=cl)

            ####################################################################################
            ### Unified Linguistic model
            model=f"{model_base}_unified_{cl}"
            run_llm_analysis(subj_id, ses, model, headmotion_model, random_model, layer=layer, cl=cl)
