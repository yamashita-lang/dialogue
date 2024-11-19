import argparse
from dialogue.modeling.perm_utils import run_rscore_perm
from dialogue.config import TOTAL_SESS, LAYERS, CONTEXT_LENGTH

parser=argparse.ArgumentParser(description='Run himalaya for each subject')
parser.add_argument('subject_idx', type=int, help='subject_idx (e.g., 1 for sub-OSU01)')
args=parser.parse_args()
subj_id=args.subject_idx-1


#####################################################################################################################################################################
### Random embedding model
basemodel="headmotion"
model="gpt_nprandom"

for ses in range(1, TOTAL_SESS[subj_id]+1):
    ### Intramodal 
    run_rscore_perm(subj_id, ses, model, basemodel, layer=0, cl=1, test_idx=0)

    ### Cross-modality
    run_rscore_perm(subj_id, ses, model, basemodel, layer=0, cl=1, test_idx=1)


#####################################################################################################################################################################
### Separate Linguistic model
basemodel="gpt_nprandom"
model_prefix='chatgpt'

for cl in CONTEXT_LENGTH:
    for layer in LAYERS:
        for ses in range(1, TOTAL_SESS[subj_id]+1):
            model=f"{model_prefix}_{cl}"

            ### Intramodal 
            run_rscore_perm(subj_id, ses, model, basemodel, layer, cl, test_idx=0)

            ### Cross-modality
            run_rscore_perm(subj_id, ses, model, basemodel, layer, cl, test_idx=1)
            
 
#####################################################################################################################################################################
### Unified Linguistic model
basemodel="gpt_nprandom"
model_prefix='chatgpt_unified'

for cl in CONTEXT_LENGTH:
    for layer in LAYERS:
        for ses in range(1, TOTAL_SESS[subj_id]+1):
            model=f"{model_prefix}_{cl}"
            run_rscore_perm(subj_id, ses, model, basemodel, layer, cl, test_idx=0)
