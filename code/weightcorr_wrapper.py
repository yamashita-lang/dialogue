from dialogue.modeling.weightcorr_utils import weight_corr, weight_corr_unified
from dialogue.config import LAYERS, CONTEXT_LENGTH
import argparse
parser=argparse.ArgumentParser(description='Run himalaya for each subject')
parser.add_argument('subject_idx', type=int, help='subject_idx (e.g., 1 for sub-OSU01)')
args=parser.parse_args()
subj_id=args.subject_idx-1



basemodel='gpt_nprandom'
model_prefix='chatgpt'

for cl in CONTEXT_LENGTH:
    model=f'{model_prefix}_{cl}'
    for layer in LAYERS:
        weight_corr(subj_id, model, basemodel, layer=layer, cl=cl) 

        unified_model_prefix='{model_prefix}_unified'
        unified_model=f'{unified_model_prefix}_{cl}'
        weight_corr_unified(subj_id, model, basemodel, unified_model, unified_model_prefix, layer=layer, cl=cl)
