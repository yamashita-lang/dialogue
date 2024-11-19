from dialogue.modeling.pca_stimulus_utils import indiv_stim_pca, get_indiv_stim_pcloads_txt
from dialogue.config import SUBJECTS_ALL, LAYERS, CONTEXT_LENGTH

model_base ='chatgpt' 


for cl in CONTEXT_LENGTH:
    for layer in LAYERS:
        model=f'{model_base}_{cl}'
        for subject_idx in range(len(SUBJECTS_ALL)):
            # modality_idx = 0: production, 1: comprehension
            indiv_stim_pca(subject_idx, modality_idx=0, layer=layer, cl=cl)
            indiv_stim_pca(subject_idx, modality_idx=1, layer=layer, cl=cl)

            get_indiv_stim_pcloads_txt(subject_idx, modality_idx=0, layer=layer, cl=cl)
            get_indiv_stim_pcloads_txt(subject_idx, modality_idx=1, layer=layer, cl=cl)
            