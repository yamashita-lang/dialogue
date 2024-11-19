import os
import numpy as np

LOCAL_DIR = os.path.realpath(__file__)
ROOT_DIR = '/set/your/root/directory/'

SUBJECTS_ALL = ["sub-OSU01", "sub-OSU02", "sub-OSU03", "sub-OSU04", "sub-OSU05", "sub-OSU06", "sub-OSU07", "sub-OSU08"]
TESTSET_RUNS=[  { 1: np.arange(1,7), 2: np.arange(7,16), 3: np.arange(16,19), 4: np.arange(19,28)},
                { 1: np.arange(1,8), 2: np.arange(8,17), 3: np.arange(17,26)},
                { 1: np.arange(1,9), 2: np.arange(9,18), 3: np.arange(18,28)},
                { 1: np.arange(1,8), 2: np.arange(8,16), 3: np.arange(16,25), 4: np.arange(25,28)},
                { 1: np.arange(1,8), 2: np.arange(8,17), 3: np.arange(17,26)},
                { 1: np.arange(1,8), 2: np.arange(8,16), 3: np.arange(16,25), 4: np.arange(25,28)},
                { 1: np.arange(1,8), 2: np.arange(8,17), 3: np.arange(17,24), 4: np.arange(24,28)},
                { 1: np.arange(1,8), 2: np.arange(8,17), 3: np.arange(17,26), 4: np.arange(26,28)}
            ]

TOTAL_SESS=np.array([ 4,  3,  3,  4,  3,  4,  4,  4])
TOTAL_RUNS=np.array([27, 25, 27, 27, 25, 27, 27, 27])
NVOXELS=np.array([69812, 69334, 65335, 62133, 66313, 69381, 64072, 72018])
NUTTERS=np.array([6269, 5232, 5511, 6345, 5366, 6353, 7077, 5737])


MODEL_FEATURE_MATRIX = {
                        "headmotion": ["headmotion"],
                        "gpt_nprandom": ["chatgptneox_nprandom_prod", "chatgptneox_nprandom_comp"],

                        "chatgpt_1": ["chatgptneox_prod_cl-1", "chatgptneox_comp_cl-1"],
                        "chatgpt_2": ["chatgptneox_prod_cl-2", "chatgptneox_comp_cl-2"],
                        "chatgpt_4": ["chatgptneox_prod_cl-4", "chatgptneox_comp_cl-4"],
                        "chatgpt_8": ["chatgptneox_prod_cl-8", "chatgptneox_comp_cl-8"],
                        "chatgpt_16": ["chatgptneox_prod_cl-16", "chatgptneox_comp_cl-16"],
                        "chatgpt_32": ["chatgptneox_prod_cl-32", "chatgptneox_comp_cl-32"],

                        "chatgpt_1_prod": ["chatgptneox_prod_cl-1"],
                        "chatgpt_2_prod": ["chatgptneox_prod_cl-2"],
                        "chatgpt_4_prod": ["chatgptneox_prod_cl-4"],
                        "chatgpt_8_prod": ["chatgptneox_prod_cl-8"],
                        "chatgpt_16_prod": ["chatgptneox_prod_cl-16"],
                        "chatgpt_32_prod": ["chatgptneox_prod_cl-32"],

                        "chatgpt_1_comp": ["chatgptneox_comp_cl-1"],
                        "chatgpt_2_comp": ["chatgptneox_comp_cl-2"],
                        "chatgpt_4_comp": ["chatgptneox_comp_cl-4"],
                        "chatgpt_8_comp": ["chatgptneox_comp_cl-8"],
                        "chatgpt_16_comp": ["chatgptneox_comp_cl-16"],
                        "chatgpt_32_comp": ["chatgptneox_comp_cl-32"],

                        "chatgpt_unified_1": ["chatgptneox_unified_cl-1"],
                        "chatgpt_unified_2": ["chatgptneox_unified_cl-2"],
                        "chatgpt_unified_4": ["chatgptneox_unified_cl-4"],
                        "chatgpt_unified_8": ["chatgptneox_unified_cl-8"],
                        "chatgpt_unified_16": ["chatgptneox_unified_cl-16"],
                        "chatgpt_unified_32": ["chatgptneox_unified_cl-32"],

                        }

ITER=5
DELAY=[2, 3, 4, 5, 6, 7]
ALPHAS=np.logspace(-2, 7, 10)
N_SCANS=430
CV=5
N_TARGETS_BATCH=100
N_PERM=1000
LAYERS=range(0, 37, 3)
CONTEXT_LENGTH=[1, 2, 4, 8, 16, 32]
N_GPTSIZE=2816
N_PC=20

DATA_DIR = f"{ROOT_DIR}derivative/"
FEATURES_DIR = f"{DATA_DIR}"
RESULTS_DIR = f"{DATA_DIR}results/"
FIG_DIR = f"{DATA_DIR}figures/"
PCA_DIR = f"{RESULTS_DIR}pca/"

for DIR in [ DATA_DIR, FEATURES_DIR, RESULTS_DIR, FIG_DIR, PCA_DIR ]:
    if not os.path.exists(DIR):
        os.makedirs(DIR)

