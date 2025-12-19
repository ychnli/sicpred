""" 
Global configurations for the CESM dataset 
"""

DATA_DIRECTORY = '/oak/stanford/groups/earlew/yuchen'

RAW_DATA_DIRECTORY = '/oak/stanford/groups/earlew/yuchen/cesm_lens/combined_data'

PROCESSED_DATA_DIRECTORY = '/scratch/users/yucli/cesm_data_processed'

MODEL_DIRECTORY = '/scratch/users/yucli/sicpred_models'

PREDICTIONS_DIRECTORY = '/scratch/users/yucli/sicpred_model_predictions'

ANALYSIS_RESULTS_DIRECTORY = '/oak/stanford/groups/earlew/yuchen/sicpred/analysis_results'

# all renamed variables 
ALL_VAR_NAMES = ["icefrac", "sst", "geopotential", "psl", "t2m"]

AVAILABLE_CESM_MEMBERS = [
    'r10i1181p1f1', 'r10i1231p1f1', 'r10i1251p1f1', 'r10i1281p1f1',
    'r10i1301p1f1', 'r1i1001p1f1', 'r1i1231p1f1', 'r1i1251p1f1',
    'r1i1281p1f1', 'r1i1301p1f1', 'r2i1251p1f1', 'r2i1281p1f1',
    'r2i1301p1f1', 'r3i1041p1f1', 'r3i1231p1f1', 'r3i1251p1f1',
    'r3i1281p1f1', 'r3i1301p1f1', 'r4i1061p1f1', 'r4i1231p1f1',
    'r4i1251p1f1', 'r4i1281p1f1', 'r4i1301p1f1', 'r5i1081p1f1',
    'r5i1231p1f1', 'r5i1251p1f1', 'r2i1021p1f1', 'r2i1231p1f1',
    'r5i1281p1f1', 'r5i1301p1f1', 'r6i1101p1f1', 'r6i1231p1f1',
    'r6i1251p1f1', 'r6i1281p1f1', 'r6i1301p1f1', 'r7i1121p1f1',
    'r7i1231p1f1', 'r7i1251p1f1', 'r7i1281p1f1', 'r7i1301p1f1',
    'r8i1141p1f1', 'r8i1231p1f1', 'r8i1251p1f1', 'r8i1281p1f1',
    'r8i1301p1f1', 'r9i1161p1f1', 'r9i1231p1f1', 'r9i1251p1f1',
    'r9i1281p1f1', 'r9i1301p1f1', 'r11i1231p1f2', 'r11i1251p1f2',
    'r11i1281p1f2', 'r11i1301p1f2', 'r12i1231p1f2', 'r12i1251p1f2',
    'r12i1281p1f2', 'r12i1301p1f2', 'r13i1231p1f2', 'r13i1251p1f2',
    'r13i1281p1f2', 'r13i1301p1f2', 'r14i1231p1f2', 'r14i1251p1f2',
    'r14i1281p1f2', 'r14i1301p1f2', 'r15i1231p1f2', 'r15i1251p1f2',
    'r15i1281p1f2', 'r15i1301p1f2', 'r16i1231p1f2', 'r16i1251p1f2',
    'r16i1281p1f2', 'r16i1301p1f2', 'r17i1231p1f2', 'r17i1251p1f2',
    'r17i1281p1f2', 'r17i1301p1f2', 'r18i1231p1f2', 'r18i1251p1f2',
    'r18i1281p1f2', 'r18i1301p1f2', 'r19i1231p1f2', 'r19i1251p1f2',
    'r19i1281p1f2', 'r19i1301p1f2', 'r20i1231p1f2', 'r20i1251p1f2',
    'r20i1281p1f2', 'r20i1301p1f2', 'r1i1011p1f2', 'r2i1031p1f2', 
    'r3i1051p1f2', 'r4i1071p1f2', 'r5i1091p1f2', 'r6i1111p1f2', 
    'r7i1131p1f2', 'r8i1151p1f2', 'r9i1171p1f2', 'r10i1191p1f2',
]
