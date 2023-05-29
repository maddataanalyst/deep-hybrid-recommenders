COL_USER_ID = 'userid'
COL_ITEM_ID = 'itemid'
COL_OVERALL = 'overall'
MODEL_NAME = 'model_name'

MISSING_VAL = 'missing'

USER_ID_ENCODER = 'userid_encoder'
ITEM_ID_ENCODER = 'itemid_encoder'

PARAM_TRAIN_TEST_SEED = 'random_state'
PARAM_TEST_SIZE = 'test_size'
PARAM_KFOLD = 'kfold'
PARAM_KFOLD_SEED = 'kfold_seed'

PARAM_ID_EMBED_SIZE = 'id_embed_size'
PARAM_BATCH_SIZE = 'batch_size'
PARAM_LOGDIR = 'logdir'
PARAM_LEARNING_RATE = 'lr'
PARAM_MAX_EPOCHS = 'max_epochs'
PARAM_HIDDEN_SIZES = 'hidden_sizes'

PARAM_UID_SUBNET_SPECS = 'uid_subet_specs'
PARAM_IID_SUBNET_SPECS = 'iid_subet_specs'
PARAM_CAT_EMBEDDING_SPECS = 'cat_embedding_specs'
PARAM_EMBED_CONCAT_DENSE_SPECS = 'cat_subnet_concat_specs'
PARAM_DENSE_LAYERS = 'dense_subnet_specs'
PARAM_DROPOUT = 'dropout'
PARAM_INNER_ACTIVATION_F = 'inner_activation_fn'
PARAM_GNN_SIZE = 'gnn_size'
PARAM_GNN_LAYERS = 'gnn_layers'
PARAM_GNN_TYPE = 'gnn_type'
PARAM_JK_TYPE = 'jk_type'
PARAM_UEMBED_SZ = 'uembed_sz'
PARAM_IEMBED_SIZE = 'iembed_sz'
PARAM_GNN_ACT_F = 'gnn_act_f'
PARAM_SUBNET_ACT_F = 'subnet_act_f'
PARAM_MAIN_REL = 'main_rel'
PARAM_USE_ITEM_FEATURES = 'use_item_features'
PARAM_NEG_SAMPLING_RATIO = 'neg_sampling_ratio'


METRIC_AVG_CROSSVAL_MSE = 'avg crossval mse'
METRIC_CROSSVAL_MSE_STD = 'std crossval mse'

METRIC_AVG_CROSSVAL_MAE = 'avg crossval mae'
METRIC_CROSSVAL_MAE_STD = 'std crossval mae'

METRIC_AVG_CROSSVAL_MAPE = 'avg crossval mape'
METRIC_CROSSVAL_MAPE_STD = 'std crossval mape'

MAIN_GRAPH_RELATION = ('user', 'rates', 'item')

TRAIN_PHASE = 'train'
TEST_PHASE = 'test'
VAL_PHASE = 'val'

MSE_METRIC = 'MSE'
MAE_METRIC = 'MAE'
MAPE_METRIC = 'MAPE'