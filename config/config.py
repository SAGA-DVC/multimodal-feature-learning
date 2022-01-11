import ml_collections

def load_config():

    cfg = ml_collections.ConfigDict()
   
    # Dataset
    cfg.dataset = ml_collections.ConfigDict()

    # Kinetics 
    cfg.dataset.kinetics = ml_collections.ConfigDict()
    cfg.dataset.kinetics.kinetics_root = '../../data/sample'
    cfg.dataset.kinetics.num_temporal_samples = 10
    cfg.dataset.kinetics.frame_size = (224, 224)
    cfg.dataset.kinetics.batch_size = 3

    #-------------------------------------------------------------------------------------------------
    # ViViT
    cfg.vivit = ml_collections.ConfigDict()

    models = ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention']
    cfg.vivit.model_name = models[3]

    cfg.vivit.num_frames = 5
    cfg.vivit.num_patches = 196
    cfg.vivit.img_size = 224

    cfg.vivit.spatial_patch_size = 16
    cfg.vivit.temporal_patch_size = 2

    tokenization_method = ['filter inflation', 'central frame']
    cfg.vivit.tokenization_method = tokenization_method[1]

    cfg.vivit.in_channels = 3
    cfg.vivit.d_model = 768

    cfg.vivit.depth = 12
    cfg.vivit.temporal_depth = 4

    cfg.vivit.num_heads = 12
    cfg.vivit.mlp_ratio = 4
    cfg.vivit.qkv_bias = True

    cfg.vivit.positional_embedding_dropout = 0
    cfg.vivit.attention_dropout = 0
    cfg.vivit.projection_dropout = 0
    cfg.vivit.dropout_1 = 0
    cfg.vivit.dropout_2 = 0

    cfg.vivit.classification_head = False
    cfg.vivit.num_classes = 400

    cfg.vivit.return_preclassifier = False
    cfg.vivit.return_prelogits = False

    cfg.vivit.weight_init = True

    #-------------------------------------------------------------------------------------------------
    # Pre-trained models
    cfg.pretrained_models = ml_collections.ConfigDict()
    cfg.pretrained_models.for_vivit = ml_collections.ConfigDict()
    cfg.pretrained_models.for_vivit.vit = 'vit_base_patch16_224'
    cfg.pretrained_models.for_vivit.deit = ''
    
    return cfg
