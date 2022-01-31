import ml_collections

def load_config():

    cfg = ml_collections.ConfigDict()
   
    # Dataset
    cfg.dataset = ml_collections.ConfigDict()

    # Kinetics 
    cfg.dataset.kinetics = ml_collections.ConfigDict()
    cfg.dataset.kinetics.kinetics_root = '../data/sample'
    cfg.dataset.kinetics.num_temporal_samples = 10
    cfg.dataset.kinetics.frame_size = (224, 224)
    cfg.dataset.kinetics.batch_size = 3

    #-------------------------------------------------------------------------------------------------
    # ViViT
    cfg.vivit = ml_collections.ConfigDict()

    models = ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention']
    cfg.vivit.model_name = models[0]

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
    cfg.vivit.num_classes = 1000

    cfg.vivit.return_preclassifier = True
    cfg.vivit.return_prelogits = False

    cfg.vivit.weight_init = False
    cfg.vivit.weight_load = True


    #-------------------------------------------------------------------------------------------------
    # Bimodal encoder
    cfg.bimodal = ml_collections.ConfigDict()

    cfg.bimodal.d_model = 768

    cfg.bimodal.depth = 12

    cfg.bimodal.num_heads = 12
    cfg.bimodal.mlp_ratio = 4
    cfg.bimodal.qkv_bias = True

    cfg.bimodal.attention_dropout = 0
    cfg.bimodal.projection_dropout = 0
    cfg.bimodal.dropout_1 = 0
    cfg.bimodal.dropout_2 = 0

    cfg.bimodal.classification_head = False
    cfg.bimodal.num_classes = 400

    cfg.bimodal.return_preclassifier = False
    cfg.bimodal.return_prelogits = True

    cfg.bimodal.weight_init = True
    cfg.bimodal.weight_load = False

    #-------------------------------------------------------------------------------------------------
    # Decoder
    cfg.decoder = ml_collections.ConfigDict()
    
    cfg.decoder.d_model = 768
    cfg.decoder.depth = 12
    cfg.decoder.num_heads = 12
    cfg.decoder.mlp_ratio = 4
    cfg.decoder.qkv_bias = True
    
    cfg.decoder.attention_dropout = 0
    cfg.decoder.projection_dropout = 0
    cfg.decoder.dropout_1 = 0
    cfg.decoder.dropout_2 = 0

    cfg.decoder.pre_norm=True
    cfg.bimodal.weight_init = True
    cfg.bimodal.weight_load = False

    #-------------------------------------------------------------------------------------------------
    # Transformer
    cfg.transformer = ml_collections.ConfigDict()

    models = ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention']
    cfg.transformer.model_name = models[0]

    cfg.transformer.num_frames = 5
    cfg.transformer.num_patches = 196
    cfg.transformer.img_size = 224

    cfg.transformer.spatial_patch_size = 16
    cfg.transformer.temporal_patch_size = 2

    tokenization_method = ['filter inflation', 'central frame']
    cfg.transformer.tokenization_method = tokenization_method[1]

    cfg.transformer.in_channels = 3
    cfg.transformer.d_model = 768

    cfg.transformer.depth = 12
    cfg.transformer.temporal_depth = 4

    cfg.transformer.num_heads = 12
    cfg.transformer.mlp_ratio = 4
    cfg.transformer.qkv_bias = True

    cfg.transformer.positional_embedding_dropout = 0
    cfg.transformer.attention_dropout = 0
    cfg.transformer.projection_dropout = 0
    cfg.transformer.dropout_1 = 0
    cfg.transformer.dropout_2 = 0

    cfg.transformer.classification_head = False
    cfg.transformer.num_classes = 1000

    cfg.transformer.return_preclassifier = True
    cfg.transformer.return_prelogits = False

    cfg.transformer.weight_init = True
    cfg.transformer.weight_load = False
    
    
    #-------------------------------------------------------------------------------------------------
    # Pre-trained models
    cfg.pretrained_models = ml_collections.ConfigDict()
    cfg.pretrained_models.vit = 'vit_base_patch16_224'
    cfg.pretrained_models.deit = 'deit_base_patch16_224'
    
    return cfg
