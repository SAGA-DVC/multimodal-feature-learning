import ml_collections

def load_config():

    cfg = ml_collections.ConfigDict()
   
    # General
    cfg.seed = 0
    cfg.device = 'cuda'
    cfg.is_distributed = True

    cfg.batch_size = 2
    cfg.num_workers = 2

    cfg.lr = 1e-4
    cfg.lr_drop = 200
    cfg.weight_decay = 1e-4
    
    cfg.output_dir = 'output'
    cfg.resume = 'PATH/TO/CHECHPOINTS'
    cfg.start_epoch = 0
    cfg.epochs = 5
    cfg.clip_max_norm = 0.1

    #-------------------------------------------------------------------------------------------------
    # Dataset
    cfg.dataset = ml_collections.ConfigDict()

    # Kinetics 
    cfg.dataset.kinetics = ml_collections.ConfigDict()
    cfg.dataset.kinetics.kinetics_root = '../data/sample'
    cfg.dataset.kinetics.num_temporal_samples = 10
    cfg.dataset.kinetics.frame_size = (224, 224)
    cfg.dataset.kinetics.batch_size = 3

    #-------------------------------------------------------------------------------------------------
    # Distributed training
    if cfg.is_distributed:
        cfg.distributed = ml_collections.ConfigDict()
        cfg.distributed.rank = 0
        cfg.distributed.world_size = 1
        cfg.distributed.gpu = 0
        cfg.distributed.device = 'cuda'
        cfg.distributed.dist_backend = 'nccl'
        cfg.distributed.dist_url = 'env://'
        

    #-------------------------------------------------------------------------------------------------
    # DVC model




    #-------------------------------------------------------------------------------------------------
    # DVC model
    cfg.dvc = ml_collections.ConfigDict()

    cfg.dvc.device = 'cuda'

    cfg.dvc.num_queries = 100
    cfg.dvc.aux_loss = False
    cfg.dvc.return_intermediate = False

    models = ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention']
    cfg.dvc.model_name = models[0]

    cfg.dvc.num_frames = 5
    cfg.dvc.num_patches = 196
    cfg.dvc.img_size = 224

    cfg.dvc.spatial_patch_size = 16
    cfg.dvc.temporal_patch_size = 2

    tokenization_method = ['filter inflation', 'central frame']
    cfg.dvc.tokenization_method = tokenization_method[1]

    cfg.dvc.in_channels = 3
    cfg.dvc.d_model = 768

    cfg.dvc.depth = 12
    cfg.dvc.temporal_depth = 4

    cfg.dvc.num_heads = 12
    cfg.dvc.mlp_ratio = 4
    cfg.dvc.qkv_bias = True

    cfg.dvc.positional_embedding_dropout = 0
    cfg.dvc.attention_dropout = 0
    cfg.dvc.projection_dropout = 0
    cfg.dvc.dropout_1 = 0
    cfg.dvc.dropout_2 = 0

    cfg.dvc.pre_norm = True

    cfg.dvc.classification_head = False
    cfg.dvc.num_classes = 1000

    cfg.dvc.return_preclassifier = True
    cfg.dvc.return_prelogits = False

    cfg.dvc.model_official = None
    cfg.dvc.weight_init = True
    cfg.dvc.weight_load = False

    cfg.dvc.cost_class = 1, 
    cfg.dvc.cost_segment = 1, 
    cfg.dvc.cost_giou = 1,
    cfg.dvc.cost_alpha = 0.25,
    cfg.dvc.cost_gamma = 2

    cfg.dvc.cls_loss_coef = 1
    cfg.dvc.bbox_loss_coef = 1
    cfg.dvc.giou_loss_coef = 1
    cfg.dvc.eos_coef = 1
    
    
    #-------------------------------------------------------------------------------------------------
    # Pre-trained models
    cfg.pretrained_models = ml_collections.ConfigDict()
    cfg.pretrained_models.vit = 'vit_base_patch16_224'
    cfg.pretrained_models.deit = 'deit_base_patch16_224'
    
    return cfg
