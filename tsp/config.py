import ml_collections

def load_config():

    cfg = ml_collections.ConfigDict()
   
    # Dataset
    cfg.dataset = ml_collections.ConfigDict()
    cfg.dataset.label_mapping_jsons = [
        "tsp/dataset/activitynet_v1-3_action_label_mapping.json", 
        "tsp/dataset/activitynet_v1-3_temporal_region_label_mapping.json"
    ]  # Paths to the mappings of each label column
    cfg.dataset.label_columns = ["action-label", "temporal-region-label"]  # Names of the label columns in the CSV files
    cfg.dataset.train_csv_filename = "tsp/dataset/activitynet_v1-3_train_tsp_groundtruth.csv"  # Path to the training CSV file
    cfg.dataset.valid_csv_filename = "tsp/dataset/activitynet_v1-3_valid_tsp_groundtruth.csv"  # Path to the validation CSV file
    cfg.dataset.unavailable_videos = 'tsp/dataset/unavailable-videos.json'

    cfg.metadata_csv_filename = "tsp/dataset/train-metadata.csv"
    # cfg.metadata_csv_filename = "tsp/dataset/val-metadata.csv"
    # cfg.metadata_csv_filename = "tsp/dataset/test-metadata.csv"
    
    #-------------------------------------------------------------------------------------------------
    # Video
    cfg.video = ml_collections.ConfigDict()
    cfg.video.clip_len = 16  # Number of frames per clip
    cfg.video.frame_rate = 30  # Frames-per-second rate at which the videos are sampled
    cfg.video.clips_per_segment = 5 # Number of clips sampled per video segment

    #-------------------------------------------------------------------------------------------------
    # Audio
    cfg.audio = ml_collections.ConfigDict()
    cfg.audio.num_mel_bins = 128
    cfg.audio.target_length = 64

    #-------------------------------------------------------------------------------------------------
    # ViViT
    cfg.vivit = ml_collections.ConfigDict()

    models = ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention']
    cfg.vivit.model_name = models[0]

    cfg.vivit.num_frames_in = 16
    cfg.vivit.img_size = 224

    cfg.vivit.spatial_patch_size = 16
    cfg.vivit.temporal_patch_size = 2

    tokenization_method = ['filter inflation', 'central frame']
    cfg.vivit.tokenization_method = tokenization_method[1]

    cfg.vivit.in_channels = 3
    cfg.vivit.d_model = 768

    cfg.vivit.depth = 6
    cfg.vivit.temporal_depth = 4

    cfg.vivit.num_heads = 12
    cfg.vivit.mlp_ratio = 4
    cfg.vivit.qkv_bias = True

    cfg.vivit.positional_embedding_dropout = 0.
    cfg.vivit.attention_dropout = 0.
    cfg.vivit.projection_dropout = 0.
    cfg.vivit.dropout_1 = 0.2
    cfg.vivit.dropout_2 = 0.2

    cfg.vivit.pre_norm = True

    cfg.vivit.classification_head = False
    # TODO Check if used
    cfg.vivit.num_classes = 1000

    cfg.vivit.return_preclassifier = False  # Set True for Feature extraction
    cfg.vivit.return_prelogits = True  # Set True for TSP & GVF extraction

    cfg.vivit.weight_init = False
    cfg.vivit.weight_load = True

    #-------------------------------------------------------------------------------------------------
    # AST
    cfg.ast = ml_collections.ConfigDict()

    cfg.ast.fstride = 10
    cfg.ast.tstride = 10
    cfg.ast.input_fdim = 128
    cfg.ast.input_tdim = 64
    cfg.ast.imagenet_pretrained = True
    cfg.ast.model_size='base224'

    cfg.ast.depth = 6
    
    cfg.ast.return_preclassifier = False  # Set True for Feature extraction
    cfg.ast.return_prelogits = True  # Set True for TSP & GVF extraction

    #-------------------------------------------------------------------------------------------------
    # Pre-trained models
    cfg.pretrained_models = ml_collections.ConfigDict()
    cfg.pretrained_models.vit = 'vit_base_patch16_224'

    ast_tiny224 = 'deit_tiny_distilled_patch16_224'
    ast_small224 = 'vit_deit_small_distilled_patch16_224'
    ast_base224 = 'vit_deit_base_distilled_patch16_224'
    ast_base384 = 'vit_deit_base_distilled_patch16_384'

    cfg.pretrained_models.ast = "deit_base_patch16_224"
    
    
    #-------------------------------------------------------------------------------------------------

    # TSP specific
    cfg.tsp = ml_collections.ConfigDict()

    # Paths to the h5 file containing global video features (GVF)
    # If None, then model will not use GVF
    cfg.tsp.train_global_video_features = "tsp/dataset/train-gvf.h5"
    cfg.tsp.val_global_video_features = "tsp/dataset/val-gvf.h5"
    cfg.tsp.backbones = ['vivit', 'ast']
    cfg.tsp.backbone_lr = 0.001  # Backbone layers learning rate
    cfg.tsp.fc_lr = 0.001
    cfg.tsp.loss_alphas = [1.0, 1.0]  # A list of the scalar alpha with which to weight each label loss

    #-------------------------------------------------------------------------------------------------

    # General
    cfg.device = 'cuda'
    # cfg.device = 'cpu'
    cfg.data_dir = '/home/arnavshah/activity-net/30fps_splits'  # Path to root directory containing the videos files
    
    cfg.train_subdir = 'train'  # Training subdirectory inside the data directory
    cfg.valid_subdir = 'val'  # Validation subdirectory inside the data directory
    cfg.output_dir = 'tsp-output'  # Path for saving checkpoints and results output

    cfg.epochs = 8
    cfg.train_only_one_epoch = False  # Train the model for only one epoch without testing on validation subset
    cfg.batch_size = 16  # Batch size per GPU
    cfg.num_workers = 8  # Number of data loading workers

    cfg.momentum = 0.9
    cfg.weight_decay = 0.005
    cfg.lr_drop = 200
    cfg.lr_gamma = 0.1

    # cfg.lr_warmup_epochs = 0 # Number of warmup epochs
    # cfg.lr_milestones = [4, 6]  # Decrease lr on milestone epoch
    # cfg.lr_gamma = 0.01  # Decrease lr by a factor of lr-gamma at each milestone epoch
    # cfg.lr_warmup_factor = 1e-5

    cfg.resume = "tsp-output/epoch_1.pth"    # Resume from checkpoint (path to checkpoint .pth)
    # cfg.resume = None 
    cfg.start_epoch = 0  # not used when resume is specified

    cfg.valid_only = False  # Test the model on the validation subset and exit

    cfg.print_freq = 50  # Print frequency in number of batches

    cfg.debug = False 
    if cfg.debug:
        # Set debug cfg here, e.g. number of samples, batch size
        cfg.epochs = 2
        cfg.batch_size=4
        cfg.print_freq = 5

    #-------------------------------------------------------------------------------------------------
    # Feature extraction, not used for TSP training
    cfg.feature_extraction = ml_collections.ConfigDict()
    cfg.feature_extraction.num_shards = 1
    cfg.feature_extraction.shard_id = 0
    cfg.feature_extraction.video_stride = 16
    cfg.feature_extraction.local_checkpoint = None
    cfg.feature_extraction.subdir = 'train'

    #-------------------------------------------------------------------------------------------------
    # Distributed Processing 
    distributed = True
    cfg.distributed = ml_collections.ConfigDict()
    if distributed:
        cfg.distributed.on = True
        cfg.distributed.sync_bn = True  # Use sync batch norm
        cfg.distributed.dist_url = "env://"  # URL used to setup dist processing (see init_process_group)
    else:
        cfg.distributed.on = False

    # Config Assertions
    assert len(cfg.dataset.label_columns) == len(cfg.dataset.label_mapping_jsons), f"Unequal number of label columns ({len(cfg.dataset.label_columns)}) and label mapping JSON files ({len(cfg.dataset.label_mapping_jsons)})"
    assert len(cfg.dataset.label_columns) == len(cfg.tsp.loss_alphas), f"Unequal number of label columns ({len(cfg.dataset.label_columns)}) and loss alphas ({len(cfg.tsp.loss_alphas)})"

    #-------------------------------------------------------------------------------------------------
    
    # Wandb (Weights and Biases)
    cfg.wandb = ml_collections.ConfigDict()
    cfg.wandb.on = True
    cfg.wandb.project = "tsp"
    cfg.wandb.entity = "saga-dvc"
    cfg.wandb.notes = "First run of TSP, continued after a break with 3 GPUs!"
    # TODO Use wandb resume

    return cfg
