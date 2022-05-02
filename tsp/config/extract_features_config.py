from ml_collections import ConfigDict

def load_config():

    cfg = ConfigDict()
   
    # Dataset
    cfg.dataset = ConfigDict()
    cfg.dataset.unavailable_videos = 'tsp/dataset/unavailable-videos.json'

    cfg.metadata_csv_filename = "tsp/dataset/train-metadata.csv"
    # cfg.metadata_csv_filename = "tsp/dataset/val-metadata.csv"
    # cfg.metadata_csv_filename = "tsp/dataset/test-metadata.csv"
    
    #-------------------------------------------------------------------------------------------------
    # Video
    cfg.video = ConfigDict()
    cfg.video.clip_len = 16  # Number of frames per clip
    cfg.video.frame_rate = 30  # Frames-per-second rate at which the videos are sampled
    cfg.video.clips_per_segment = 5 # Number of clips sampled per video segment
    cfg.video.stride = 16

    #-------------------------------------------------------------------------------------------------
    # Audio
    cfg.audio = ConfigDict()
    
    # AST
    cfg.audio.num_mel_bins = 128
    cfg.audio.target_length = 64

    # VGGish
    # cfg.audio.num_mel_bins = 64
    # cfg.audio.target_length = 96

    #-------------------------------------------------------------------------------------------------
    # ViViT
    cfg.vivit = ConfigDict()

    models = ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention']
    cfg.vivit.model_name = models[0]

    cfg.vivit.num_frames_in = cfg.video.clip_len
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

    cfg.vivit.positional_embedding_dropout = 0.
    cfg.vivit.attention_dropout = 0.
    cfg.vivit.projection_dropout = 0.
    cfg.vivit.dropout_1 = 0.2
    cfg.vivit.dropout_2 = 0.2

    cfg.vivit.pre_norm = True

    cfg.vivit.classification_head = False

    cfg.vivit.num_classes = 1000

    cfg.vivit.return_preclassifier = False
    cfg.vivit.return_prelogits = True

    cfg.vivit.weight_init = False
    cfg.vivit.weight_load = False

    #-------------------------------------------------------------------------------------------------
    # AST
    cfg.ast = ConfigDict()

    cfg.ast.fstride = 10
    cfg.ast.tstride = 10
    cfg.ast.input_fdim = 128
    cfg.ast.input_tdim = 64
    cfg.ast.imagenet_pretrained = True
    cfg.ast.model_size='base224'

    cfg.ast.depth = 12
    
    cfg.ast.return_preclassifier = False  # Set True for Feature extraction
    cfg.ast.return_prelogits = True  # Set True for TSP & GVF extraction

    #-------------------------------------------------------------------------------------------------
    # Pre-trained models
    cfg.pretrained_models = ConfigDict()
    cfg.pretrained_models.vit = 'vit_base_patch16_224'

    ast_tiny224 = 'deit_tiny_distilled_patch16_224'
    ast_small224 = 'vit_deit_small_distilled_patch16_224'
    ast_base224 = 'vit_deit_base_distilled_patch16_224'
    ast_base384 = 'vit_deit_base_distilled_patch16_384'

    cfg.pretrained_models.ast = "deit_base_patch16_224"
    cfg.pretrained_models.ast_audioset = "/home/arnavshah/pretrained-weights/ast-weights.pth"

    cfg.pretrained_models.vivit = "/home/arnavshah/pretrained-weights/vivit-weights-tempPatch2-numTokens1569.pt"
    
    #-------------------------------------------------------------------------------------------------
    # TSP specific
    cfg.tsp = ConfigDict()

    # One to one matching between modalities and backbones
    cfg.tsp.modalities = ['video', 'audio']
    cfg.tsp.backbones = ['vivit', 'ast']

    #-------------------------------------------------------------------------------------------------

    # General
    cfg.device = 'cuda:0'
    # cfg.device = 'cpu'

    cfg.data_dir = '/home/arnavshah/activity-net/30fps_splits'  # Path to root directory containing the videos files
    cfg.subdir = 'train'
    cfg.output_dir = '/home/arnavshah/tsp/video-features-vivit-ast/train'  # Path for saving checkpoints and results output

    cfg.batch_size = 64  # Batch size per GPU
    cfg.num_workers = 8  # Number of data loading workers


    cfg.num_shards = 4
    cfg.shard_id = 0
    # cfg.r2plus1d_34_weights = '/home/arnavshah/pretrained-weights/r2plus1d_34_max_gvf_anet.pth'

    cfg.local_checkpoint = None
    # cfg.local_checkpoint = "/home/arnavshah/tsp/tsp-output-vivit-Kpretrained/epoch_4.pth"

    return cfg
