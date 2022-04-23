'''
If you want to switch between Deformable DVC and regular DVC, change all parameters having the "Switch DVC" comment.

'''

import ml_collections

def load_config():

    cfg = ml_collections.ConfigDict()
   
    # General
    cfg.seed = 0
    cfg.device = 'cuda'    # change to 'cuda' when using distributed training

    cfg.batch_size = 16
    cfg.num_workers = 2

    cfg.print_freq = 10

    cfg.lr = 1e-4
    cfg.lr_drop = 200
    cfg.weight_decay = 1e-4
    cfg.clip_max_norm = 0.1

    cfg.checkpoint_rate = 10
    cfg.eval_rate = 5
        
    cfg.output_dir = 'output'
    # cfg.output_dir = 'output_temp'

    cfg.resume = 'output/checkpoint.pth'

    # cfg.resume = None

    cfg.start_epoch = 0    # set in main.py if cfg.resume is True (saved as part of the checkpoint)
    cfg.epochs = 50

    cfg.use_raw_videos = False    # Switch DVC
    cfg.use_differentiable_mask = True


    #-------------------------------------------------------------------------------------------------
    # Distributed training
    # is_distributed, rank, world_size, gpu - doesn't matter what it is in cfg. It is set in init_distributed_mode() in utils/misc.py
    cfg.distributed = ml_collections.ConfigDict()
    cfg.distributed.is_distributed = False    
    cfg.distributed.rank = 0
    cfg.distributed.world_size = 1
    cfg.distributed.gpu = 0
    # cfg.distributed.device = 'cuda'
    cfg.distributed.dist_backend = 'nccl'
    cfg.distributed.dist_url = 'env://'


    #-------------------------------------------------------------------------------------------------
    # Wandb (Weights and Biases)
    cfg.wandb = ml_collections.ConfigDict()
    cfg.wandb.on = True
    cfg.wandb.project = "simple-end-to-end"
    cfg.wandb.entity = "saga-dvc"
    cfg.wandb.notes = "DVC v2 bs 16 run"
    cfg.wandb.run_name = 'dvc-testing'


    #-------------------------------------------------------------------------------------------------
    # Dataset
    cfg.dataset = ml_collections.ConfigDict()

    # ActivityNet
    cfg.dataset.activity_net = ml_collections.ConfigDict()

    cfg.dataset.activity_net.anet_path = './anet_data'
    cfg.dataset.activity_net.raw_video_folder = '../activity-net/30fps_splits'
    cfg.dataset.activity_net.video_features_folder = '/home/arnavshah/tsp/tsp-features-r2plus1d-34'
    cfg.dataset.activity_net.invalid_videos_json = './anet_data/invalid_ids.json'

    cfg.dataset.activity_net.vocab_file_path = './vocab.pkl'
    cfg.dataset.activity_net.min_freq = 2

    cfg.dataset.activity_net.max_caption_len_all = 20
    
    data_rescale = ['interpolate', 'uniform']    # do not use uniform for now - TODO - determine rescale length
    cfg.dataset.activity_net.data_rescale = data_rescale[0]
    cfg.dataset.activity_net.video_feature_sample_rate = 2
    cfg.dataset.activity_net.video_rescale_len = 400    # Switch DVC - avg len in train is 220
    cfg.dataset.activity_net.audio_feature_sample_rate = 2
    cfg.dataset.activity_net.audio_rescale_len = 50    # Switch DVC

    cfg.dataset.activity_net.num_mel_bins = 128
    cfg.dataset.activity_net.audio_target_length = 64

    cfg.dataset.activity_net.max_gt_target_segments = 10
    cfg.dataset.activity_net.num_classes = 200    # no action class not included 


    # Kinetics 
    cfg.dataset.kinetics = ml_collections.ConfigDict()
    cfg.dataset.kinetics.kinetics_root = '../data/sample'
    cfg.dataset.kinetics.num_temporal_samples = 10
    cfg.dataset.kinetics.frame_size = (224, 224)
    cfg.dataset.kinetics.batch_size = 1



    #-------------------------------------------------------------------------------------------------
    # DVC model
    cfg.dvc = ml_collections.ConfigDict()

    # cfg.dvc.input_modalities = ['video', 'audio']
    cfg.dvc.input_modalities = ['video']
    # cfg.dvc.input_modalities = ['audio']

    cfg.dvc.num_queries = 20
    cfg.dvc.d_model = 512
    cfg.dvc.aux_loss = False
    cfg.dvc.num_classes = cfg.dataset.activity_net.num_classes

    cfg.dvc.use_deformable_detr = True    # Switch DVC

    cfg.dvc.smoothing = 0.3

    cfg.dvc.cls_loss_coef = 1
    cfg.dvc.bbox_loss_coef = 5
    cfg.dvc.giou_loss_coef = 2
    cfg.dvc.self_iou_loss_coef = 2
    cfg.dvc.captions_loss_coef = 2
    cfg.dvc.context_loss_coef = 2
    cfg.dvc.eos_coef = 0.1

    cfg.dvc.losses = ['labels', 'segments', 'cardinality', 'captions']
    if cfg.use_differentiable_mask:
        cfg.dvc.losses.append('contexts')


    # Matcher args
    cfg.dvc.matcher = ml_collections.ConfigDict()

    cfg.dvc.matcher.cost_class = 1 
    cfg.dvc.matcher.cost_segment = 5 
    cfg.dvc.matcher.cost_giou = 2
    cfg.dvc.matcher.cost_alpha = 0.25
    cfg.dvc.matcher.cost_gamma = 2.0


    # Deformable DETR
    cfg.dvc.detr = ml_collections.ConfigDict()

    cfg.dvc.detr.feature_dim = cfg.dvc.d_model    # dim of frame-level feature vector
    cfg.dvc.detr.d_model = cfg.dvc.d_model 
    
    cfg.dvc.detr.hidden_dropout_prob = 0.5
    cfg.dvc.detr.layer_norm_eps = 1e-12 

    cfg.dvc.detr.num_heads = 8 

    cfg.dvc.detr.num_feature_levels = 4    # number of feature levels in Multiscale Deformable Attention 
    cfg.dvc.detr.dec_n_points = 4    # number of sampling points per attention head per feature level for decoder
    cfg.dvc.detr.enc_n_points = 4    # number of sampling points per attention head per feature level for encoder

    cfg.dvc.detr.enc_layers = 6    # depth
    cfg.dvc.detr.dec_layers = 6    # depth

    cfg.dvc.detr.transformer_dropout_prob = 0.1
    cfg.dvc.detr.transformer_ff_dim = 2048
    cfg.dvc.detr.video_rescale_len = cfg.dataset.activity_net.video_rescale_len

    cfg.dvc.detr.return_intermediate = True    # TODO - check use
    

    # Sparse DETR
    cfg.dvc.sparse_detr = ml_collections.ConfigDict()
    cfg.dvc.sparse_detr.type = "audio_video"
    cfg.dvc.sparse_detr.feature_dim = 768  #   dim of frame-level feature vector (default = 500)
    cfg.dvc.sparse_detr.d_model= 768
    cfg.dvc.sparse_detr.hidden_dim = 768   #   Dimensionality of the hidden layer in the feed-forward networks within the Transformer
    cfg.dvc.sparse_detr.num_queries = 100  #   number of queries givin to decoder
    cfg.dvc.sparse_detr.hidden_dropout_prob = 0.5
    cfg.dvc.sparse_detr.layer_norm_eps = 1e-12 
    cfg.dvc.sparse_detr.num_heads = 8 #   the number of heads in the multiheadattention models
    cfg.dvc.sparse_detr.num_feature_levels = 4  #  number of feature levels in multiscale Deformable Attention 
    cfg.dvc.sparse_detr.dec_n_points = 4   #   number of sampling points per attention head per feature level for decoder
    cfg.dvc.sparse_detr.enc_n_points = 4   #   number of sampling points per attention head per feature level for encoder
    cfg.dvc.sparse_detr.enc_layers = 6 #   number of sub-encoder-layers in the encoder
    cfg.dvc.sparse_detr.dec_layers = 6 #   number of sub-decoder-layers in the decode
    cfg.dvc.sparse_detr.transformer_dropout_prob = 0.1 #   the dropout value
    cfg.dvc.sparse_detr.transformer_ff_dim = 2048  #    the dimension of the feedforward network model
    cfg.dvc.sparse_detr.rho=0.3 
    cfg.dvc.sparse_detr.use_enc_aux_loss=False
    cfg.dvc.sparse_detr.eff_query_init=False
    cfg.dvc.sparse_detr.eff_specific_head=False
    cfg.dvc.sparse_detr.return_intermediate=False

    # Caption Decoder
    # vocab_size, seq_len, embedding_matrix - these parameters are set in /models/__init__.py
    cfg.dvc.caption = ml_collections.ConfigDict()

    cfg.dvc.caption.d_model = cfg.dvc.d_model

    cfg.dvc.caption.depth = 6

    cfg.dvc.caption.num_heads = 8
    cfg.dvc.caption.mlp_ratio = 4
    cfg.dvc.caption.qkv_bias = True

    cfg.dvc.caption.positional_embedding_dropout = 0.1
    cfg.dvc.caption.attention_dropout = 0.1
    cfg.dvc.caption.projection_dropout = 0.1
    cfg.dvc.caption.bridge_dropout = 0.1
    cfg.dvc.caption.mlp_dropout_1 = 0.1
    cfg.dvc.caption.mlp_dropout_2 = 0.1

    cfg.dvc.caption.pre_norm = False

    cfg.dvc.caption.model_official = None
    cfg.dvc.caption.weight_init = True
    cfg.dvc.caption.weight_load = False

    cfg.dvc.caption.emb_weights_req_grad = True
    cfg.dvc.caption.return_intermediate = False

    # TODO - handle embedding matrix loading better
    cfg.dvc.caption.pretrained_word_embed_dim = 300
    cfg.dvc.caption.glove_file_path = f'../dvc/data/glove.6B.{cfg.dvc.caption.pretrained_word_embed_dim}d.txt'
    # cfg.dvc.caption.glove_file_path = f'../dvc/data/glove.840B.300d.txt'
    cfg.dvc.caption.embedding_matrix_file_path = 'embedding_matrix.pkl'


    # ViViT
    cfg.dvc.vivit = ml_collections.ConfigDict()

    models = ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention']
    cfg.dvc.vivit.model_name = models[0]

    cfg.dvc.vivit.num_frames_in = 30
    cfg.dvc.vivit.img_size = 224

    cfg.dvc.vivit.spatial_patch_size = 16
    cfg.dvc.vivit.temporal_patch_size = 2

    cfg.dvc.vivit.num_frames = cfg.dvc.vivit.num_frames_in // cfg.dvc.vivit.temporal_patch_size
    cfg.dvc.vivit.num_patches = (cfg.dvc.vivit.img_size // cfg.dvc.vivit.spatial_patch_size) ** 2

    tokenization_method = ['filter inflation', 'central frame']
    cfg.dvc.vivit.tokenization_method = tokenization_method[1]

    cfg.dvc.vivit.in_channels = 3
    cfg.dvc.vivit.d_model = cfg.dvc.d_model

    cfg.dvc.vivit.depth = 2
    cfg.dvc.vivit.temporal_depth = 4

    cfg.dvc.vivit.num_heads = 12
    cfg.dvc.vivit.mlp_ratio = 4
    cfg.dvc.vivit.qkv_bias = True

    cfg.dvc.vivit.positional_embedding_dropout = 0.1
    cfg.dvc.vivit.attention_dropout = 0.1
    cfg.dvc.vivit.projection_dropout = 0.1
    cfg.dvc.vivit.mlp_dropout_1 = 0.2
    cfg.dvc.vivit.mlp_dropout_2 = 0.2

    cfg.dvc.vivit.pre_norm = True

    cfg.dvc.vivit.classification_head = False
    cfg.dvc.vivit.num_classes = cfg.dvc.num_classes

    cfg.dvc.vivit.return_preclassifier = True
    cfg.dvc.vivit.return_prelogits = False

    cfg.dvc.vivit.model_official = None
    cfg.dvc.vivit.weight_init = True
    cfg.dvc.vivit.weight_load = False


    # AST
    cfg.dvc.ast = ml_collections.ConfigDict()

    cfg.dvc.ast.fstride = 10
    cfg.dvc.ast.tstride = 10
    cfg.dvc.ast.input_fdim = 128
    cfg.dvc.ast.input_tdim = 64

    cfg.dvc.ast.imagenet_pretrained = True
    cfg.dvc.ast.model_size='base224'

    cfg.dvc.ast.depth = 2
    
    cfg.dvc.ast.return_preclassifier = True  # Set True for Feature extraction
    cfg.dvc.ast.return_prelogits = False  # Set True for TSP & GVF extraction

    # Decoder
    cfg.dvc.decoder = ml_collections.ConfigDict()

    cfg.dvc.decoder.d_model = cfg.dvc.d_model

    cfg.dvc.decoder.depth = 2

    cfg.dvc.decoder.num_heads = 8
    cfg.dvc.decoder.mlp_ratio = 4
    cfg.dvc.decoder.qkv_bias = True

    cfg.dvc.decoder.positional_embedding_dropout = 0.1
    cfg.dvc.decoder.attention_dropout = 0.1
    cfg.dvc.decoder.projection_dropout = 0.1
    cfg.dvc.decoder.mlp_dropout_1 = 0.2
    cfg.dvc.decoder.mlp_dropout_2 = 0.2

    cfg.dvc.decoder.pre_norm = True

    cfg.dvc.decoder.model_official = None
    cfg.dvc.decoder.weight_init = True
    cfg.dvc.decoder.weight_load = False

    cfg.dvc.decoder.return_intermediate = False

    
    #-------------------------------------------------------------------------------------------------
    # Pre-trained models
    cfg.pretrained_models = ml_collections.ConfigDict()
    cfg.pretrained_models.vit = 'vit_base_patch16_224'
    cfg.pretrained_models.deit = 'deit_base_patch16_224'


    #-------------------------------------------------------------------------------------------------
    # Evaluate inferences
    cfg.eval = ml_collections.ConfigDict()
    cfg.eval.submission = 'output/test.json'
    # cfg.eval.submission = 'sample_submission.json'
    # cfg.eval.references = ['./anet_data/val_1.json', './anet_data/val_2.json']
    cfg.eval.references = ['./anet_data/val_1.json']
    cfg.eval.tious = [0.3, 0.5, 0.7, 0.9]
    cfg.eval.max_proposals_per_video = 100
    cfg.eval.verbose = False
    cfg.eval.is_submission_json = True

    return cfg
