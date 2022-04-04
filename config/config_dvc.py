import ml_collections

def load_config():

    cfg = ml_collections.ConfigDict()
   
    # General
    cfg.seed = 0
    cfg.device = 'cuda:3'

    cfg.batch_size = 3
    cfg.num_workers = 2

    cfg.lr = 1e-4
    cfg.lr_drop = 200
    cfg.weight_decay = 1e-4
        
    cfg.output_dir = 'output'
    cfg.resume = None
    cfg.start_epoch = 0
    cfg.epochs = 1
    cfg.clip_max_norm = 0.1



    #-------------------------------------------------------------------------------------------------
    # Dataset
    cfg.dataset = ml_collections.ConfigDict()

    # ActivityNet
    cfg.dataset.activity_net = ml_collections.ConfigDict()

    cfg.dataset.activity_net.anet_path = '../activity-net/captions'
    # cfg.dataset.activity_net.video_folder = '../activity-net/splits'
    cfg.dataset.activity_net.features_path = './data_features'
    cfg.dataset.activity_net.invalid_videos_json = '../activity-net/captions/invalid_ids.json'

    cfg.dataset.activity_net.vocab_file_path = './vocab.pkl'

    cfg.dataset.activity_net.max_caption_len_all = 20
    cfg.dataset.activity_net.vocab_size = 5747
    
    cfg.dataset.activity_net.feature_sample_rate = 2
    cfg.dataset.activity_net.data_rescale = True
    cfg.dataset.activity_net.rescale_len = 30
    cfg.dataset.activity_net.data_norm = False

    cfg.dataset.activity_net.max_gt_target_segments = 10

    cfg.dataset.activity_net.num_queries = 100
    cfg.dataset.activity_net.num_classes = 100

    # Kinetics 
    cfg.dataset.kinetics = ml_collections.ConfigDict()
    cfg.dataset.kinetics.kinetics_root = '../data/sample'
    cfg.dataset.kinetics.num_temporal_samples = 10
    cfg.dataset.kinetics.frame_size = (224, 224)
    cfg.dataset.kinetics.batch_size = 1

    #-------------------------------------------------------------------------------------------------
    # Distributed training
    cfg.distributed = ml_collections.ConfigDict()
    cfg.distributed.is_distributed = True
    cfg.distributed.rank = 0
    cfg.distributed.world_size = 1
    cfg.distributed.gpu = 0
    cfg.distributed.device = 'cuda'
    cfg.distributed.dist_backend = 'nccl'
    cfg.distributed.dist_url = 'env://'


    #-------------------------------------------------------------------------------------------------
    # DVC model
    cfg.dvc = ml_collections.ConfigDict()

    cfg.dvc.glove_file_path = '../dvc/data/glove.6B.50d.txt'
    cfg.dvc.pretrained_word_embed_dim = 50
    cfg.dvc.emb_weights_req_grad = False
    cfg.dvc.embedding_matrix_file_path = 'embedding_matrix.pkl'

    cfg.dvc.num_queries = 100
    cfg.dvc.aux_loss = False
    cfg.dvc.return_intermediate = False

    models = ['spatio temporal attention', 'factorised encoder', 'factorised self attention', 'factorised dot product attention']
    cfg.dvc.model_name = models[0]

    cfg.dvc.num_frames_in = 30
    cfg.dvc.img_size = 224

    cfg.dvc.spatial_patch_size = 16
    cfg.dvc.temporal_patch_size = 2

    tokenization_method = ['filter inflation', 'central frame']
    cfg.dvc.tokenization_method = tokenization_method[1]

    cfg.dvc.in_channels = 3
    cfg.dvc.d_model = 768

    cfg.dvc.depth = 2
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

    cfg.dvc.cost_class = 1 
    cfg.dvc.cost_segment = 1 
    cfg.dvc.cost_giou = 1
    cfg.dvc.cost_alpha = 0.25
    cfg.dvc.cost_gamma = 2.0

    cfg.dvc.smoothing = 0.1

    cfg.dvc.cls_loss_coef = 1
    cfg.dvc.bbox_loss_coef = 1
    cfg.dvc.giou_loss_coef = 1
    cfg.dvc.captions_loss_coef = 1
    cfg.dvc.eos_coef = 1


    #-------------------------------------------------------------------------------------------------
    # Deformable DETR
    cfg.dvc.detr = ml_collections.ConfigDict()

    cfg.dvc.detr.feature_dim = 768  #   dim of frame-level feature vector (default = 500)
    cfg.dvc.detr.hidden_dim = 768   #   Dimensionality of the hidden layer in the feed-forward networks within the Transformer
    cfg.dvc.detr.num_queries = 100  #   number of queries givin to decoder
    cfg.dvc.detr.hidden_dropout_prob = 0.5
    cfg.dvc.detr.layer_norm_eps = 1e-12 
    cfg.dvc.detr.nheads = 12 #   the number of heads in the multiheadattention models
    cfg.dvc.detr.num_feature_levels = 4  #  number of feature levels in multiscale Deformable Attention 
    cfg.dvc.detr.dec_n_points = 4   #   number of sampling points per attention head per feature level for decoder
    cfg.dvc.detr.enc_n_points = 4   #   number of sampling points per attention head per feature level for encoder
    cfg.dvc.detr.enc_layers = 2 #   number of sub-encoder-layers in the encoder
    cfg.dvc.detr.dec_layers = 2 #   number of sub-decoder-layers in the decode
    cfg.dvc.detr.transformer_dropout_prob = 0.1 #   the dropout value
    cfg.dvc.detr.transformer_ff_dim = 2048  #    the dimension of the feedforward network model



    
    #-------------------------------------------------------------------------------------------------
    # Pre-trained models
    cfg.pretrained_models = ml_collections.ConfigDict()
    cfg.pretrained_models.vit = 'vit_base_patch16_224'
    cfg.pretrained_models.deit = 'deit_base_patch16_224'
    

    #-------------------------------------------------------------------------------------------------
    # Wandb (Weights and Biases)
    cfg.wandb = ml_collections.ConfigDict()
    cfg.wandb.on = False
    cfg.wandb.project = "simple-end-to-end"
    cfg.wandb.entity = "saga-dvc"
    cfg.wandb.notes = "Testing the flow of the DVC model"


    #-------------------------------------------------------------------------------------------------
    # Evaluate inferences
    cfg.eval = ml_collections.ConfigDict()
    cfg.eval.submission = 'output/test.json'
    cfg.eval.references = ['../activity-net/captions/val_1.json', '../activity-net/captions/val_2.json']
    cfg.eval.tious = [0.3, 0.5, 0.7, 0.9]
    cfg.eval.max_proposals_per_video = 100
    cfg.eval.verbose = False

    return cfg
