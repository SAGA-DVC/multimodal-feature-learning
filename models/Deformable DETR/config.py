import ml_collections
def load_config():

    cfg = ml_collections.ConfigDict()
   

    #-------------------------------------------------------------------------------------------------
    # Deformable DETR
    cfg.detr = ml_collections.ConfigDict()

    cfg.detr.feature_dim = 500  #   dim of frame-level feature vector (default = 500)
    cfg.detr.hidden_dim = 512   #   Dimensionality of the hidden layer in the feed-forward networks within the Transformer
    cfg.detr.num_queries = 100  #   number of queries givin to decoder
    cfg.detr.hidden_dropout_prob = 0.5
    cfg.detr.layer_norm_eps = 1e-12 
    cfg.detr.nheads = 8 #   the number of heads in the multiheadattention models
    cfg.detr.num_feature_levels = 4  #  number of feature levels in multiscale Deformable Attention 
    cfg.detr.dec_n_points = 4   #   number of sampling points per attention head per feature level for decoder
    cfg.detr.enc_n_points = 4   #   number of sampling points per attention head per feature level for encoder
    cfg.detr.enc_layers = 6 #   number of sub-encoder-layers in the encoder
    cfg.detr.dec_layers = 6 #   number of sub-decoder-layers in the decode
    cfg.detr.transformer_dropout_prob = 0.1 #   the dropout value
    cfg.detr.transformer_ff_dim = 2048  #    the dimension of the feedforward network model
    return cfg

