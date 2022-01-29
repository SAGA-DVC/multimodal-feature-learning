import ml_collections

def load_config():

    cfg = ml_collections.ConfigDict()
   
    # # Dataset
    # cfg.dataset = ml_collections.ConfigDict()

    # # Kinetics 
    # cfg.dataset.kinetics = ml_collections.ConfigDict()
    # cfg.dataset.kinetics.kinetics_root = '../../data/sample'
    # cfg.dataset.kinetics.num_temporal_samples = 10
    # cfg.dataset.kinetics.frame_size = (224, 224)
    # cfg.dataset.kinetics.batch_size = 3

    #-------------------------------------------------------------------------------------------------
    # AST
    cfg.ast = ml_collections.ConfigDict()

    cfg.ast.label_dim = 527
    cfg.ast.fstride = 10
    cfg.ast.tstride = 10
    cfg.ast.input_fdim = 128
    cfg.ast.input_tdim = 1024
    cfg.ast.imagenet_pretrain = True
    cfg.ast.model_size='base384'

    # assume the task has 527 classes
    cfg.ast.label_dim = 527


    cfg.ast.depth = 12

    #-------------------------------------------------------------------------------------------------
    #Pre-trained models
    cfg.pretrained_models = ml_collections.ConfigDict()
    cfg.pretrained_models.for_ast = ml_collections.ConfigDict()
    cfg.pretrained_models.for_ast.tiny224 = 'vit_deit_tiny_distilled_patch16_224'
    cfg.pretrained_models.for_ast.small224 = 'vit_deit_small_distilled_patch16_224'
    cfg.pretrained_models.for_ast.base224 = 'vit_deit_base_distilled_patch16_224'
    cfg.pretrained_models.for_ast.base384 = 'vit_deit_base_distilled_patch16_384'
    return cfg
