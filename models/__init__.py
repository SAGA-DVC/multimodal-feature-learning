from pathlib import Path
import pickle

import torch
import numpy as np
import timm
from .deformable_dvc import DeformableDVC
from .dvc import DVC
from .matcher import build_matcher
from .criterion import SetCriterion
from config.config_dvc import load_config

# TODO - file.close()?
def build_model_and_criterion(args, dataset):

    # device = torch.device(args.device)

    model_official = None
    
    if args.model_official is not None:
        model_official = timm.create_model(args.model_official, pretrained=True)
        model_official.eval()

    matcher = build_matcher(args)
    
    embedding_matrix_file = Path(args.embedding_matrix_file_path)

    if embedding_matrix_file.exists():
        embedding_matrix = pickle.load(open(embedding_matrix_file, 'rb'))
    else:
        embedding_matrix = build_word_embedding_matrix(args.glove_file_path, dataset.vocab, args.pretrained_word_embed_dim)
        pickle.dump(embedding_matrix, open(embedding_matrix_file, 'wb'))

    if args.use_deformable_detr:

        model = DeformableDVC(model_name=args.model_name, 
                    num_frames_in=args.num_frames_in, 
                    img_size=args.img_size, 
                    spatial_patch_size=args.spatial_patch_size, 
                    temporal_patch_size=args.temporal_patch_size,
                    tokenization_method=args.tokenization_method, 
                    in_channels=args.in_channels, 
                    d_model=args.d_model, 
                    vocab_size=len(dataset.vocab), 
                    seq_len=dataset.max_caption_len_all, 
                    embedding_matrix=embedding_matrix, 
                    emb_weights_req_grad=False,
                    depth=args.depth, 
                    temporal_depth=args.temporal_depth,
                    num_heads=args.num_heads, 
                    mlp_ratio=args.mlp_ratio, 
                    qkv_bias=args.qkv_bias,
                    positional_embedding_dropout=args.positional_embedding_dropout,
                    attention_dropout=args.attention_dropout, 
                    projection_dropout=args.projection_dropout, 
                    dropout_1=args.dropout_1, 
                    dropout_2=args.dropout_2, 
                    pre_norm=args.pre_norm,
                    classification_head=args.classification_head, 
                    num_classes=args.num_classes,
                    num_queries=args.num_queries,
                    aux_loss=args.aux_loss,
                    return_preclassifier=args.return_preclassifier, 
                    return_prelogits=args.return_prelogits, 
                    weight_init=args.weight_init, 
                    weight_load=args.weight_load, 
                    model_official=model_official,
                    return_intermediate=args.return_intermediate,
                    matcher=matcher,
                    detr_args=args.detr
                )
    
    else :
        
        model = DVC(model_name=args.model_name, 
                    num_frames_in=args.num_frames_in, 
                    img_size=args.img_size, 
                    spatial_patch_size=args.spatial_patch_size, 
                    temporal_patch_size=args.temporal_patch_size,
                    tokenization_method=args.tokenization_method, 
                    in_channels=args.in_channels, 
                    d_model=args.d_model, 
                    vocab_size=len(dataset.vocab), 
                    seq_len=dataset.max_caption_len_all, 
                    embedding_matrix=embedding_matrix, 
                    emb_weights_req_grad=False,
                    depth=args.depth, 
                    temporal_depth=args.temporal_depth,
                    num_heads=args.num_heads, 
                    mlp_ratio=args.mlp_ratio, 
                    qkv_bias=args.qkv_bias,
                    positional_embedding_dropout=args.positional_embedding_dropout,
                    attention_dropout=args.attention_dropout, 
                    projection_dropout=args.projection_dropout, 
                    dropout_1=args.dropout_1, 
                    dropout_2=args.dropout_2, 
                    pre_norm=args.pre_norm,
                    classification_head=args.classification_head, 
                    num_classes=args.num_classes,
                    num_queries=args.num_queries,
                    aux_loss=args.aux_loss,
                    return_preclassifier=args.return_preclassifier, 
                    return_prelogits=args.return_prelogits, 
                    weight_init=args.weight_init, 
                    weight_load=args.weight_load, 
                    model_official=model_official,
                    return_intermediate=args.return_intermediate,
                    matcher=matcher
                )

    weight_dict = {'loss_ce': args.cls_loss_coef, 
                'loss_bbox': args.bbox_loss_coef,
                'loss_giou': args.giou_loss_coef,
                'loss_caption': args.captions_loss_coef}

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # losses = ['labels', 'segments', 'cardinality']
    losses = ['labels', 'segments', 'cardinality', 'captions']

    criterion = SetCriterion(num_classes=args.num_classes, matcher=matcher, weight_dict=weight_dict,
                            eos_coef=args.eos_coef, losses=losses, pad_idx=dataset.PAD_IDX, smoothing=args.smoothing,
                            focal_alpha=0.25, focal_gamma=2, )

    # # postprocessors = {'bbox': PostProcess(args)}

    return model, criterion



def build_word_embedding_matrix(glove_file_path, vocab, pretrained_word_embed_dim):

    """
    Initialises the embedding matrix with the ith row corresponding to the embedding of the ith word in the vocabulary. 
    Loads the words and their respective embeddings from the GloVe file and sets up a dictionary mapping of words and their corresponding embeddings 

    Parameters: 
        glove_file_path (string): path of file where embeddings are stored (eg: '<path>/glove.6B/glove.6B.100d.txt')
        vocabulary (torchtext.vocab.Vocab): mapping of all the words in the training dataset to indices and vice versa
        pretrained_word_embed_dim (int): dimension of word embeddings used in the model
    
    Returns: 
        embedding_matrix: of shape (vocab_size, pretrained_word_embed_dim) 
    """
    
    embedding_index = {}
    file = open(glove_file_path)
    for line in file:
        data = line.split(" ")
        word = data[0]
        embedding = np.asarray(data[1:], dtype='float32')
        embedding_index[word] = embedding
    file.close()

    embedding_matrix = np.random.normal(0, 0.1, (len(vocab), pretrained_word_embed_dim))
    for i, word in enumerate(vocab.get_itos()) :
        if word in embedding_index.keys():
            embedding_matrix[i] = embedding_index[word]
    return torch.Tensor(embedding_matrix)