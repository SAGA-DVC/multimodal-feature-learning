from pathlib import Path
import pickle

import torch
import numpy as np
import timm
from .sparse.cap_module import CapUnimodalSparseDVC
from .sparse.prop_module import PropUnimodalSparseDVC
from .regular.dvc import DVC
from .matcher import build_matcher
from .criterion import SetCriterion
from config.config_dvc import load_config

# TODO - file.close()?
def build_model_and_criterion(procedure, args, dataset, use_differentiable_mask=False):

    model_official = None
    
    matcher = build_matcher(args.matcher)
    
    if procedure == 'train_cap':
        embedding_matrix_file = Path(args.caption.embedding_matrix_file_path)

        if embedding_matrix_file.exists():
            embedding_matrix = pickle.load(open(embedding_matrix_file, 'rb'))
        else:
            embedding_matrix = build_word_embedding_matrix(args.caption.glove_file_path, dataset.vocab, args.caption.pretrained_word_embed_dim)
            pickle.dump(embedding_matrix, open(embedding_matrix_file, 'wb'))

        model = CapUnimodalSparseDVC(input_modalities=args.input_modalities,
                            num_queries=args.num_queries,
                            d_model=args.d_model, 
                            num_classes=args.num_classes,
                            aux_loss=args.aux_loss,
                            threshold=args.threshold,
                            max_eseq_length=args.max_eseq_length,
                            vocab=dataset.vocab, 
                            seq_len=dataset.max_caption_len_all, 
                            embedding_matrix=embedding_matrix,
                            sparse_detr_args=args.sparse_detr, 
                            caption_args=args.caption,
                        )

        weight_dict = {'loss_caption': args.caption_loss_coef,
                        'loss_mask_prediction': args.mask_prediction_coef
                        }


        if args.aux_loss:
            caption_aux_weight_dict = {}
            for i in range(args.caption.depth - 1):
                caption_aux_weight_dict.update({'loss_caption' + f'_{i}': weight_dict['loss_caption']})
        
        if args.sparse_detr.use_enc_aux_loss:
            enc_aux_weight_dict = {}
            for i in range(args.sparse_detr.enc_layers - 1):
                enc_aux_weight_dict.update({k + f'_enc_{i}': v for k, v in weight_dict.items()})

        if args.aux_loss:
            weight_dict.update(caption_aux_weight_dict)
        
        if args.sparse_detr.use_enc_aux_loss:
            weight_dict.update(enc_aux_weight_dict)
    

    elif procedure == 'train_prop':
        model = PropUnimodalSparseDVC(input_modalities=args.input_modalities,
                            num_queries=args.num_queries,
                            d_model=args.d_model, 
                            num_classes=args.num_classes,
                            aux_loss=args.aux_loss,
                            threshold=args.threshold,
                            max_eseq_length=args.max_eseq_length,
                            sparse_detr_args=args.sparse_detr, 
                        )

        weight_dict = {'loss_ce': args.cls_loss_coef,
                'loss_counter': args.counter_loss_coef, 
                'loss_bbox': args.bbox_loss_coef,
                'loss_giou': args.giou_loss_coef,
                'loss_self_iou': args.self_iou_loss_coef,
                'loss_context': args.context_loss_coef,
                'loss_mask_prediction': args.mask_prediction_coef,
                'loss_corr': args.corr_coef,
                }

        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.sparse_detr.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != 'loss_caption'})
        
        if args.sparse_detr.use_enc_aux_loss:
            enc_aux_weight_dict = {}
            for i in range(args.sparse_detr.enc_layers - 1):
                enc_aux_weight_dict.update({k + f'_enc_{i}': v for k, v in weight_dict.items()})

        if args.aux_loss:
            weight_dict.update(aux_weight_dict)
        
        if args.sparse_detr.use_enc_aux_loss:
            weight_dict.update(enc_aux_weight_dict)
    


    criterion = SetCriterion(procedure=procedure, is_multimodal=len(args.input_modalities) == 2, 
                            num_classes=args.num_classes, matcher=matcher, weight_dict=weight_dict,
                            eos_coef=args.eos_coef, losses=args.losses, pad_idx=dataset.PAD_IDX, smoothing=args.smoothing,
                            focal_alpha=0.25, focal_gamma=2, lloss_gau_mask=args.lloss_gau_mask, lloss_beta=args.lloss_gau_mask)

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