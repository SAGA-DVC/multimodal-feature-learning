import os
import torch
import torch.distributed as dist
from ml_collections import ConfigDict

def accuracy(output, target, topk=(1,)):
    '''
    Computes the accuracy over the k top predictions for the specified values of k
    '''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)  # pred: (B, maxk) 
        pred = pred.t()  # (maxk, B)


        # broadcast pred(maxk, B) to target[None](1, B, 1)  ==> correct(maxk, B, 1)
        correct = pred.eq(target[None])

        res = []
        for k in topk:
          correct_k = correct[:k].flatten().sum(dtype=torch.float32)
          res.append(correct_k * (100.0 / batch_size))
        return res

def print_master_only(is_master):
    '''
    This function disables printing when not in master process
    '''
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def torch_save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(distributed_cfg):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        distributed_cfg.rank = int(os.environ['RANK'])
        distributed_cfg.world_size = int(os.environ['WORLD_SIZE'])
        distributed_cfg.gpu = int(os.environ['LOCAL_RANK'])
        if distributed_cfg.world_size == 1:
            print("Not using distributed mode")
            distributed_cfg = None
            return
    else:
        print("Not using distributed mode") 
        distributed_cfg.world_size = 1
        return

    torch.cuda.set_device(distributed_cfg.gpu)
    distributed_cfg.backend = 'nccl'
    print(f'| distributed init (rank {distributed_cfg.rank}): {distributed_cfg.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=distributed_cfg.backend,
        init_method=distributed_cfg.dist_url,
        world_size=distributed_cfg.world_size,
        rank=distributed_cfg.rank
    )

    print_master_only(distributed_cfg.rank == 0)
