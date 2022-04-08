import os
import datetime
import time
from collections import defaultdict, deque
import math

import torch
import torch.distributed as dist

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
    '''torch.save if master process calls it'''
    if is_main_process():
        torch.save(*args, **kwargs)


def write_to_file_on_master(file, mode, content):
    '''Write `content` to `file` if called by master process, on master node'''
    if is_main_process():
        with open(file, mode) as f:
            f.write(content)


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

class SmoothedValue(object):
    '''
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    '''

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{global_avg:.2f}'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        '''
        Warning: does not synchronize the deque!
        '''
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        try:
            return self.total / self.count
        except ZeroDivisionError:
            # Happens when all outputs get masked and a SmoothedValue does not
            # get updated at all. So count stays 0
            # This will probably lead to NaN everywhere
            # TODO: Check this
            return math.nan

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f'"{type(self).__name__}" object has no attribute "{attr}"')

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                f'{name}: {str(meter)}'
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, device=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.2f}')
        data_time = SmoothedValue(fmt='{avg:.2f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available() and torch.device('cpu') != device:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max_mem: {memory:.2f}GB'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        GB = 1024.0 ** 3
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available() and torch.device('cpu') != device:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated(device) / GB), flush=True)
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)), flush=True)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} total time: {total_time_str}\n')