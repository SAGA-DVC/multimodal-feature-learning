import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import wandb

from tsp.utils import is_main_process


def plot_grad_flow_line(named_parameters, epoch, batch_idx, prefix=None, output_dir="output", wandb_log=False):
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            try:
                ave_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())
            except AttributeError:
                print(p.grad)
                print("Grad None")
        elif "bias" not in n:
            print(n, "requires_grad false")

    plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.plot(max_grads, alpha=0.3, color="c")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.2)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(f'{output_dir}/{prefix}_epoch_{epoch}_batch_{batch_idx}_grad.png', bbox_inches="tight")
    if wandb_log and is_main_process():
        wandb.save(f'{output_dir}/{prefix}_epoch_{epoch}_batch_{batch_idx}_grad.png')
    

def plot_grad_flow_bar(named_parameters, epoch, batch_idx, prefix=None, output_dir="output", wandb_log=False):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            try:
                ave_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())
            except AttributeError:
                print(p.grad)
                print("Grad None")
        elif "bias" not in n:
            print(n, "requires_grad false")

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.2)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(f'{output_dir}/{prefix}_epoch_{epoch}_batch_{batch_idx}_grad.png', bbox_inches="tight")
    if wandb_log and is_main_process():
        wandb.save(f'{output_dir}/{prefix}_epoch_{epoch}_batch_{batch_idx}_grad.png')