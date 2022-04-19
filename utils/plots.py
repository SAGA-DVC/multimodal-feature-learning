import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wandb

from utils.misc import is_main_process


def plot_grad_flow_line_plot(named_parameters, epoch, batch_idx, output_dir='output', wandb_log=False):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        # if p.grad == None:
        #     print(n, p.requires_grad)
        #     print("Grad None!!")
        if(p.requires_grad) and ("bias" not in n) and (p.grad != None):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())

    plt.figure(figsize=(20, 20), dpi=80)

    plt.plot(ave_grads, alpha=0.3, color="b")

    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=5)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.01) # zoom in on the lower gradient regions

    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title(f"Gradient flow for epoch {epoch}, batch {batch_idx}")
    # plt.grid(True)

    plt.savefig(os.path.join(output_dir, f"grads/E{epoch}_B{batch_idx}_line.png"), bbox_inches='tight')
    
    if wandb_log and is_main_process():
        wandb.save(os.path.join(output_dir, f"grads/E{epoch}_B{batch_idx}_line.png"))



def plot_grad_flow_bar_plot(named_parameters, epoch, batch_idx, output_dir='output', wandb_log=False):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        # if p.grad == None:
        #     print(n, p.requires_grad)
        #     print("Grad None!!")
        if(p.requires_grad) and ("bias" not in n) and (p.grad != None):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    
    plt.figure(figsize=(20, 20), dpi=80)

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")

    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=5)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.01) # zoom in on the lower gradient regions

    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title(f"Gradient flow for epoch {epoch}, batch {batch_idx}")
    # plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.savefig(os.path.join(output_dir, f"grads/E{epoch}_B{batch_idx}_bar.png"), bbox_inches='tight')

    if wandb_log and is_main_process():
        wandb.save(os.path.join(output_dir, f"grads/E{epoch}_B{batch_idx}_bar.png"))