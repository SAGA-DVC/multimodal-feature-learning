from pprint import pprint
import time
import os

import torch
import numpy as np
import wandb

from tsp import utils
from tsp.tsp_model import TSPModel, add_combiner, concat_combiner
from utils import plots


def epoch_loop(model: TSPModel, criterion, optimizer, dataloader, device, epoch, print_freq, label_columns, loss_alphas, wandb_log, output_dir):
    model.train()

    metric_logger = utils.MetricLogger(delimiter=' ')
    # for g in optimizer.param_groups:
    #     metric_logger.add_meter(
    #         f'{g["name"]}-lr', utils.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter(
        'clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    header = f'Train Epoch {epoch}:'

    for (batch_idx, batch) in enumerate(metric_logger.log_every(dataloader, print_freq, header, device=device)):
        start_time = time.time()
        clip = {
            'video': batch['video'].to(device),
            'audio': batch['audio'].to(device)
        }

        # Global video feature (video + audio features)
        gvf = batch['gvf'].to(device) if 'gvf' in batch else None

        # targets has the class index directly, not one-hot-encoded
        # See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        targets = [batch[x].to(device)
                   for x in label_columns]  # [(B, 1), (B, 1)]

        # Forward pass through TSPModel
        outputs = model(clip, gvf=gvf)  # [(B, 2), (B, c)]

        # compute losses for each label column
        head_losses, loss = [], 0
        for output, target, alpha in zip(outputs, targets, loss_alphas):
            head_loss = criterion(output, target)
            head_losses.append(head_loss)
            loss += alpha * head_loss

        # backprop
        optimizer.zero_grad()
        loss.backward()

        if utils.is_main_process() and batch_idx % print_freq == 0:
            plots.plot_grad_flow_line(model.module..named_parameters(), epoch=epoch, batch_idx=batch_idx, prefix='fc', output_dir=output_dir)
            plots.plot_grad_flow_line(model.module.backbones[0].named_parameters(), epoch=epoch, batch_idx=batch_idx, prefix='vivit', output_dir=output_dir)
            plots.plot_grad_flow_line(model.module.backbones[1].named_parameters(), epoch=epoch, batch_idx=batch_idx, prefix='ast', output_dir=output_dir)

        optimizer.step()

        compute_and_log_metrics(
            metric_logger=metric_logger,
            phase="train",
            loss=loss,
            outputs=outputs,
            targets=targets,
            head_losses=head_losses,
            label_columns=label_columns,
            optimizer=optimizer,
            epoch=epoch,
            batch_idx=batch_idx,
            wandb_log=wandb_log
        )

        # for g in optimizer.param_groups:
        #     metric_logger.meters[f'{g["name"]}-lr'].update(g['lr'])
        metric_logger.meters['clips/s'].update(
            clip['video'].shape[0] / (time.time() - start_time))

    # lr_scheduler.step()


def evaluate(model: TSPModel, criterion, dataloader, device, epoch, print_freq, label_columns, loss_alphas, output_dir, wandb_log):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter=' ')
    header = f'Valid Epoch {epoch}:'

    with torch.no_grad():
        for (batch_idx, batch) in enumerate(metric_logger.log_every(dataloader, print_freq, header, device=device)):
            clip = {
                'video': batch['video'].to(device, non_blocking=True),
                'audio': batch['audio'].to(device, non_blocking=True)
            }

            # Global video feature (video + audio features)
            gvf = batch['gvf'].to(
                device, non_blocking=True) if 'gvf' in batch else None

            # Targets
            targets = [batch[x].to(device, non_blocking=True)
                       for x in label_columns]

            # Forward pass through model
            outputs = model(clip, gvf=gvf)

            # compute losses for each label column
            head_losses, loss = [], 0
            for output, target, alpha in zip(outputs, targets, loss_alphas):
                head_loss = criterion(output, target)
                head_losses.append(head_loss)
                loss += alpha * head_loss

            compute_and_log_metrics(
                metric_logger=metric_logger,
                phase="val",
                loss=loss,
                outputs=outputs,
                targets=targets,
                head_losses=head_losses,
                label_columns=label_columns,
                optimizer=None,
                epoch=epoch,
                batch_idx=batch_idx,
                wandb_log=wandb_log
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    results = write_metrics_results_to_file(
        metric_logger, epoch, label_columns, output_dir)
    print(results)


def compute_and_log_metrics(metric_logger, phase, loss, outputs, targets, head_losses, label_columns, optimizer, epoch, batch_idx, wandb_log=False):
    log = {
        "epoch": epoch,
        "batch": batch_idx,
        "loss": loss.item()
    }

    for output, target, head_loss, label_column in zip(outputs, targets, head_losses, label_columns):
        mask = target != -1   # target == -1 => sample has no output for this head
        output, target = output[mask], target[mask]  # filter out -1
        head_num_samples = output.shape[0]

        if head_num_samples:
            head_acc = utils.accuracy(output, target, topk=(1,))[0]
            log[f"accuracy-{label_column}"] = head_acc.item()
            log[f"num_samples-{label_column}"] = head_num_samples
            metric_logger.meters[f'acc-{label_column}'].update(
                head_acc.item(), n=head_num_samples)

        log[f"loss-{label_column}"] = head_loss.item()
        metric_logger.meters[f'loss-{label_column}'].update(head_loss.item())

    if wandb_log and utils.is_main_process():
        log_dict = {
            f"{phase}/{key}": value
            for key, value in log.items()
        }
        # if optimizer:
        #     for g in optimizer.param_groups:
        #         log_dict[f"{phase}/{g['name']}-lr"] = getattr(metric_logger, f"{g['name']}-lr").global_avg

        wandb.log(log_dict)
    pprint(log)
    print()

    metric_logger.update(loss=loss.item())


def write_metrics_results_to_file(metric_logger, epoch, label_columns, output_dir):
    results = f'** Valid Epoch {epoch}: '
    for label_column in label_columns:
        results += f' <{label_column}> Accuracy {metric_logger.meters[f"acc-{label_column}"].global_avg:.3f}'
        results += f' Loss {metric_logger.meters[f"loss-{label_column}"].global_avg:.3f};'

    results += f' Total Loss {metric_logger.meters["loss"].global_avg:.3f}'
    avg_acc = np.average(
        [metric_logger.meters[f'acc-{label_column}'].global_avg for label_column in label_columns])
    results += f' Avg Accuracy {avg_acc:.3f}'

    results = f'{results}\n'
    utils.write_to_file_on_master(file=os.path.join(output_dir, 'results.txt'),
                                  mode='a',
                                  content=results)

    return results
