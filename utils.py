from config import *

import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim as optim


def setup_train(args):
    set_up_gpu(args)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)

    pp.pprint({k: v for k, v in vars(args).items() if v is not None}, width=1)
    return export_root


def create_experiment_export_folder(args):
    experiment_dir, experiment_description = args.experiment_dir, args.experiment_description
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
    os.mkdir(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path


def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def load_weights(model, path):
    pass


def save_test_result(export_root, result):
    filepath = Path(export_root).joinpath('test_result.txt')
    with filepath.open('w') as f:
        json.dump(result, f, indent=2)


def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def set_up_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    args.num_gpu = len(args.device_idx.split(","))


def load_pretrained_weights(model, path):
    chk_dict = torch.load(os.path.abspath(path))
    model_state_dict = chk_dict[STATE_DICT_KEY] if STATE_DICT_KEY in chk_dict else chk_dict['state_dict']
    model.load_state_dict(model_state_dict)


def setup_to_resume(args, model, optimizer):
    chk_dict = torch.load(os.path.join(os.path.abspath(args.resume_training), 'models/checkpoint-recent.pth'))
    model.load_state_dict(chk_dict[STATE_DICT_KEY])
    optimizer.load_state_dict(chk_dict[OPTIMIZER_STATE_DICT_KEY])


def create_optimizer(model, args):
    if args.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)


# 我们将把这个函数添加到 utils.py
def test_with(model, test_loader, metric_ks, device):
    """
    Evaluates the model on the test set.
    """
    model.eval() # Set model to evaluation mode
    
    # meters for tracking average metrics
    val_meters = AverageMeterSet()

    with torch.no_grad():
        tqdm_dataloader = tqdm(test_loader, desc="Evaluating")
        for batch_idx, batch in enumerate(tqdm_dataloader):
            # Move batch data to the correct device
            # This part needs to accurately reflect the structure of `batch`
            # from your dataloader. From previous errors and typical BERT4Rec setup:
            # batch[0]: input sequence (e.g., historical items for BERT)
            # batch[1]: candidates for evaluation (positive item + negative items)
            # batch[2]: binary labels for candidates (1 for positive, 0 for negative)

            seqs = batch[0].to(device)
            candidates = batch[1].to(device)
            labels = batch[2].to(device)

            # Get model scores for candidates
            # model(seqs) typically outputs (batch_size, seq_len, vocab_size)
            # For next-item prediction, we usually take the prediction for the last item in sequence.
            # Then gather scores for only the candidate items.
            scores = model(seqs)[:, -1, :] # (batch_size, vocab_size) - prediction for the next item
            
            # scores_for_candidates shape: (batch_size, num_candidates)
            scores_for_candidates = scores.gather(dim=1, index=candidates)

            # Calculate metrics for this batch
            # This is where the core ranking metric calculation happens.
            # The project MUST have this logic somewhere in the trainer or an evaluation file.
            # I'll create a placeholder function `_calculate_ranking_metrics_batch` below
            # to show what it would look like. You would ideally find this function in the project.
            batch_metrics = _calculate_ranking_metrics_batch(scores_for_candidates, labels, metric_ks)

            # Update meters
            for k_val in metric_ks:
                val_meters.update(f'N@{k_val}', batch_metrics[f'N@{k_val}'])
                val_meters.update(f'R@{k_val}', batch_metrics[f'R@{k_val}'])
            
            # Optional: update tqdm description
            # tqdm_dataloader.set_postfix(val_meters.averages(format_string='{:.3f}')) # Too verbose for tqdm

    final_metrics = val_meters.averages()
    return final_metrics


# Placeholder/Example for _calculate_ranking_metrics_batch
# This function is what you'd ideally find/adapt from the project's trainer or evaluation module.
# It calculates NDCG and Recall for a batch of predictions.
def _calculate_ranking_metrics_batch(scores_for_candidates, labels, metric_ks):
    """
    Calculates ranking metrics (NDCG@K, Recall@K) for a batch.
    Assumes labels are binary (1 for positive, 0 for negative) and
    positive item is typically at candidates[:, 0] (as setup by leave-one-out and negative sampling).
    """
    batch_size = scores_for_candidates.size(0)
    metrics = defaultdict(float)

    # For each sample in the batch
    for i in range(batch_size):
        user_scores = scores_for_candidates[i]
        user_labels = labels[i] # This should be [1, 0, 0, ..., 0] for one positive and many negatives

        # Find the rank of the positive item (which is at index 0 in `user_candidates` conceptually)
        # Sort scores in descending order to get ranks
        # The position of the 1st element (positive item) in the sorted list is its rank
        # We need to find the index of the positive item (label=1) in the original `candidates` list,
        # then find its rank based on `user_scores`.

        # A common simplified setup: the positive item is always the first one in the candidates list
        # (index 0), and all other candidates are negatives.
        # This is very typical for leave-one-out and random negative sampling for evaluation.

        # Find the rank of the item with label 1 (the positive item)
        # We get the indices that would sort the scores in descending order
        _, sorted_indices = torch.sort(user_scores, descending=True)
        
        # The true positive item's original index among candidates is the one where label is 1
        true_positive_original_idx = (user_labels == 1).nonzero(as_tuple=True)[0].item()

        # Find the rank of the true positive item
        rank = (sorted_indices == true_positive_original_idx).nonzero(as_tuple=True)[0].item() + 1 # 1-based rank

        for k in metric_ks:
            # Recall@K
            if rank <= k:
                metrics[f'R@{k}'] += 1.0 # Hit
                # NDCG@K
                metrics[f'N@{k}'] += 1.0 / math.log2(rank + 1)
            
    # Average across the batch
    for key in metrics:
        metrics[key] /= batch_size # Divide by batch_size to get average for this batch

    return metrics


# Add these imports at the top of utils.py if they are not already there
import math
from tqdm import tqdm
from collections import defaultdict


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
