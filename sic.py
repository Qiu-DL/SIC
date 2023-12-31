"""
Forked from SCAN (https://github.com/wvangansbeke/Unsupervised-Classification).
"""

import argparse
import os
import torch
import sys
import numpy as np

from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_model,construct_semantic_space
from utils.evaluate_utils import get_predictions, hungarian_evaluate, kmeans, sic_evaluate
from utils.train_utils import sic_train
from utils.utils import Logger, get_features_eval, get_knn_indices, mkdir_if_missing
from datetime import datetime
import time
import copy

FLAGS = argparse.ArgumentParser(description='SIC Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--gpu', type=str, default='0,1')

# The best hyper_parameters for cifar10 dataset, and others can be found on table2 of SIC.
FLAGS.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
FLAGS.add_argument('--gamma_u', type=float, default=0.05, help='gamma_u is a uniqueness score for constructing semantic space')
FLAGS.add_argument('--gamma_r', type=int, default=500, help='gamma_r is a number for nearest nouns to each image center when constructing semantic space')
FLAGS.add_argument('--xi_c', type=int, default=0.9, help='Top xi_c branch images for each class when computing image centers')
FLAGS.add_argument('--xi_a', type=int, default=30, help='xi_a is a number for nearest texts to each image center in adjusted center-based mapping')
FLAGS.add_argument('--topk', type=int, default=20, help='The number of nearest neighbors in image consistency learning')
FLAGS.add_argument('--entropy_weight', type=float, default=5.0, help='The hyper-para(lambda) for the entropy_loss')
FLAGS.add_argument('--ce_weight', type=float, default=0.1, help='The hyper-para(beta) for the ce_loss')

FLAGS.add_argument('--center_init', action='store_true', help='Use the k-means centers to initialize the fc layer')
FLAGS.add_argument('--version', type=str, default='log', help='Record the version')
FLAGS.add_argument('--checkpoint', type=str, default='checkpoint.pth.tar')

args = FLAGS.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    p = create_config(args.config_env, args.config_exp, args.topk, args.checkpoint, '')
    p['optimizer_image']['optimizer_kwargs']['lr'] = args.lr

    mkdir_if_missing(os.path.join('/log', p['train_db_name']))

    # Log
    # logfile_name = '/path/to/training.log'
    logfile_name = os.path.join('/log', p['train_db_name'], 'training.log')
    sys.stdout = Logger(filename=logfile_name, stream=sys.stdout)

    # Model
    print(colored('Get model', 'blue'))
    model, preprocess = get_model(p)
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    # CUDNN
    torch.backends.cudnn.benchmark = True


    # Compute knn indices for image consistency learning
    if not os.path.exists(os.path.join('/log', p['train_db_name'], 'train_knn_indices.npy')):
        train_dataset = get_train_dataset(p, preprocess, split='train')
        val_dataset = get_val_dataset(p, preprocess)
        base_train_dataloader = get_val_dataloader(p, train_dataset)
        base_val_dataloader = get_val_dataloader(p, val_dataset)
        train_knn_indices, train_accuracy, val_knn_indices, val_accuracy = get_knn_indices(model, base_train_dataloader, base_val_dataloader,
                                                                                   args.topk)
        np.save(os.path.join('/log', p['train_db_name'], 'train_knn_indices.npy'), train_knn_indices)
        np.save(os.path.join('/log', p['train_db_name'], 'val_knn_indices.npy'), val_knn_indices)
        print('train_dataset knn accuracy: {:.2f}'.format(train_accuracy))
        print('val_dataset knn accuracy: {:.2f}'.format(val_accuracy))


    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_dataset = get_train_dataset(p, preprocess, split='train', to_neighbors_dataset = True)
    val_dataset = get_val_dataset(p, preprocess, to_neighbors_dataset = True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))

    # Get image centers
    if not os.path.exists(os.path.join('/log', p['train_db_name'], 'image_center.pth.tar')):
        dataloader = get_val_dataloader(p, train_dataset)
        features, targets = get_features_eval(dataloader, model)
        image_centers = kmeans(features, targets)
        image_centers = torch.from_numpy(image_centers).cuda()
        torch.save([features,targets,image_centers],os.path.join('/log', p['train_db_name'], 'image_center.pth.tar'))
    else:
        features, targets, image_centers = torch.load(os.path.join('/log', p['train_db_name'], 'image_center.pth.tar'))

    # Image optimizer
    if p['update_cluster_head_only']:
        for name, param in model.named_parameters():
            if 'cluster_head_i' in name :            # context vectors
                param.requires_grad = True
            else:
                param.requires_grad = False
        head_i_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        image_optimizer = torch.optim.Adam(head_i_params, **p['optimizer_image']['optimizer_kwargs'])
        print('image_optimizer:', image_optimizer)

    # Construct semantic space
    text_dataloader = construct_semantic_space(p, image_centers, model, args)


    # Loss function
    print(colored('Get loss', 'blue'))
    from losses.losses import SICLoss
    criterion = SICLoss(args, p['num_classes'])
    criterion = criterion.cuda()
    print(criterion)

    # Checkpoint
    if False and os.path.exists(p['scan_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['scan_checkpoint']), 'blue'))
        checkpoint = torch.load(p['scan_checkpoint'], map_location='cpu')
        model.module.load_state_dict(checkpoint['model'])
        image_optimizer.load_state_dict(checkpoint['image_optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss_head = checkpoint['best_loss_head']
        best_acc = 0
        best_clustering_stats = None

    else:
        print(colored('No checkpoint file at {}'.format(p['scan_checkpoint']), 'blue'))
        start_epoch = 0
        best_acc = 0
        best_clustering_stats = None
        best_loss_head = 0


    # Cluster_head_i initialized by image centers
    if args.center_init:
        print("cluster_head_i param init")
        alpha = 5*10e-3
        for i in range(p['num_heads']):
            model.module.cluster_head_i[i].weight.data = 2*alpha*image_centers.cuda()
            model.module.cluster_head_i[i].bias.data = -alpha*(torch.norm(image_centers, dim=1)**2).cuda()

    # Computer image centers according to the confident samples
    from utils.compute_center import ComputeCenter
    cpt_center = ComputeCenter(num_cluster=p['num_classes'])

    early_stop_count = 0

    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Train
        print('Train ...')
        sic_train(p, args, train_dataloader, text_dataloader, [image_centers, features, targets],
                                                model, image_optimizer, criterion, cpt_center, epoch+1, p['update_cluster_head_only'])

        # Evaluate
        print('Make prediction on validation set ...')
        predictions = get_predictions(p, val_dataloader, model)
        scan_stats = sic_evaluate(predictions)
        lowest_loss_head = scan_stats['lowest_loss_head']
        print('Evaluate with hungarian matching algorithm ...')
        clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
        print('mlp_predict', clustering_stats)

        # Save the best clustering stats
        if clustering_stats['ACC'] > best_acc:
            print("best epoch",epoch)
            early_stop_count = 0
            best_acc = clustering_stats['ACC']
            best_clustering_stats = clustering_stats

        else:
            early_stop_count += 1
            if early_stop_count >= 5:
                break

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'image_optimizer': image_optimizer.state_dict(), 'model': model.module.state_dict(),
                    'epoch': epoch + 1, 'best_loss_head': best_loss_head},
                     p['scan_checkpoint'])


    print('best_clustering_stats:')
    print(best_clustering_stats)


if __name__ == "__main__":
    main()

