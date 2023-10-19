"""
Forked from SCAN (https://github.com/wvangansbeke/Unsupervised-Classification).
"""
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing


def create_config(config_file_env, config_file_exp, topk, checkpoint, best_model):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    cfg['num_neighbors'] = topk
    cfg['backbone'] = 'ViT-B/32'

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, 'pretext')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(pretext_dir)
    cfg['pretext_dir'] = pretext_dir
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    cfg['top{}_neighbors_train_path'.format(cfg['num_neighbors'])] = os.path.join(pretext_dir,
                                                    'top{}-train-neighbors.npy'.format(cfg['num_neighbors']))
    cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-val-neighbors.npy')

    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    if cfg['setup'] in ['clustering', 'selflabel']:
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        scan_dir = os.path.join(base_dir, 'clustering')
        mkdir_if_missing(base_dir)
        mkdir_if_missing(scan_dir)
        cfg['scan_dir'] = scan_dir
        cfg['scan_checkpoint'] = os.path.join(scan_dir, checkpoint)
        cfg['scan_model'] = scan_dir
        cfg['best_scan_model'] = os.path.join(scan_dir, best_model)  # using in selflabel
        cfg['scan_best_clustering_results'] = os.path.join(scan_dir, 'best_clustering_results.pth.tar')

    return cfg 
