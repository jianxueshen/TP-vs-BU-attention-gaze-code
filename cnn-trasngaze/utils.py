import numpy as np
import os
import random
import torch
import argparse
from os.path import join


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def fixations2seq(fixations, max_len):
    processed_fixs = []
    for fix in fixations:
        processed_fixs.append({
            'tgt_seq_y': torch.tensor(np.array(fix.get('Y', []))[:max_len]),
            'tgt_seq_x': torch.tensor(np.array(fix.get('X', []))[:max_len]),
            'tgt_seq_t': torch.tensor(np.array(fix.get('T', []))[:max_len]),
            'img_name': fix.get('name', 'none'),  # 提供一个默认值
            'class': fix.get('condition', 'none')  # 提供一个默认值
        })
    return processed_fixs
    
def get_args_parser_test():
    parser = argparse.ArgumentParser('Gaze Transformer Tester', add_help=False)
    parser.add_argument('--eye_data_path', default= './dataset', type=str, help="Dataset Directory")
    parser.add_argument('--img_ftrs_dir', default= './dataset/image_features', type=str, help="Directory of precomputed ResNet features")
    parser.add_argument('--im_h', default=20, type=int, help="Height of feature map input to encoder")
    parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")
    parser.add_argument('--patch_size', default=16, type=int, help="Patch size of feature map input with respect to fixation image dimensions (320X512)")
    parser.add_argument('--max_len', default=7, type=int, help="Maximum length of scanpath")
    parser.add_argument('--num_encoder', default=6, type=int, help="Number of transformer encoder layers")
    parser.add_argument('--num_decoder', default=6, type=int, help="Number of transformer decoder layers")
    parser.add_argument('--hidden_dim', default=512, type=int, help="Hidden dimensionality of transformer layers")
    parser.add_argument('--nhead', default=8, type=int, help="Number of heads for transformer attention layers")
    parser.add_argument('--img_hidden_dim', default=2048, type=int, help="Channel size of initial ResNet feature map")
    parser.add_argument('--trained_model', default='/home/oct/Gazeformer/saved_models/trained/train_22-02-2025-21-14-57/gazeformer_6E_6D_16B_512d_epoch1_valacc96.91_valloss0.118.pth', type=str, help="Trained model checkpoint to run for inference")
    parser.add_argument('--seed', default=42, type=int, help="Seed")
    parser.add_argument('--cuda', default=0, type=int, help="CUDA core to load models and data")
    parser.add_argument('--zerogaze', default=False, action='store_true', help="ZeroGaze setting flag")
    parser.add_argument('--num_samples', default=10, type=int, help="Number of scanpaths sampled per test case")
    return parser
    
def get_args_parser_train():
    parser = argparse.ArgumentParser('Gaze Transformer Trainer', add_help=False)
    parser.add_argument('--head_lr', default=1e-6, type=float, help="Learning rate for SlowOpt")
    parser.add_argument('--tail_lr', default=1e-6, type=float, help="Learning rate for FastOpt")
    parser.add_argument('--dataset_dir', default= './dataset', type=str, help="Dataset Directory")
    parser.add_argument('--train_file', default= 'coco_search18_TP_and_FV_fixations_train_512x320.json', type=str, help="Training fixation file")
    parser.add_argument('--valid_file', default= 'coco_search18_TP_and_FV_fixations_val_512x320.json', type=str, help="Validation fixation file")
    parser.add_argument('--img_ftrs_dir', default= './dataset/image_features', type=str, help="Directory of precomputed ResNet features")
    parser.add_argument('--im_h', default=20, type=int, help="Height of feature map input to encoder")
    parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")
    parser.add_argument('--patch_size', default=16, type=int, help="Patch size of feature map input with respect to fixation image dimensions (320X512)")
    parser.add_argument('--seed', default=42, type=int, help="seed")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch Size")
    parser.add_argument('--epochs', default=20, type=int, help="Maximum number of epochs to train")
    parser.add_argument('--max_len', default=7, type=int, help="Maximum length of scanpath")
    parser.add_argument('--num_encoder', default=3, type=int, help="Number of transformer encoder layers")
    parser.add_argument('--num_decoder', default=3, type=int, help="Number of transformer decoder layers")
    parser.add_argument('--hidden_dim', default=512, type=int, help="Hidden dimensionality of transformer layers")
    parser.add_argument('--nhead', default=8, type=int, help="Number of heads for transformer attention layers")
    parser.add_argument('--img_hidden_dim', default=2048, type=int, help="Channel size of initial ResNet feature map")
    parser.add_argument('--encoder_dropout', default=0.4, type=float, help="Encoder dropout rate")
    parser.add_argument('--decoder_dropout', default=0.4, type=float, help="Decoder and fusion step dropout rate")
    parser.add_argument('--cls_dropout', default=0.4, type=float, help="Final scanpath prediction dropout rate")
    parser.add_argument('--retraining', default=False, action='store_true', help="Retraining from a checkpoint")
    parser.add_argument('--last_checkpoint', default='./saved_models/gazeformer_6E_6D_32_512d_70.pkg', type=str, help="Checkpoint for retraining")
    parser.add_argument('--model_root', default='./saved_models/trained', type=str, help="Checkpoint directory")
    parser.add_argument('--cuda', default=0, type=int, help="CUDA core to load models and data")
    parser.add_argument('--num_workers', default=6, type=int, help="Number of workers for data loader")
    return parser

    
def save_model_train(args, model, SlowOpt, FastOpt, epoch, check_point_path):
    state = {
        "epoch": epoch,
        "args": vars(args),
        "model":
        model.module.state_dict()
        if hasattr(model, "module") else model.state_dict(),
        "optim_slow":
        SlowOpt.state_dict(),
        "optim_fast":
        FastOpt.state_dict(),    
    }
    torch.save(state, check_point_path)
    
    

