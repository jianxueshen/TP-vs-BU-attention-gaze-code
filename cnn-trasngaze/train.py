from typing import Optional, List
from timeit import default_timer as timer
import argparse
from datetime import datetime
import os
from os.path import join
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from models import Transformer
from gazeformer import gazeformer
from utils import seed_everything, fixations2seq, get_args_parser_train, save_model_train
from dataset import fixation_dataset, COCOSearch18Collator
from sklearn.metrics import precision_score, recall_score, f1_score

torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def calculate_metrics(out_token, token_gt):
    _, predicted = torch.max(out_token, 1)
    _, target = torch.max(token_gt, 1)
    
    correct = (predicted == target).sum().item()
    total = target.size(0)
    accuracy = 100. * correct / total
    
    predicted_np = predicted.cpu().numpy()
    target_np = target.cpu().numpy()
    
    precision = precision_score(target_np, predicted_np, average='weighted')
    recall = recall_score(target_np, predicted_np, average='weighted')
    f1 = f1_score(target_np, predicted_np, average='weighted')
    
    return accuracy, precision, recall, f1

def train(epoch, args, model, SlowOpt, FastOpt, loss_fn, train_dataloader, model_dir, 
          model_name, device = 'cuda:0'):
    model.train()
    losses = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_batches = 0

    with tqdm(train_dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            batch_imgs, batch_tgt, batch_tgt_padding_mask, batch_firstfix, batch_class = batch
            batch_imgs = batch_imgs.to(device)
            batch_tgt = batch_tgt.to(device)
            batch_class = batch_class.to(device)

            out_token = model(src=batch_imgs, tgt=batch_tgt)

            SlowOpt.zero_grad()
            FastOpt.zero_grad()

            token_gt = batch_class.float()
            loss = loss_fn(out_token, token_gt)
            loss.backward()
            losses += loss.item()
            accuracy, precision, recall, f1 = calculate_metrics(out_token, token_gt)
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_batches += 1

            SlowOpt.step()
            FastOpt.step()
            
            tepoch.set_postfix(loss=losses/total_batches, accuracy=total_accuracy/total_batches, precision=total_precision/total_batches, recall=total_recall/total_batches, f1=total_f1/total_batches)
    return losses / total_batches, total_accuracy / total_batches, total_precision / total_batches, total_recall / total_batches, total_f1 / total_batches

def evaluate(model, loss_fn, valid_dataloader, device='cuda:0'):
    model.eval()
    losses = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_batches = 0 

    with tqdm(valid_dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            batch_imgs, batch_tgt, batch_tgt_padding_mask, batch_firstfix, batch_class = batch
            batch_imgs = batch_imgs.to(device)
            batch_tgt = batch_tgt.to(device)
            batch_class = batch_class.to(device)

            with torch.no_grad():
                out_token = model(src=batch_imgs, tgt=batch_tgt)

            token_gt = batch_class.float()
            loss = loss_fn(out_token, token_gt)
            
            losses += loss.item()

            accuracy, precision, recall, f1 = calculate_metrics(out_token, token_gt)
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_batches += 1

            tepoch.set_postfix(loss=losses/total_batches, accuracy=total_accuracy/total_batches, precision=total_precision/total_batches, recall=total_recall/total_batches, f1=total_f1/total_batches)
    return losses / total_batches, total_accuracy / total_batches, total_precision / total_batches, total_recall / total_batches, total_f1 / total_batches

def main(args):
    #seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.cuda))
    device_id = args.cuda
    retraining = args.retraining
    last_checkpoint = args.last_checkpoint
    if retraining:
        model_dir = '/'.join(args.last_checkpoint.split('/')[:-1])
        args = argparse.Namespace(**json.load(open(join(model_dir, 'config.json'))))
        logfile = 'logs/output_' + last_checkpoint.split('/')[-2].split('_')[-1]+'.txt'
        args.cuda = device_id
    else:
        timenow = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") 
        logfile = 'logs/output_' + timenow + '.txt'
        model_dir = join(args.model_root, 'train_' + timenow)
        os.makedirs(model_dir, exist_ok=True)
        
        os.makedirs('logs', exist_ok=True)
        open(logfile, 'w').close()
        with open(logfile, "a") as myfile:
            myfile.write(str(vars(args)) + '\n\n')
            myfile.close()
    print(str(vars(args)) + '\n\n')
    with open(join(model_dir, 'config.json'), "w") as outfile:
        json.dump(vars(args), outfile)
        outfile.close()


    model_name = 'gazeformer_'+str(args.num_encoder)+'E_'+str(args.num_decoder)+'D_'+str(args.batch_size)+'B_'+str(args.hidden_dim)+'d'
    dataset_root = args.dataset_dir
    train_file = args.train_file
    valid_file = args.valid_file
    with open(join(dataset_root,
                   train_file)) as json_file:
        fixations_train = json.load(json_file)
    with open(join(dataset_root,
                   valid_file)) as json_file:
        fixations_valid = json.load(json_file)

        
    seq_train = fixations2seq(fixations =fixations_train, max_len = args.max_len)
            
    seq_valid = fixations2seq(fixations = fixations_valid, max_len = args.max_len)

    train_dataset = fixation_dataset(seq_train, img_ftrs_dir = args.img_ftrs_dir)
    valid_dataset = fixation_dataset(seq_valid, img_ftrs_dir = args.img_ftrs_dir)
    #target embeddings
    embedding_dict = np.load(open(join(dataset_root, 'embeddings.npy'), mode='rb'), allow_pickle = True).item()

    collate_fn = COCOSearch18Collator(embedding_dict, args.max_len, args.im_h, args.im_w, args.patch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=collate_fn)

    #model
    transformer = Transformer(num_encoder_layers=args.num_encoder, nhead = args.nhead, 
                              d_model = args.hidden_dim, num_decoder_layers=args.num_decoder, 
                              encoder_dropout = args.encoder_dropout, decoder_dropout = args.decoder_dropout, 
                              dim_feedforward = args.hidden_dim, img_hidden_dim = args.img_hidden_dim, 
                              device = device).to(device)

    model = gazeformer(transformer, spatial_dim = (args.im_h, args.im_w), 
                       dropout=args.cls_dropout, max_len = args.max_len, device = device).to(device)

    loss = nn.CrossEntropyLoss()

    #Disjoint optimization
    head_params = list(model.transformer.encoder.parameters()) + list(model.token_predictor.parameters())  
    SlowOpt = torch.optim.AdamW(head_params, lr=args.head_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    tail_params = list(model.transformer.decoder.parameters()) + list(model.querypos_embed.parameters()) + list(model.firstfix_linear.parameters()) 
    FastOpt = torch.optim.AdamW(tail_params, lr=args.tail_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)

    best_val_loss = np.inf
    patience = 5  # 早期停止的耐心次数
    counter = 0
    start_epoch = 1

    if retraining:
        checkpoint = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        SlowOpt.load_state_dict(checkpoint['optim_slow'])
        FastOpt.load_state_dict(checkpoint['optim_fast'])
        start_epoch = checkpoint['epoch'] + 1
        print("Retraining from", start_epoch)

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = timer()
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train(epoch=epoch, args=args, model=model,
                               SlowOpt=SlowOpt, FastOpt=FastOpt, loss_fn=loss,
                               train_dataloader=train_dataloader, model_dir=model_dir,
                               model_name=model_name, device=device)
        end_time = timer()
        valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1 = evaluate(model=model, loss_fn=loss,
                               valid_dataloader=valid_dataloader, device=device)
        
        output_str = (f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train accuracy: {train_accuracy:.2f}%, "
                  f"Train precision: {train_precision:.2f}%, Train recall: {train_recall:.2f}%, Train F1: {train_f1:.2f}, "
                  f"Val loss: {valid_loss:.3f}, Val accuracy: {valid_accuracy:.2f}%, "
                  f"Val precision: {valid_precision:.2f}%, Val recall: {valid_recall:.2f}%, Val F1: {valid_f1:.2f}, "
                  f"Epoch time = {(end_time - start_time):.3f}s\n")
        
        print(output_str)
        with open(logfile, "a") as myfile:
            myfile.write(output_str)
            myfile.close()

        # Save model
        checkpoint_path = join(model_dir, f"{model_name}_epoch{epoch}_valacc{valid_accuracy:.2f}_valloss{valid_loss:.3f}.pth")
        save_model_train(args, model, SlowOpt, FastOpt, epoch, checkpoint_path)
    
        # Early stopping
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break        


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gazeformer Train', parents=[get_args_parser_train()])
    args = parser.parse_args()
    main(args)
