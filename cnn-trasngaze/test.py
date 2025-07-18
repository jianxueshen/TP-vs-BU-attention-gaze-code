import argparse
import os
from os.path import join
import json
import numpy as np
import torch

import matplotlib.pyplot as plt

from models import Transformer
from gazeformer import gazeformer
from utils import seed_everything, get_args_parser_test
from metrics import postprocessScanpaths, get_seq_score, get_seq_score_time
from tqdm import tqdm
import warnings
from PIL import Image
warnings.filterwarnings("ignore")

def test(args, image_ftrs, eye_data):
    trained_model = args.trained_model
    device = torch.device('cuda:{}'.format(args.cuda))
    transformer = Transformer(num_encoder_layers=args.num_encoder, nhead = args.nhead, d_model = args.hidden_dim, num_decoder_layers=args.num_decoder, dim_feedforward = args.hidden_dim, img_hidden_dim = args.img_hidden_dim, device = device).to(device)
    model = gazeformer(transformer = transformer, spatial_dim = (args.im_h, args.im_w), max_len = args.max_len, device = device).to(device)
    model.load_state_dict(torch.load(trained_model, map_location=device)['model'])
    model.eval()

    # Load image features
    image_features = image_ftrs.to(device).unsqueeze(0)

    # Load eye movement data
    eye_data_tensor = torch.tensor([eye_data["X"], eye_data["Y"], eye_data["T"]], dtype=torch.float32).T
    if eye_data_tensor.shape[0] < args.max_len:
        padding = torch.full((args.max_len - eye_data_tensor.shape[0], eye_data_tensor.shape[1]), -3.0)
        eye_data_tensor = torch.cat((eye_data_tensor, padding), dim=0)
    else:
        eye_data_tensor = eye_data_tensor[:args.max_len]
    eye_data_tensor = eye_data_tensor.unsqueeze(0).to(device)

    # Run model
    with torch.no_grad():
        out_token = model(image_features, eye_data_tensor)
        _, predicted = torch.max(out_token, 1)
        result = 'top-down' if predicted.item() == 1 else 'bottom-up'
        print(f'Prediction: {result}')

    return result
    

def plot_eye_data_on_image(image, eye_data):
    plt.imshow(image)
    for i, (x, y, t) in enumerate(zip(eye_data["X"], eye_data["Y"], eye_data["T"])):
        circle = plt.Circle((x, y), radius=t/10, edgecolor='r', facecolor='none', linewidth=2)
        plt.gca().add_patch(circle)
        plt.text(x, y, str(i+1), color='white', fontsize=12, ha='center', va='center')
        if i > 0:
            plt.plot([eye_data["X"][i-1], x], [eye_data["Y"][i-1], y], 'r-', linewidth=2)
    plt.title('Eye Movement Data on Image')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def main(args):
    #seed_everything(args.seed)
    img_feature_name = "000000178980.jpg"
    for root, dirs, files in os.walk(args.img_ftrs_dir):
        for file in files:
            if file == img_feature_name.replace('jpg', 'pth'):
                image_ftrs = torch.load(join(root, file))
                break
    img_dir = "/home/oct/COCO_Search18-and-FV"
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file == img_feature_name:
                image_path = join(root, file)
                image = Image.open(image_path).convert('RGB')
                break
    eye_data = {
        "X": [254,
            346,
            298,
            175,
            295,
            159,
            167,
            219,
            153,
            159,
            162,
            175,
            230,
            157,
            90,
            271],
        "Y": [162,
            119,
            153,
            214,
            168,
            251,
            220,
            223,
            230,
            210,
            219,
            87,
            85,
            173,
            198,
            195],
        "T": [341.0,
            304.0,
            131.0,
            448.0,
            173.0,
            31.0,
            373.0,
            426.0,
            43.0,
            34.0,
            341.0,
            163.0,
            247.0,
            130.0,
            158.0,
            248.0]
    } 
    test(args, image_ftrs, eye_data)
    plot_eye_data_on_image(image, eye_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gaze Transformer Test', parents=[get_args_parser_test()])
    args = parser.parse_args()
    main(args)
    
