# Import libraries
import argparse
import numpy as np

import json
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, models

from PIL import Image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model_name = checkpoint['model_name']

    model = pretrained_model(model_name)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def label_mapping(input_file):
    with open(input_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def pretrained_model():
    #Load a pre-trained network
    model = models.vgg19(pretrained=True)

    # Prevent backpropigation on parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

def process_image(image):
    expects_means = [0.485, 0.456, 0.406]
    expects_std = [0.229, 0.224, 0.225]
           
    pil_image = Image.open(image).convert("RGB")
    
    in_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(expects_means, expects_std)])
    pil_image = in_transforms(pil_image)

    return pil_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image, model, topk=5,gpu):
    # Eval
    model.eval()
    
    if gpu:
        output = output.cuda().float()
    else:
        output = output.float()
    
    # Unsqueeze returns a new tensor with a dimension of size one
    image = image.unsqueeze(0)
    
    # Disabling gradient calculation 
    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)
        
        # Calculate the exponentials
        top_prob = top_prob.exp()
        
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_prob.numpy()[0], mapped_classes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Find name of flower")

    parser.add_argument("-k", "--topk", type=int, required=True, help="top probability items")
    parser.add_argument("-m", "--model", type=str, required=True, help="filepath, filename")
    parser.add_argument("-i", "--image", type=str, required=True, help="filepath,filename image")
    parser.add_argument("-l", "--label", type=str, required=True, help="filepath, filename labels")
    parser.add_argument("-g", "--gpu", help="use GPU", default=False, action="store_true")

    args = parser.parse_args()

    # load model
    model = load_checkpoint(args.model)
    if args.gpu:
        model.to('cuda')
    else:
        model.to('cpu')

    # load image
    img_pil = process_image(args.image)

    # predict
    predict(img_pil, model, args.topk)