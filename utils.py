import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict
import json


# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(device, filepath):
    
    checkpoint = torch.load(filepath)
    
    # Chosen network   
    arch = 'vgg16' ## Need to ensure this is the same as what is being loaded
    
    model = eval('models.' +  arch + '(pretrained=True)')
    
    for param in model.parameters():
        param.requires_grad=False
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(checkpoint['input_size'],checkpoint['hidden_layers'])),
        ('ReLu1',nn.ReLU()),
        ('Dropout1',nn.Dropout(p=0.2)),
        ('fc2',nn.Linear(checkpoint['hidden_layers'],512)),
        ('ReLu2',nn.ReLU()),
        ('Dropout2',nn.Dropout(p=0.2)),
        ('fc3',nn.Linear(512,checkpoint['output_size'])),
        ('output',nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model

    # Converting image to PIL image using image file path
    pil_im = Image.open(f'{image}' + '.jpg')

    # Building image transform
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    ## Transforming image for use with network
    pil_tfd = transform(pil_im)
    
    # Converting to Numpy array 
    array_im_tfd = np.array(pil_tfd)
    
    return array_im_tfd

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(device, image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Loading model - using .cpu() for working with CPUs
    loaded_model = load_checkpoint(device, model)

    loaded_model.to(device)
    
    # Pre-processing image
    img = process_image(image_path)
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0).to(device)

    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(img_add_dim)

    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Loading index and class mapping
    class_to_idx = loaded_model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list
