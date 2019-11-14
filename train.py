# imports 
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', action='store', help = 'Enter path to data directory.')
parser.add_argument('--save_dir', action='store', default = '/home/workspace/ImageClassifier/', help = 'Enter path to save model checkpoint.')
parser.add_argument('--arch', action='store', default = 'vgg16', help = 'Enter model architecture(e.g. vgg11).')
parser.add_argument('--learning_rate', action='store', default = 0.002, type = float, help = 'Enter learning rate.')
parser.add_argument('--hidden_units', action='store', default = 512, type = int, help = 'Enter number of units in hidden layer.')
parser.add_argument('--epochs', action='store', default = 1, type = int, help = 'Enter number of epochs for training.')
parser.add_argument('--gpu', action="store_true", default=False, help = 'Include to use GPU for training.')

args = parser.parse_args()
arch = args.arch
save_dir = args.save_dir
data_dir = args.data_dir
epochs = args.epochs
gpu = args.gpu
hidden_units = args.hidden_units
learning_rate = args.learning_rate

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



### Load the Data ###
# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.Resize(255),
                                       transforms.RandomCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

# Using the image datasets and the transforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)
vloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)

# Label - mapping 
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

      
        
### Build and train network ###
# Use GPU if available and requested
if gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

# Load pre-trained network
model = eval('models.' +  arch + '(pretrained=True)')

# Define layer sizes and learning rate
input_size = model.classifier[0].in_features

output_size = 102 # Number of flower classes

# Define un-trained network
from collections import OrderedDict
model.classifier = nn.Sequential(OrderedDict([
                           ('fc1',nn.Linear(input_size, hidden_units)),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=0.2)),
                           ('fc2',nn.Linear(hidden_units,512)),
                           ('ReLu2',nn.ReLU()),
                           ('Dropout2',nn.Dropout(p=0.2)),
                           ('fc3',nn.Linear(512,output_size)),
                           ('output',nn.LogSoftmax(dim=1))
                           ]))

# Define error function
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# Move model to correct device if not already there
model.to(device);


# Define deep learning method
 
steps = 0 
running_loss = 0
print_every = 10


for epoch in range(epochs):
    running_loss = 0
    for inputs,labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zeroing parameter gradients
        optimizer.zero_grad()

        # forward and backward passes
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # validation step 
        if steps % print_every == 0:
            
            valid_loss = 0
            accuracy = 0 
            model.eval()
            
            # Gradients are turned off as no longer in training
            with torch.no_grad():
                ## change testloader to vloader
                for inputs, labels in vloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(vloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(vloader):.3f}")
        
            running_loss = 0
            
            # Turning training back on
            model.train()
            
            
# Save the checkpoint
checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'class_to_idx': train_data.class_to_idx,
              'hidden_layers': hidden_units,
              'state_dict': model.state_dict()}

torch.save(checkpoint, save_dir + '/checkpoint_SOH.pth')
