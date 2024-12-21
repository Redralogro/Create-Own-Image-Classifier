import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models

import argparse
import os

from helpers import val_test_tranforms, data_transforms
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Create Own Image Classifier Training code')
parser.add_argument('data_dir')
parser.add_argument('--save_dir',default='checkpoints')
parser.add_argument('--arch', default="vgg16", choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'])
parser.add_argument('--learning_rate', action="store", default=0.0025)
parser.add_argument('--hidden_units', type=int, action="store", default=512)
parser.add_argument('--epochs', action="store", default=12)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(train_dir, data_transforms)
image_datasets['val'] = datasets.ImageFolder(train_dir, val_test_tranforms)
image_datasets['test'] = datasets.ImageFolder(train_dir, val_test_tranforms)

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True)
dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size = 64)
dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64)

arch_to_model = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19
}
# Initialize the pretrained model
model_func = arch_to_model.get(args.arch)
if model_func:
    model = model_func(pretrained=True)
else:
    raise ValueError(f"Unsupported architecture: {args.arch}")

for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(
          nn.Linear(25088, args.hidden_units),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(args.hidden_units, 128),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(128, 102),
        )

model.classifier = classifier
if args.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device( "mps")
else: device = torch.device('cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.0025)
model.to(device)

from tqdm import tqdm

num_epochs = 12
cumulative_train_loss = 0

for epoch in tqdm(range(num_epochs)):
    model.train()
    for data, target in dataloaders['train']:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(data)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()
        
        cumulative_train_loss += loss.item()

    # Run validation at the end of every 1 epochs
    if (epoch + 1) % 1 == 0:
        validation_loss = 0
        validation_accuracy = 0
        model.eval()
        with torch.no_grad():
            for data, target in dataloaders['val']:
                data, target = data.to(device), target.to(device)

                # Get predictions and calculate validation loss
                outputs = model(data)
                loss = criterion(outputs, target)
                validation_loss += loss.item()

                # Calculate probabilities using softmax
                softmax_probs = F.softmax(outputs, dim=1)
                max_probs, predicted_labels = torch.max(softmax_probs, dim=1)
                
                # Check correctness of predictions
                matches = predicted_labels.eq(target)
                validation_accuracy += matches.float().mean().item()

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Training Loss: {cumulative_train_loss / len(dataloaders['train']):.3f}, "
              f"Validation Loss: {validation_loss / len(dataloaders['val']):.3f}, "
              f"Validation Accuracy: {validation_accuracy / len(dataloaders['val']):.3f}")
        
        cumulative_train_loss = 0
        model.train()

test_loss = 0
correct_predictions = 0
total_samples = 0
model.eval()  # Enable evaluation mode

with torch.no_grad():
    for data, target in dataloaders['test']:
        data, target = data.to(device), target.to(device)

        # Forward pass and loss calculation
        predictions = model(data)
        test_loss += criterion(predictions, target).item()

        # Calculate probabilities and make predictions
        probabilities = F.softmax(predictions, dim=1)
        _, predicted_class = torch.max(probabilities, dim=1)
        
        # Update accuracy metrics
        correct_predictions += (predicted_class == target).sum().item()
        total_samples += target.size(0)

# Compute final accuracy
test_accuracy = correct_predictions / total_samples

print(f"Test Loss: {test_loss / len(dataloaders['test']):.3f}, "
      f"Test Accuracy: {test_accuracy:.3f}")
from datetime import datetime

checkpoint = {
    'epochs': num_epochs,
    'learning_rate': 2e-3,
    'architecture': args.arch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion_state_dict': criterion.state_dict(),
    'class_to_idx': image_datasets['train'].class_to_idx
}

os.makedirs(args.save_dir,exist_ok=True)
torch.save(checkpoint, f'{args.save_dir}/{args.arch}_{datetime.now()}.pth')