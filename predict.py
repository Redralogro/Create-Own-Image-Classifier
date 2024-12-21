import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from helpers import val_test_tranforms
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Create Own Image Classifier Predict code')
parser.add_argument('input')
parser.add_argument('checkpoint')
parser.add_argument('--top_k', type = int, default=5)
parser.add_argument('--category_names', type = str, default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

if args.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device( "mps")
else: device = torch.device('cpu')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    
    # Note: For me the simplest way was to just use the already defined transformation 
    return val_test_tranforms(image)

def predict(image_path: str, model: torch.nn.Module, topk: int = 5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device).eval()
    with torch.no_grad():
        output:torch.Tensor = model.forward(process_image(image_path).unsqueeze(0).float().to(device))
        probs, labels = F.softmax(output.data,dim=1).topk(topk)
        classes: list = []
        for label in labels.cpu().numpy()[0]:
            classes.append(list(model.class_to_idx.keys())[label])
        return probs.cpu().numpy()[0].tolist(), classes

# Load the checkpoint data
checkpoint = torch.load(args.checkpoint)
arch_to_model = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19
}
# Initialize the pretrained model
model_func = arch_to_model.get(checkpoint['architecture'])
if model_func:
    model = model_func(pretrained=True)
else:
    raise ValueError(f"Unsupported architecture: {checkpoint['architecture']}")

# Freeze the feature extractor layers
for param in model.features.parameters():
    param.requires_grad = False

# Define a new classifier for the model
model.classifier = nn.Sequential(
    nn.Linear(25088, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(512, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(128, 102)
)

# Load model weights from checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
probs, classes = predict(args.input, model)

print(probs,[cat_to_name[cat] for cat in classes])