import argparse
import torch
from torchvision import models
import json
from PIL import Image
import numpy as np

def get_input_args():
    """Parse command-line arguments for the prediction script."""
    parser = argparse.ArgumentParser(description='Make a prediction using a trained neural network model.')

    # Arguments for the prediction
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    return parser.parse_args()

def load_checkpoint(filepath):
    """Load a model checkpoint and rebuild the model."""
    checkpoint = torch.load(filepath)

    # Load the appropriate model architecture
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported architecture {checkpoint['arch']}")

    # Rebuild the classifier
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    # Open the image file
    pil_image = Image.open(image_path)
    
    # Resize the image, keeping the aspect ratio
    if pil_image.size[0] < pil_image.size[1]:
        pil_image = pil_image.resize((256, int(256 * pil_image.size[1] / pil_image.size[0])))
    else:
        pil_image = pil_image.resize((int(256 * pil_image.size[0] / pil_image.size[1]), 256))
    
    # Get the dimensions for the center crop
    width, height = pil_image.size
    new_width, new_height = 224, 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = left + new_width
    bottom = top + new_height

    # Crop the image
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Convert image to Numpy array and scale pixel values
    np_image = np.array(pil_image) / 255.0
    
    # Normalize the image with mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions to color channel first
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image_path, model, topk=5, device='cpu'):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    model.eval()
    model.to(device)

    # Preprocess image and make prediction
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.forward(image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)

    # Convert to list
    top_p = top_p.cpu().numpy().squeeze().tolist()
    top_class = top_class.cpu().numpy().squeeze().tolist()

    # Invert class_to_idx
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[c] for c in top_class]

    return top_p, classes

def main():
    # Get command-line arguments
    args = get_input_args()

    # Load the model from checkpoint
    model = load_checkpoint(args.checkpoint)

    # Set device
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    
    # Make prediction
    probs, classes = predict(args.image_path, model, topk=args.top_k, device=device)
    
    # Load category names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Print the results
    flower_names = [cat_to_name[str(cls)] for cls in classes]
    print(f"Top {args.top_k} classes: {flower_names}")
    print(f"Probabilities: {probs}")

if __name__ == '__main__':
    main()