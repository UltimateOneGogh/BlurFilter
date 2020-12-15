import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128

# scale imported image and transform it into a torch tensor
image_to_tensor = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])
# mean and std from images in train set of original vgg network
vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

tensor_to_image = transforms.ToPILImage()


def gram_matrix(inp):
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a feature map (N=c*d)
    a, b, c, d = inp.size()

    features = inp.view(a * b, c * d)

    # compute the gram product and 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return features @ features.t() / (a * b * c * d)


def open_image(path):
    image = Image.open(path)
    # fake batch dimension required to fit network's input dimensions
    tensor = image_to_tensor(image).unsqueeze(0)
    return tensor.to(device)


def get_image_from_source(img):
    tensor = image_to_tensor(img).unsqueeze(0)
    return tensor.to(device)


def plot_image(tensor, title=None):
    # clone the tensor to not do changes on it and remove the fake batch dimension
    tensor = tensor.cpu().clone().squeeze(0)
    image = tensor_to_image(tensor)
    plt.imshow(image)
    plt.axis(False)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
