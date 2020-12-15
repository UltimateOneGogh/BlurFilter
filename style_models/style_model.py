import seaborn as sns
import copy
from tqdm.autonotebook import trange
import torch.nn.functional as F
from torch import nn, optim
from style_models.utils import *

sns.set()


class ContentLoss(nn.Module):
    def __init__(self, target, content_weight):
        super().__init__()
        # 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        self.content_weight = content_weight
        self.loss = None

    def forward(self, inp):
        self.loss = F.mse_loss(inp, self.target)
        self.loss *= self.content_weight
        return inp


class StyleLoss(nn.Module):
    def __init__(self, target, style_weight):
        super().__init__()
        self.target_gram = gram_matrix(target).detach()
        self.style_weight = style_weight
        self.loss = None

    def forward(self, inp):
        G = gram_matrix(inp)
        self.loss = F.mse_loss(G, self.target_gram)
        self.loss *= self.style_weight
        return inp


class TotalVariationRegularizationLoss(nn.Module):
    def __init__(self, total_variation_weight):
        super().__init__()
        self.total_variation_weight = total_variation_weight
        self.loss = None

    def forward(self, inp):
        self.loss = torch.sum(torch.abs(inp[:, :, :, :-1] - inp[:, :, :, 1:])) + torch.sum(
            torch.abs(inp[:, :, :-1, :] - inp[:, :, 1:, :]))
        self.loss *= self.total_variation_weight
        return inp


# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean=None, std=None):
        super(Normalization, self).__init__()
        if mean is None:
            mean = vgg_normalization_mean
        if std is None:
            std = vgg_normalization_std
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.as_tensor(mean).view(-1, 1, 1)
        self.std = torch.as_tensor(std).view(-1, 1, 1)

    def forward(self, image):
        """ normalize """
        return (image - self.mean) / self.std


def get_style_model_and_losses(cnn,
                               style_img, content_img,
                               style_weight, content_weight, total_variation_weight,
                               content_layers=None,
                               style_layers=None):
    if not style_layers:
        style_layers = style_layers_default
    if not content_layers:
        content_layers = content_layers_default

    cnn = copy.deepcopy(cnn)

    content_losses = []
    style_losses = []
    variation_loss = TotalVariationRegularizationLoss(total_variation_weight)

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(variation_loss, Normalization().to(device))

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target, content_weight)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target = model(style_img).detach()
            style_loss = StyleLoss(target, style_weight)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses, variation_loss


def run_style_transfer(cnn,
                       content_img, style_img,
                       input_img_type='same', num_steps=900,
                       style_weight=1e9, content_weight=1e-3, total_variation_weight=1e-3):
    content_img = get_image_from_source(content_img)
    style_img = get_image_from_source(style_img)

    plt.figure(figsize=(18, 10))
    plt.subplot(1, 3, 1)
    plot_image(content_img, title='Content Image')
    plt.subplot(1, 3, 2)
    plot_image(style_img, title='Style Image')

    if input_img_type == 'same':
        input_img = content_img.clone()
    elif input_img_type == 'random':
        input_img = torch.randn(content_img.data.size(), device=device)
    else:
        raise NotImplementedError(f"Unsupported input image type {input_img_type}")

    model, style_losses, content_losses, variation_loss = get_style_model_and_losses(cnn, style_img, content_img,
                                                                                     style_weight, content_weight,
                                                                                     total_variation_weight)
    optimizer = optim.Adam([input_img.requires_grad_()])

    for i in trange(num_steps):
        # correct the values of updated input image
        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)

        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        variation_score = variation_loss.loss

        loss = style_score + content_score + variation_score
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(
                f'\r{i}: Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f} Variation Loss: {variation_score.item():4f}',
                end='')

    input_img.data.clamp_(0, 1)

    # plt.subplot(1, 3, 3)
    # plot_image(input_img, title='Output Image')
    return input_img
