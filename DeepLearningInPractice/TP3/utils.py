import time
import copy
import torch
from torch import nn
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets.cifar import CIFAR10
import numpy as np

cifar_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class SubCIFAR10(CIFAR10):
    def __init__(self, root, offset=0, length=100, label=-1, **kwargs):
        self.offset = offset
        self.length = length
        super().__init__(root, **kwargs)
        if label >= 0:
            labels = np.array(self.targets)
            mask = labels == label
            self.data = self.data[mask]
            self.targets = labels[mask].tolist()
        
    def __getitem__(self, index):
        return super().__getitem__(index % self.length + self.offset)

    def __len__(self):
        return self.length
    
    
class SubSTL10(torchvision.datasets.STL10):
    """ This subclass allows to return only images with ONE given label """
    def __init__(self, *args, label=-1, split="train", **kwargs):
        super(SubSTL10, self).__init__(*args, **kwargs)

        if not isinstance(label, int) or label < 0:
            raise ValueError("Please provide an acceptable label index")
        if split != "train":
            raise ValueError("Only works with annotated labels")
            
        self.data = self.data[self.labels == label]
        self.labels = self.labels[self.labels == label]
        
    
    
def train_an_epoch(net, criterion, trainloader, optimizer, device, silent=False, callback=lambda x: x, **kwargs):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(range(len(trainloader)), disable=silent) as t:
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(callback(inputs, **kwargs))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            t.set_postfix(loss=loss.data.tolist())
            t.update()
    return loss.data.tolist()

            
def test(net, testloader, criterion, device, silent=False, callback=lambda x:x, **kwargs):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    n_samples = len(testloader)
    with torch.no_grad():
        with tqdm(range(len(testloader)), disable=silent) as t:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(callback(inputs, **kwargs))
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                t.set_postfix(loss=test_loss/(batch_idx+1), acc=correct/total)
                t.update()
    return correct/total



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    # See https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html for details
    
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



def splines(image):
    image = warp_images(image,
                          (0, 0, h - 1, w - 1), interpolation_order=1, approximate_grid=2)
    image = np.transpose(image, axes=(1, 2, 0)).copy()
    return image


def plot_history(**kwargs):
    n = len(kwargs)
    plt.figure(figsize=(20,10)) 
    for i, (key, array) in enumerate(kwargs.items()):
        plt.subplot(1,n,i+1)
        plt.plot(array)
        plt.grid(True)
        plt.title(key)
    plt.show()

    
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
def plot_examples(trainloader):
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    plt.figure(figsize=(20,10))

    plt.subplot(121)
    plt.grid(False)
    imshow(torchvision.utils.make_grid(images))

    plt.subplot(122)
    plt.hist(labels, bins=len(cifar_classes), rwidth=0.9, color='#607c8e')
    ax = plt.gca()
    plt.xticks(np.arange(len(cifar_classes))*0.9+0.5, cifar_classes, rotation=45, rotation_mode="anchor", ha="right")
    plt.show()

    
def get_loaders(input_size=32, train_transform=None, test_transform=None, batch_size=10, callback=SubCIFAR10, **kwargs):
    if train_transform is None:
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(input_size),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    if test_transform is None:
        test_transform = transforms.Compose(
            [transforms.Resize(input_size),
             transforms.CenterCrop(input_size),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    trainset = callback("datasets", download=True, transform=train_transform, **kwargs)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    validset = callback("datasets", download=True, transform=test_transform, offset=100, length=1000, **kwargs)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)
    testset = callback("datasets", download=True, transform=test_transform, offset=100, length=50000, **kwargs)
    testloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)
    return trainloader, validloader, testloader