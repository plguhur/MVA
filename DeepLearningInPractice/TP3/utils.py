import time
import copy
import torch
from torch import nn
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torchvision.datasets.cifar import CIFAR10

class SubCIFAR10(CIFAR10):
    def __init__(self, root, offset=0, length=100, **kwargs):
        self.offset = offset
        self.length = length
        super().__init__(root, **kwargs)
        
    def __getitem__(self, index):
        return super().__getitem__(index % self.length + self.offset)

    def __len__(self):
        return self.length
    
    
def train_an_epoch(net, criterion, trainloader, optimizer, device, silent=False):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(range(len(trainloader)), disable=silent) as t:
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
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

            
def test(net, testloader, criterion, device, silent=False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    n_samples = len(testloader)
    with torch.no_grad():
        with tqdm(range(len(testloader)), disable=silent) as t:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
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






def plot_history(**kwargs):
    n = len(kwargs)
    plt.figure(figsize=(20,10)) 
    for i, (key, array) in enumerate(kwargs.items()):
        plt.subplot(1,n,i+1)
        plt.plot(array)
        plt.grid(True)
        plt.title(key)
    plt.show()
        
    
def get_loaders(input_size=32, train_transform=None, test_transform=None, batch_size=10):
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
    trainset = SubCIFAR10("datasets", download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    validset = SubCIFAR10("datasets", download=True, transform=test_transform, offset=100, length=1000)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)
    testset = SubCIFAR10("datasets", download=True, transform=test_transform, offset=100, length=50000)
    testloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)
    return trainloader, validloader, testloader