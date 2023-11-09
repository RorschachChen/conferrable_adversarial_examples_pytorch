import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models import ResNet18


def prepare_models(dataset, model_name, use_saved, surrogate_no, reference_no, device):
    surrogate_models = []
    reference_models = []

    if use_saved:
        model_source = ResNet18()
        model_source.load_state_dict(torch.load(f'saved_models/{dataset}/source/0/{model_name}_model.th'))

        # Create reference models
        for i in range(reference_no):
            reference_model = ResNet18()
            reference_model.load_state_dict(
                torch.load(f'saved_models/{dataset}/reference/{i + 1}/{model_name}_model.th'))
            reference_models.append(reference_model)

        # Create surrogate models
        for i in range(surrogate_no):
            surrogate_model = ResNet18()
            surrogate_model.load_state_dict(
                torch.load(f'saved_models/{dataset}/surrogate/{i + 1}/{model_name}_model.th'))
            surrogate_models.append(surrogate_model)
    else:
        model_source = ResNet18()
        model_source.load_state_dict(torch.load(f'saved_models/{dataset}/source/0/{model_name}_model.th'))

        MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
        STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])
        orig_train = CIFAR10(root='/home/cym/data', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(orig_train, batch_size=128, shuffle=True, num_workers=0)

        # Create reference models
        for i in range(reference_no):
            reference_model = ResNet18()
            reference_models.append(reference_model)

        # Fit reference models
        for no, r_model in enumerate(reference_models):
            print(f"Fitting reference No: {no}")
            train(r_model, train_loader, device)
            os.makedirs(f'saved_models/{dataset}/reference/{no + 1}/', exist_ok=True)
            torch.save(r_model.state_dict(), f'saved_models/{dataset}/reference/{no + 1}/{model_name}_model.th')

        # get pseudo label
        model_source.to(device)
        model_source.eval()
        preds = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                output = model_source(images)
                pred = output.data.max(1)[1]
                preds.append(pred.detach().cpu())
        preds = torch.cat(preds, dim=0)
        preds_numpy = preds.numpy()
        orig_train.targets = preds_numpy
        train_loader_surrogate = DataLoader(orig_train, batch_size=128, shuffle=True, num_workers=0)

        # Create surrogate models
        for i in range(surrogate_no):
            surrogate_model = ResNet18()
            surrogate_models.append(surrogate_model)

        # Fit surrogate models
        for no, s_model in enumerate(surrogate_models):
            print(f"Fitting surrogate No: {no}")
            train(s_model, train_loader_surrogate, device)
            os.makedirs(f'saved_models/{dataset}/surrogate/{no + 1}/', exist_ok=True)
            torch.save(s_model.state_dict(), f'saved_models/{dataset}/surrogate/{no + 1}/{model_name}_model.th')

    return model_source, reference_models, surrogate_models


def train(model, data_loader, device):
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    for _ in range(200):
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


if __name__ == "__main__":
    prepare_models('cifar10', 'resnet18', False, 16, 18, 'cuda')