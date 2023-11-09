import argparse
import csv
import os
import time

import h5py
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from cem_helper_func import prepare_models
from models import *


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    return torch.sum(-input_logits * torch.log(target_logits))


def conferrable_ensemble_method(dataset, device, surrogate_no=2, reference_no=2):
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    orig_train = CIFAR10(root='/home/cym/data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(orig_train, batch_size=1, shuffle=True, num_workers=0)

    source_model, reference_models, surrogate_models = prepare_models(dataset='cifar10', model_name='resnet18',
                                                                      use_saved=False,
                                                                      surrogate_no=surrogate_no,
                                                                      reference_no=reference_no, device=device)

    source_model.eval()
    for reference_model in reference_models:
        reference_model.eval()

    for surrogate_model in surrogate_models:
        surrogate_model.eval()

    epsilon = [0.15]
    alpha = 1
    beta = 1
    gamma = 1
    threshold = 0.85
    time_str = time.strftime("%Y_%m_%d_%H")
    source_model = source_model.to(device)
    source_model.eval()
    for e in epsilon:
        print(f"Epsilon: {e}")
        fingerprint = []
        fingerprint_labels = []
        fingerprint_ground_truth = []
        fingerprint_stats = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            initial_source_prediction = source_model(images)
            pred = initial_source_prediction.data.max(1)[1][0]
            if pred != labels[0]:
                print("Example initially predicted wrong. Skipping..")
                continue
            adversarial_example = images.detach()
            i = 0
            while i < 100:
                adversarial_example.requires_grad_()
                with torch.enable_grad():
                    source_prediction = source_model(adversarial_example)  # (1, 10)
                    reference_prediction = torch.zeros_like(initial_source_prediction)  # (1, 10)
                    model_selection = np.random.binomial(1, 0.7, reference_no)
                    mask_r = np.ma.make_mask(model_selection)
                    reference_models_dropped = np.array(reference_models)[mask_r]
                    for reference_model in reference_models_dropped:
                        reference_model = reference_model.to(device)
                        prediction_r = reference_model(adversarial_example)
                        reference_prediction += prediction_r
                    reference_prediction /= len(reference_models_dropped)  # (1, 10)

                    surrogate_prediction = torch.zeros_like(initial_source_prediction)  # (1, 10)
                    model_selection = np.random.binomial(1, 0.7, surrogate_no)
                    mask_s = np.ma.make_mask(model_selection)
                    surrogate_models_dropped = np.array(surrogate_models)[mask_s]
                    for surrogate_model in surrogate_models_dropped:
                        surrogate_model = surrogate_model.to(device)
                        prediction_s = surrogate_model(adversarial_example)
                        surrogate_prediction += prediction_s
                    surrogate_prediction /= len(surrogate_models_dropped)  # (1, 10)

                    ensemble_output = surrogate_prediction * (
                            torch.ones_like(reference_prediction) - reference_prediction)  # (1, 10)
                    source_pred = source_prediction.data.max(1)[1][0]
                    first_loss_input = ensemble_output[0][source_pred]
                    first_loss = softmax_mse_loss(torch.tensor([1.0, 0.0]).cuda(),
                                                  torch.tensor([first_loss_input, 1.0 - first_loss_input]).cuda())
                    second_loss = softmax_mse_loss(F.softmax(initial_source_prediction, dim=1),
                                                   torch.clamp(F.softmax(source_prediction, dim=1), 1e-7, 1. - 1e-7))
                    third_loss = softmax_mse_loss(F.softmax(source_prediction, dim=1),
                                                  F.softmax(surrogate_prediction, dim=1))
                    loss = alpha * first_loss - beta * second_loss + gamma * third_loss
                grad = torch.autograd.grad(loss, [adversarial_example])[0]
                adversarial_example = adversarial_example.detach() + alpha * torch.sign(grad.detach())  # TODO
                adversarial_example = torch.min(torch.max(adversarial_example, images - epsilon), images + epsilon)
                adversarial_example = torch.clamp(adversarial_example, 0, 1)
                i += 1

            # Conferrability is calculated using all models.
            reference_prediction_last = torch.zeros_like(initial_source_prediction)
            for reference_model in reference_models:
                prediction_r = reference_model(adversarial_example)
                reference_prediction_last += prediction_r
            reference_prediction_last /= len(reference_models)

            surrogate_prediction_last = torch.zeros_like(initial_source_prediction)
            for surrogate_model in surrogate_models:
                prediction_s = surrogate_model(adversarial_example)
                surrogate_prediction_last += prediction_s
            surrogate_prediction_last /= len(surrogate_models)

            ensemble_output_last = surrogate_prediction_last * (
                    torch.ones_like(reference_prediction_last) - reference_prediction_last)

            # Add example if it's adversarial to source model.
            source_prediction = source_model(adversarial_example)
            label = source_prediction.data.max(1)[1]
            if label != labels[0]:
                # Add example to fingerprint if conferrability score is high enough
                conferrability_score = ensemble_output_last.data.max(1)[0]
                print(f"Conferrability Score: {conferrability_score:.2f}")
                if conferrability_score >= threshold:
                    fingerprint.append(adversarial_example.value().numpy())
                    fingerprint_labels.append(label)
                    fingerprint_ground_truth.append(labels[0])
                    fingerprint_stats.append(
                        {
                            "source_label": f"{label} - {100 * source_prediction.data.max(1)[0]:.2f}%",
                            "average_surrogate_label": f"{surrogate_prediction.data.max(1)[0]:.2f} - {100 * surrogate_prediction.data.max(1)[0]:.2f}%",
                            "average_reference_label": f"{reference_prediction.data.max(1)[0]:.2f} - {100 * reference_prediction.data.max(1)[0]:.2f}%",
                            "real_label": labels[0],
                            "conferrability_score": round(conferrability_score, 2)
                        }
                    )

        # Save fingerprint stats.
        keys = fingerprint_stats[0].keys()
        stat_path = f'stats/{dataset}/' + time_str
        if not os.path.exists(stat_path):
            os.makedirs(stat_path)

        with open(f'{stat_path}/fingerprint_stats.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(fingerprint_stats)

        # Save generated fingerprints.
        fingerprint = np.reshape(np.array(fingerprint), (-1, 32, 32, 3))
        fingerprint_labels = np.reshape(np.array(fingerprint_labels), (len(fingerprint_labels), 1))
        fingerprint_ground_truth = np.reshape(np.array(fingerprint_ground_truth), (len(fingerprint_ground_truth), 1))

        dataset_path = f'fingerprints/{dataset}/e_{e}/' + time_str
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        with h5py.File(f'{dataset_path}/fingerprint.h5', 'w') as h5f:
            h5f.create_dataset('fingerprint', data=fingerprint)
            h5f.create_dataset('fingerprint_labels', data=fingerprint_labels)
            h5f.create_dataset('fingerprint_ground_truth', data=fingerprint_ground_truth)


if __name__ == "__main__":
    # Arguments parsing
    arguments_parser = argparse.ArgumentParser(description="Run the Exp.")
    arguments_parser.add_argument("dataset", type=str)
    arguments_parser.add_argument("model", type=str)
    arguments_parser.add_argument("device", type=str, default='cuda:2')
    arguments_parser.add_argument("surrogate_no", type=int)
    arguments_parser.add_argument("reference_no", type=int)
    arguments_parser.add_argument("--use_saved", help="Use saved models", action="store_true")
    args = arguments_parser.parse_args()

    conferrable_ensemble_method(args.dataset, args.device, args.surrogate_no, args.reference_no)
