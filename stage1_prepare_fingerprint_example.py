import argparse
import csv
import os
import time

import h5py
import numpy as np
import torch
from tensorflow import keras
from models import *


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    # input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)
    return torch.sum(-input_logits * torch.log(target_logits))


def conferrable_ensemble_method(dataset, surrogate_no=2, reference_no=2):
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255

    # If subtract pixel mean is enabled
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean

    # Convert class vectors to binary class vectors.
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    idx_train = np.random.permutation(len(x_train))
    x_train, y_train = x_train[idx_train], y_train[idx_train]

    source_model = ResNet18()
    source_model.load_state_dict(torch.load('checkpoint/source/model_last.th'))

    reference_models = [ResNet18()]
    reference_models[0].load_state_dict(torch.load('checkpoint/source/model_last.th'))

    surrogate_models = [ResNet18()]
    surrogate_models[0].load_state_dict(torch.load('checkpoint/source/model_last.th'))

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
    for e in epsilon:
        print(f"Epsilon: {e}")
        fingerprint = []
        fingerprint_labels = []
        fingerprint_ground_truth = []
        fingerprint_stats = []
        for example_no, example in enumerate(x_train[:10]):
            if example_no == 10:
                break
            ground_truth = np.argmax(y_train[example_no])
            print(f"Example No: {example_no} Label: {ground_truth}")
            example = torch.Tensor(example).unsqueeze(0).cuda()
            initial_source_prediction = source_model(example)
            output = initial_source_prediction.data.max(1)[1]
            if output[0] != ground_truth:
                print("Example initially predicted wrong. Skipping..")
                continue
            adversarial_example = example.detach()
            i = 0
            while i < 100:
                adversarial_example.requires_grad_()
                with torch.enable_grad():
                    source_prediction = source_model(adversarial_example)

                    reference_prediction = torch.zeros_like(initial_source_prediction)
                    model_selection = np.random.binomial(1, 0.7, reference_no)
                    mask_r = np.ma.make_mask(model_selection)
                    reference_models_dropped = np.array(reference_models)[mask_r]
                    for reference_model in reference_models_dropped:
                        prediction_r = reference_model(adversarial_example)
                        reference_prediction += prediction_r
                    reference_prediction /= len(reference_models_dropped)

                    surrogate_prediction = torch.zeros_like(initial_source_prediction)
                    model_selection = np.random.binomial(1, 0.7, surrogate_no)
                    mask_s = np.ma.make_mask(model_selection)
                    surrogate_models_dropped = np.array(surrogate_models)[mask_s]
                    for surrogate_model in surrogate_models_dropped:
                        prediction_s = surrogate_model(adversarial_example)
                        surrogate_prediction += prediction_s
                    surrogate_prediction /= len(surrogate_models_dropped)

                    ensemble_output = surrogate_prediction * (
                            torch.ones_like(reference_prediction) - reference_prediction)  # (1, 10)
                    first_loss_input = ensemble_output[0] -
                    first_loss = softmax_mse_loss(torch.tensor([1.0, 0.0]).cuda(),
                                                  torch.tensor([first_loss_input, 1.0 - first_loss_input]).cuda())
                    second_loss = softmax_mse_loss(F.softmax(initial_source_prediction, dim=1),
                                                   torch.clamp(F.softmax(source_prediction, dim=1), 1e-7, 1. - 1e-7))
                    third_loss = softmax_mse_loss(F.softmax(source_prediction, dim=1),
                                                  F.softmax(surrogate_prediction, dim=1))
                    loss = alpha * first_loss - beta * second_loss + gamma * third_loss
                grad = torch.autograd.grad(loss, [adversarial_example])[0]
                adversarial_example = adversarial_example.detach() + alpha * torch.sign(grad.detach())  # TODO
                adversarial_example = torch.min(torch.max(adversarial_example, example - epsilon), example + epsilon)
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

            # Add example if it's adversarial to source model.
            source_prediction = source_model(adversarial_example)
            label = source_prediction.data.max(1)[1]
            if label != ground_truth:
                # Add example to fingerprint if conferrability score is high enough
                conferrability_score = tf.reshape(tf.reduce_max(ensemble_output_last), ()).numpy()
                print(f"Conferrability Score: {conferrability_score:.2f}")
                if conferrability_score >= threshold:
                    fingerprint.append(adversarial_example.value().numpy())
                    fingerprint_labels.append(label)
                    fingerprint_ground_truth.append(ground_truth)
                    fingerprint_stats.append(
                        {
                            "source_label": f"{label} - {100 * np.max(source_prediction):.2f}%",
                            "average_surrogate_label": f"{np.argmax(surrogate_prediction):.2f} - {100 * np.max(surrogate_prediction):.2f}%",
                            "average_reference_label": f"{np.argmax(reference_prediction):.2f} - {100 * np.max(reference_prediction):.2f}%",
                            "real_label": ground_truth,
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
    arguments_parser.add_argument("surrogate_no", type=int)
    arguments_parser.add_argument("reference_no", type=int)
    arguments_parser.add_argument("--use_saved", help="Use saved models", action="store_true")
    args = arguments_parser.parse_args()

    conferrable_ensemble_method(args.dataset, args.surrogate_no, args.reference_no)
