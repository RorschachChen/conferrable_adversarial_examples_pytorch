import h5py
import numpy as np
import torch
from tensorflow import keras
from models import ResNet18
import matplotlib.pyplot as plt

dataset = 'cifar10'
model_name = "resnet18"
device = 'cuda:2'
surrogate_no = reference_no = 5
epsilon = np.arange(0.01, 0.161, 0.015)
model_source = ResNet18()
model_source.load_state_dict(torch.load(f'saved_models/{dataset}/source/0/{model_name}_model.th'))
model_source.to(device)
model_source.eval()

# Create reference models
reference_models = []
for i in range(reference_no):
    reference_model = ResNet18()
    reference_model.load_state_dict(
        torch.load(f'saved_models/{dataset}/reference/{i + 8}/{model_name}_model.th'))
    reference_model.eval()
    reference_models.append(reference_model)

# Create surrogate models
surrogate_models = []
for i in range(surrogate_no):
    surrogate_model = ResNet18()
    surrogate_model.load_state_dict(
        torch.load(f'saved_models/{dataset}/surrogate/{i + 8}/{model_name}_model.th'))
    surrogate_model.eval()
    surrogate_models.append(surrogate_model)

(_, _), (_, y_test) = keras.datasets.cifar10.load_data()

total_scores = []
for e in epsilon:
    print(f"Epsilon: {e}")
    print("Load Adversarial Examples")
    with h5py.File(f'adversarial_examples/e_{e}/adversarial_examples.h5', 'r') as h5file1:
        bim_examples = np.array(h5file1["BIM"])
        pgd_examples = np.array(h5file1["PGD"])
        fgm_examples = np.array(h5file1["FGM"])
        cwl_examples = np.array(h5file1["CWL"])
    print("Load CEM Examples")
    with h5py.File(f'fingerprints/e_{e}/01_29/fingerprint.h5', 'r') as h5file2:
        fingerprint = np.array(h5file2["fingerprint"])
    cem_examples = np.reshape(np.array(fingerprint), (-1, 32, 32, 3))
    adversarial_examples = [bim_examples, pgd_examples, fgm_examples, cwl_examples, cem_examples]
    scores_for_e = []
    for examples_unfiltered in adversarial_examples:
        with torch.no_grad():
            examples_unfiltered = torch.tensor(examples_unfiltered).to(device)
        labels_source_unfiltered = model_source(examples_unfiltered)
        labels_source_unfiltered = labels_source_unfiltered.data.max(1)[1].detach().cpu()

        # Filter unsuccessful.
        examples_list = []
        no = -1
        for label_source_unfiltered, ground_truth in zip(labels_source_unfiltered, y_test[:100]):
            no += 1
            if np.argmax(label_source_unfiltered) != np.argmax(ground_truth):
                examples_list.append(examples_unfiltered[no])

        # Continue with filtered 100 example
        try:
            examples = np.array(examples_list[:100])
        except:
            examples = np.array(examples_list)

        mean_conferrability_scores = 0
        for example in examples:
            example = np.reshape(example, [1, 32, 32, 3])
            with torch.no_grad():
                label_source = model_source(examples_unfiltered)
                # Calculate conferrability scores for each class.
                reference_prediction = torch.zeros_like(label_source)

                for model_reference in reference_models:
                    prediction_r = model_reference(example)
                    reference_prediction += prediction_r
                reference_prediction /= len(reference_models)

                surrogate_prediction = torch.zeros_like(label_source)

                for surrogate_model in surrogate_models:
                    prediction_s = surrogate_model(example)
                    surrogate_prediction += prediction_s
                surrogate_prediction /= len(surrogate_models)

                conferrability_score = np.max(
                    surrogate_prediction * (torch.ones_like(reference_prediction) - reference_prediction))
                mean_conferrability_scores += conferrability_score
        mean_conferrability_scores /= len(examples)
        scores_for_e.append(mean_conferrability_scores)
    total_scores.append(scores_for_e)
bim_scores = [score[0] for score in total_scores]
pgd_scores = [score[1] for score in total_scores]
fgm_scores = [score[2] for score in total_scores]
cwlinf_scores = [score[3] for score in total_scores]
cem_scores = [score[4] for score in total_scores]

plt.plot(epsilon, bim_scores, '-x', label="BIM")
plt.plot(epsilon, pgd_scores, '-x', label="PGD")
plt.plot(epsilon, fgm_scores, '-x', label="FGM")
plt.plot(epsilon, cwlinf_scores, '-x', label="CWLInf")
plt.plot(epsilon, cem_scores, '-x', label="CEM")
plt.xlabel("Epsilon (L_inf)")
plt.ylabel("Conferrability")
plt.title("Conferrability Scores")
plt.legend()
plt.grid()
plt.savefig("fig_last.png")
