import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models import ResNet18

e = 0.025
dataset = 'cifar10'
model_name = "resnet18"
device = 'cuda:2'
# Decision threshold
p = 0.75
model_source = ResNet18()
model_source.load_state_dict(torch.load(f'saved_models/{dataset}/source/0/{model_name}_model.th'))
model_source.to(device)
model_source.eval()
surrogate_model = ResNet18()
surrogate_model.load_state_dict(torch.load(f'saved_models/{dataset}/surrogate/15/{model_name}_model.th'))
surrogate_model.to(device)
surrogate_model.eval()
reference_model = ResNet18()
reference_model.load_state_dict(torch.load(f'saved_models/{dataset}/reference/15/{model_name}_model.th'))
reference_model.to(device)
reference_model.eval()

with h5py.File(f'fingerprints/e_{e}/fingerprint_train.h5', 'r') as h5file:
    fingerprint = np.array(h5file["fingerprint"])
    fingerprint_labels = np.array(h5file["fingerprint_labels"])
    fingerprint_ground_truth = np.array(h5file["fingerprint_ground_truth"])
fingerprint = np.reshape(np.array(fingerprint), (-1, 32, 32, 3))
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
confusion_matrix = confusion_matrix(
    fingerprint_ground_truth,
    fingerprint_labels,
    # labels=labels,
    normalize='all'
)
fig, ax = plt.subplots(figsize=(10, 4))
ConfusionMatrixDisplay(confusion_matrix * 100, labels).plot(include_values=True, cmap='magma', ax=ax)  # ax=ax
im = ax.images
cbar = im[-1].colorbar
cbar.set_label('Normalized % Ratio')
plt.tight_layout()
plt.show()

tensor_fingerprint = torch.Tensor(fingerprint)
tensor_fingerprint.to(device)
source_preds = model_source(tensor_fingerprint)
surrogate_preds = surrogate_model(tensor_fingerprint)
reference_preds = reference_model(tensor_fingerprint)

verified_examples = source_preds.eq(surrogate_preds.data.view_as(source_preds)).sum()
CAEAcc_surr = verified_examples / len(source_preds)
print(f"Surrogate CAEAcc: {CAEAcc_surr}")

# Apply decision threshold
if CAEAcc_surr < p:
    print("Model is a Reference Model.")
else:
    print("Model is a Surrogate Model")

verified_examples = source_preds.eq(reference_preds.data.view_as(source_preds)).sum()
CAEAcc_surr = verified_examples / len(source_preds)
print(f"Surrogate CAEAcc: {CAEAcc_surr}")

# Apply decision threshold
if CAEAcc_surr < p:
    print("Model is a Reference Model.")
else:
    print("Model is a Surrogate Model")
