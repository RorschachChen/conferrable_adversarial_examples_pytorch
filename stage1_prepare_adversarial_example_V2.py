import os

import h5py
import numpy as np
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, CarliniLInfMethod
from art.estimators.classification import PyTorchClassifier
from tensorflow import keras

from models import *

epsilon = np.arange(0.01, 0.161, 0.015)
MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
model_source = ResNet18()
model_source.load_state_dict(torch.load('checkpoint/source/model_last.th'))
model_source = model_source.cuda()
loss = torch.nn.CrossEntropyLoss()
wrapped_classifier = PyTorchClassifier(model=model_source, loss=loss, input_shape=(3, 32, 32), channels_first=True,
                                       nb_classes=10)
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32') / 255
x_test -= MEAN_CIFAR10
x_test /= STD_CIFAR10
np.random.seed(101)
idx_test = np.random.permutation(len(x_test))
x_test, y_test = x_test[idx_test], y_test[idx_test]
x_test = np.transpose(x_test, [0, 3, 1, 2])
for e in epsilon:
    print(f"Epsilon: {e}")

    bim = BasicIterativeMethod(estimator=wrapped_classifier, eps=e, eps_step=0.01, max_iter=100, targeted=False)
    pgd = ProjectedGradientDescent(estimator=wrapped_classifier, norm=2, eps=e, eps_step=0.01, max_iter=10,
                                   targeted=False)
    fgm = FastGradientMethod(estimator=wrapped_classifier, norm=np.inf, eps=e, eps_step=e, targeted=False)
    cwl = CarliniLInfMethod(classifier=wrapped_classifier, targeted=False, confidence=0.5, learning_rate=0.01,
                            max_iter=50)

    print("Generate BIM")
    bim_examples = bim.generate(x_test[:100])
    print("Generate PGD")
    pgd_examples = pgd.generate(x_test[:100])
    print("Generate FGM")
    fgm_examples = fgm.generate(x_test[:100])
    print("Generate CWL")
    cwl_examples = cwl.generate(x_test[:100])

    dataset_path = f'adversarial_examples/e_{e}/'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    with h5py.File(f'{dataset_path}/adversarial_examples.h5', 'w') as h5f:
        h5f.create_dataset('BIM', data=bim_examples)
        h5f.create_dataset('PGD', data=pgd_examples)
        h5f.create_dataset('FGM', data=fgm_examples)
        h5f.create_dataset('CWL', data=cwl_examples)
