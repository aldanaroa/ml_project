# Creator: Hoang-Dung Do

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as func
import json


def load_data_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return np.array(dict[b'data']), np.array(dict[b'labels'])


def get_labels():
    with open("data/cifar/batches.meta", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b"label_names"]


def load_data():
    x1, y1 = load_data_batch("data/cifar/data_batch_1")
    x2, y2 = load_data_batch("data/cifar/data_batch_2")
    x3, y3 = load_data_batch("data/cifar/data_batch_3")
    x4, y4 = load_data_batch("data/cifar/data_batch_4")
    x5, y5 = load_data_batch("data/cifar/data_batch_5")

    x_train_ = np.vstack([x1, x2, x3, x4, x5])
    y_train_ = np.hstack([y1, y2, y3, y4, y5])

    scaler_ = sklearn.preprocessing.StandardScaler()
    x_train_ = scaler_.fit_transform(x_train_)

    x_test_, y_test_ = load_data_batch("data/cifar/test_batch")
    x_test_ = scaler_.transform(x_test_)

    # import matplotlib.pyplot as plt
    #
    # plt.imshow(x[10].reshape(3, 32, 32).transpose(1, 2, 0))
    # plt.show()

    return x_train_, x_test_, y_train_, y_test_


class ConvoNeuralNet(nn.Module):
    def __init__(self):
        super(ConvoNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(16 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = x.view(-1, 16 * 8 * 8)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cnn = ConvoNeuralNet()
# cnn = torch.nn.Sequential(
#     nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
#     nn.ReLU(),
#     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Linear(16 * 5 * 5, 120),
#     nn.ReLU(),
#     nn.Linear(120, 84),
#     nn.ReLU(),
#     nn.Linear(84, 10))

x_train, x_test, y_train, y_test = load_data()

x_train_tensor = torch.tensor(x_train, dtype=torch.double).reshape((-1, 3, 32, 32))
x_test_tensor = torch.tensor(x_test, dtype=torch.double).reshape((-1, 3, 32, 32))
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# Create an object that can compute "negative log likelihood of a softmax"
loss_func = nn.CrossEntropyLoss()

# Use stochastic gradient descent to train the model
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)

# Use 100 training samples at a time to compute the gradient.
batch_size = 1000

# Make 10 passes over the training data, each time using batch_size samples to compute gradient
num_epoch = 50

x_cv_train = x_train_tensor[0:40000]
x_cv = x_train_tensor[40001:50000]
y_cv_train = y_train_tensor[0:40000]
y_cv = y_train_tensor[40001:50000]

train_loss = []
cv_loss = []
train = []
cv = []
for epoch in range(num_epoch):
    running_loss = 0.0
    loss = 0
    for i in range(0, len(x_cv_train), batch_size):

        x_batch = x_cv_train[i:i + batch_size]  # Slice out a mini-batch of features
        y_batch = y_cv_train[i:i + batch_size]  # Slice out a mini-batch of targets

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(x_batch.float())
        loss = loss_func(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i == 40000 - batch_size:
            print('epoch[%d] loss: %.3f' % (epoch + 1, loss.item()))
            running_loss = 0.0

    _, y_train_pred = torch.max(cnn(x_cv_train.float()), 1)
    _, y_pred = torch.max(cnn(x_cv.float()), 1)
    train_accuracy = sklearn.metrics.accuracy_score(y_cv_train, y_train_pred.detach().numpy())
    cv_accuracy = sklearn.metrics.accuracy_score(y_cv, y_pred.detach().numpy())

    train_loss.append(loss.item())

    test_outputs = cnn(x_cv.float())
    cv_loss_val = loss_func(test_outputs, y_cv)
    cv_loss.append(cv_loss_val.item())

    train.append(train_accuracy)
    cv.append(cv_accuracy)

result = {
    "train": train,
    "cv": cv,
    "train_lost": train_loss,
    "cv_loss": cv_loss}

with open("result/cifar_cv.json", 'w') as f:
    json.dump(result, f)

with open("result/cifar_cv.json", 'r') as f:
    data = json.load(f)
    plt.plot(data["train"], label="train")
    plt.plot(data["cv"], label="cv")
    plt.legend()
    plt.show()

# torch.save(cnn.state_dict(), "cifar.pth")

# cnn.load_state_dict(torch.load("cifar.pth"))

# print("\tTraining accuracy: {0:.2f}%".format(train_accuracy * 100))
# print("\tTesting accuracy: {0:.2f}%".format(test_accuracy * 100))
