# Creator: Hoang-Dung Do

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.metrics
import torch
import torch.nn.functional as func
import torch.nn as nn
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

    return x_train_, x_test_, y_train_, y_test_


def train_test_split(x_tensor_, y_tensor_):
    """
    :param x_tensor_:
    :param y_tensor_:
    :return:
    """
    x_train_ = x_tensor_[0:40000]
    x_test_ = x_tensor_[40001:50000]
    y_train_ = y_tensor_[0:40000]
    y_test_ = y_tensor_[40001:50000]

    return x_train_, x_test_, y_train_, y_test_


def epoch_search(cnn, x_train, x_cv, y_train, y_cv, epoch_min=1, epoch_max=100):
    training_loss_ = []
    validation_loss_ = []

    batch_size = 1000
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.02, momentum=0.9)

    for epoch in range(epoch_min, epoch_max + 1):
        running_loss = 0.0
        loss = 0

        for i in range(0, len(x_train), batch_size):

            x_batch = x_train[i:i + batch_size]  # Slice out a mini-batch of features
            y_batch = y_train[i:i + batch_size]  # Slice out a mini-batch of targets

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
                print('epoch[%d] loss: %.3f' % (epoch, loss.item()))
                running_loss = 0.0

        with torch.no_grad():
            train_outputs = cnn(x_train.float())
            t_l = loss_func(train_outputs, y_train)
            training_loss_.append(t_l.item())

            validation_outputs = cnn(x_cv.float())
            v_l = loss_func(validation_outputs, y_cv)
            validation_loss_.append(v_l.item())

    return cnn, training_loss_, validation_loss_


def run(cnn, x_train, x_cv, y_train, y_cv, model_file, result_file, max_epoch):
    trained_cnn, training_loss, validation_loss = epoch_search(cnn, x_train, x_cv, y_train, y_cv,
                                                               epoch_min=1, epoch_max=max_epoch)
    torch.save(trained_cnn.state_dict(), model_file)
    result = {
        "train": training_loss,
        "cv": validation_loss}
    with open(result_file, 'w') as f:
        json.dump(result, f)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(16 * 8 * 8, 250)
        self.fc2 = nn.Linear(250, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = x.view(-1, 16 * 8 * 8)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN3L(nn.Module):
    def __init__(self):
        super(CNN3L, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def show_conv1_layer(model):
    kernels = model.conv1.weight.detach()
    fig, axarr = plt.subplots(1, kernels.size(0))
    for idx in range(kernels.size(0)):
        img = kernels[idx].detach().numpy().transpose(1, 2, 0) + 0.5
        axarr[idx].imshow(img)
    plt.show()


img_train, img_test, label_train, label_test = load_data()

img_train_tensor = torch.tensor(img_train, dtype=torch.double).reshape((-1, 3, 32, 32))
label_train_tensor = torch.tensor(label_train)

img_test_tensor = torch.tensor(img_test, dtype=torch.double).reshape((-1, 3, 32, 32))
label_test_tensor = torch.tensor(label_test)

img_cv_train, img_cv, label_cv_train, label_cv = train_test_split(img_train_tensor, label_train_tensor)

# run(SimpleCNN(), img_cv_train, img_cv, label_cv_train, label_cv, "simple_cnn.pth", "result/cifar_simple.json", 20)
run(CNN3L(), img_cv_train, img_cv, label_cv_train, label_cv, "dropout_cnn.pth", "result/cifar_dropout.json", 200)


# with open("result/cifar_dropout.json", 'r') as f:
#     data = json.load(f)
#     plt.plot(data["train"], label="train")
#     plt.plot(data["cv"], label="cv")
#     plt.legend()
#     plt.show()
#
# simple_cnn = SimpleCNN()
# simple_cnn.load_state_dict(torch.load("dropout_2L_cnn.pth"))
# _, label_pred = torch.max(simple_cnn(img_test_tensor.float()), 1)
# simple_accuracy = sklearn.metrics.accuracy_score(label_test, label_pred)
# print("\tSimple cnn accuracy: {0:.2f}%".format(simple_accuracy * 100))

# dropout_cnn = DropoutCNN()
# dropout_cnn.load_state_dict(torch.load("dropout_cnn.pth"))
# _, label_pred = torch.max(dropout_cnn(img_test_tensor.float()), 1)
# dropout_accuracy = sklearn.metrics.accuracy_score(label_test, label_pred)
# print("\tDropout cnn accuracy: {0:.2f}%".format(dropout_accuracy * 100))

# show_conv1_layer(simple_cnn)