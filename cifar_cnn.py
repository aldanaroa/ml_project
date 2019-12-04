# Creator: Hoang-Dung Do

import matplotlib.pyplot as plt
import sklearn.metrics
import torch
import torch.nn as nn
import json

import cifar_dataprocess
import cifar_models


def epoch_search(cnn, x_train, x_cv, y_train, y_cv, epoch_min=1, epoch_max=100):
    training_loss_ = []
    validation_loss_ = []
    learning_rate = 0.01
    batch_size = 1000
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)

    print("Start training")

    for epoch in range(epoch_min, epoch_max + 1):
        running_loss = 0.0
        train_loss = 0.0
        num_batches = 0
        print("Epoch %d" % epoch)
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size].clone()  # Slice out a mini-batch of features
            y_batch = y_train[i:i + batch_size].clone()  # Slice out a mini-batch of targets

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(x_batch)
            loss = loss_func(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss += loss.item()
            num_batches += 1

        with torch.no_grad():
            training_loss_.append(train_loss / num_batches)

            val_loss = 0.0
            val_num_batches = 0

            for j in range(0, len(x_cv), batch_size):
                x_val = x_cv[j:j + batch_size]  # Slice out a mini-batch of features
                y_val = y_cv[j:j + batch_size]  # Slice out a mini-batch of targets

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                val_outputs = cnn(x_val)
                loss = loss_func(val_outputs, y_val)

                val_loss += loss.item()
                val_num_batches += 1

            validation_loss_.append(val_loss / val_num_batches)

    return cnn, training_loss_, validation_loss_


def train_and_export(cnn, x_train, x_cv, y_train, y_cv, model_file, result_file, max_epoch):
    trained_cnn, training_loss, validation_loss = epoch_search(cnn, x_train, x_cv, y_train, y_cv,
                                                               epoch_min=1, epoch_max=max_epoch)
    torch.save(trained_cnn.state_dict(), model_file)
    result = {
        "train": training_loss,
        "cv": validation_loss}
    with open(result_file, 'w') as f:
        json.dump(result, f)


def show_conv1_layer(model):
    kernels = model.conv1.weight.detach()
    fig, axarr = plt.subplots(1, kernels.size(0))
    for idx in range(kernels.size(0)):
        img = kernels[idx].detach().numpy().transpose(1, 2, 0) + 0.5
        axarr[idx].imshow(img)
    plt.show()


def plot_loss(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        plt.plot(data["train"], label="train")
        plt.plot(data["cv"], label="cv")
        plt.legend()
        plt.show()


def score(net, x, y):
    batch_size = 1000
    accuracy = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i: i + batch_size].clone()
        outputs = net(x_batch)
        _, y_pred = torch.max(outputs, 1)

        y_batch = y[i: i + batch_size]
        accuracy += sklearn.metrics.accuracy_score(y_batch, y_pred.cpu()) * (batch_size / len(x))

    print("\tSimple cnn accuracy: {0:.2f}%".format(accuracy * 100))


x_train, x_cv, y_train, y_cv, x_test, y_test = cifar_dataprocess.aug_split()

x_train_tensor = torch.tensor(x_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train)

x_cv_tensor = torch.tensor(x_cv, dtype=torch.float)
y_cv_tensor = torch.tensor(y_cv)

x_test_tensor = torch.tensor(x_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test)

net = cifar_models.CNN4Conv()
train_and_export(net, x_train_tensor, x_cv_tensor, y_train_tensor, y_cv_tensor,
                 "model/cifar10_3conv.pth", "model/cifar10_3conv.json", 60)

score(net, x_test_tensor, y_test_tensor)
