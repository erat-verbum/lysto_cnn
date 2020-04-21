# INITIALISATION

# Import Libraries
from IPython.core.display import display, HTML
import numpy as np
import math

from matplotlib import pyplot as plt

import PIL.Image
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from tqdm.notebook import tqdm
USE_CUDA = torch.cuda.is_available()


# Shrink the margins on the Jupyter Notebook
display(HTML("<style>.container { width:100% !important; }</style>"))


# Pytorch functions
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v


def toTensor(v, dtype=torch.float, requires_grad=False):
    # return cuda(Variable(v.detach().clone().type(dtype).requires_grad_(True)))
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))


def toNumpy(v):
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()


def debug_memory():
    import collections
    import gc
    import resource
    import torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))

# FUNCTION: calc_optimal_hyperparameters_cnn


def calc_optimal_hyperparameters_cnn(X, Y, params, model_name,
                                     nfolds=3, ntrain=0, verbose=0, load_all_data=False):

    # Arrays to keep track of metrics
    corrs = []
    r2s = []
    rmses = []

    av_loss_train_epochs = np.zeros(params['num_epochs'])
    av_loss_val_epochs = np.zeros(params['num_epochs'])

    # Set the maximum number of epochs that will be updated based on when validation error
    # starts increasing again during training
    max_epochs = params['num_epochs']

    if verbose > 0:
        print('Calculating for model parameters:\n', params)

    # Split data according to number of folds
    if nfolds > 1:
        kf = KFold(n_splits=nfolds, random_state=1, shuffle=True)
        splits = kf.split(X)
    else:
        X_len = X.shape[0]  # len(X)
        if ntrain == 0:
            ntrain = int(X_len * 0.67)
        splits = [([i for i in range(ntrain)], [
                   i for i in range(ntrain, X_len, 1)])]

    for j, (train_idx, validate_idx) in enumerate(splits):

        if verbose > 0:
            print('Fold {}:'.format(j))

        ntrain = np.sum(train_idx)
        nval = np.sum(validate_idx)

        # Divide data into training an validation sets for this fold
        X_trainf = X[train_idx]
        Y_trainf = Y[train_idx]
        X_valf = X[validate_idx]
        Y_valf = Y[validate_idx]

        # Define Image transformations
        image_transforms = {
            # Train uses data augmentation
            'train':
            transforms.Compose([
                # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                # transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(size=224),  # resnet standards dimensions
                transforms.Resize(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],  # resnet standard mean
                                     [0.229, 0.224, 0.225])  # resnet standard std
            ]),
            # Validation does not use augmentation
            'val':
            transforms.Compose([
                transforms.Resize(size=224),
                # transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
        }

        # Define datasets
        class ImageDataset(torch.utils.data.Dataset):
            def __init__(self, X, Y, transform=None):
                self.data = X
                self.target = torch.from_numpy(
                    Y.reshape((Y.shape[0], 1))).float()
                self.transform = transform

            def __getitem__(self, index):
                x = PIL.Image.fromarray(self.data[index])
                y = self.target[index]

                # Normalize your data here
                if self.transform:
                    x = self.transform(x)

                return x, y

            def __len__(self):
                return len(self.data)

        # Datasets
        dataset_train = ImageDataset(
            X_trainf, Y_trainf, image_transforms['train'])
        dataset_val = ImageDataset(X_valf, Y_valf, image_transforms['val'])

        # Data Loaders (Input Pipeline)
        loader_train = torch.utils.data.DataLoader(
            dataset=dataset_train, batch_size=params['batch_size'], shuffle=True)
        loader_val = torch.utils.data.DataLoader(
            dataset=dataset_val, batch_size=params['batch_size'], shuffle=False)

        # Instantiate Model
        # model = models.wide_resnet101_2(pretrained=True)
        model = models.resnet152(pretrained=True)

        # Freeze all model parameters
        for param in model.parameters():
            param.requires_grad = False

        # New final layer with 1 class
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, 2 * num_ftrs),
                                       nn.Dropout(0.5),
                                       nn.ReLU(),
                                       torch.nn.Linear(2 * num_ftrs, 2 * num_ftrs),
                                       nn.Dropout(0.5),
                                       nn.ReLU(),
                                       torch.nn.Linear(2 * num_ftrs, 2 * num_ftrs),
                                       nn.Dropout(0.5),
                                       nn.ReLU(),
                                       torch.nn.Linear(2 * num_ftrs, 2 * num_ftrs),
                                       nn.Dropout(0.5),
                                       nn.ReLU(),
                                       torch.nn.Linear(2 * num_ftrs, 2 * num_ftrs),
                                       nn.Dropout(0.5),
                                       nn.ReLU(),
                                       nn.Linear(2 * num_ftrs, 1)
                                       )
        # Send model to GPU
        model = cuda(model)

        # Loss and Optimizer
        criterion = nn.MSELoss()  # CrossEntropyLoss()  #loss function
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params['learning_rate'])

        # Track losses
        loss_train_epochs = []
        loss_val_epochs = []

        # TRAIN THE MODEL
        loss_train_epochs = np.zeros(params['num_epochs'])
        loss_val_epochs = np.zeros(params['num_epochs'])

        # Loop Over Epochs
        if verbose > 0:
            disable_progress_bar = False
        else:
            disable_progress_bar = True

        for epoch in tqdm(range(max_epochs)):
            loss_train_epoch = 0
            loss_val_epoch = 0

            # Train Model For One Epoch + Calculate loss on training data
            model.train()
            for i, (features_train, labels_train) in enumerate(loader_train):

                # Move features and labels to GPU
                features_train = cuda(features_train)
                labels_train = cuda(labels_train)

                # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer
                outputs = model(features_train)
                loss = criterion(outputs, labels_train)
                loss.backward()
                optimizer.step()

                # Track loss
                loss_train_epoch += loss

            loss_train_epoch = loss_train_epoch / params['batch_size']
            loss_train_epochs[epoch] = loss_train_epoch

            # Calculate loss on validation data
            model.eval()
            for i, (features_val, labels_val) in enumerate(loader_val):

                # Convert torch tensor to Variable
                if not load_all_data:
                    features_val = cuda(features_val)
                    labels_val = cuda(labels_val)

                # Predict labels
                labels_pred = model(features_val)

                # Calculate loss
                loss = criterion(labels_pred, labels_val)

                # Track loss
                loss_val_epoch += loss

            loss_val_epoch = loss_val_epoch / params['batch_size']
            loss_val_epochs[epoch] = loss_val_epoch

            if verbose > 1:
                if epoch == 0:
                    print('Epoch \t\t Train Loss \t\t Val Loss')

                print('{} \t\t {} \t {}'.format(
                    epoch, loss_train_epoch, loss_val_epoch))

            if j == 0 and epoch > 1:
                if np.mean(loss_val_epochs[epoch - 2:epoch - 1]) <= loss_val_epoch:
                    max_epochs = epoch + 1
                    break

        # Store epoch losses across folds
        av_loss_train_epochs = av_loss_train_epochs + \
            np.asarray(loss_train_epochs) / nfolds
        av_loss_val_epochs = av_loss_val_epochs + \
            np.asarray(loss_val_epochs) / nfolds

        # Calculate correlation, R2 and RMSE over final state of model
        model.eval()

        # Predict labels
        Y_predf_np = []
        model.eval()
        for i, (features_val, labels_val) in enumerate(loader_val):
            # Convert torch tensor to Variable
            if not load_all_data:
                features_val = cuda(features_val)
                labels_val = cuda(labels_val)

            # Predict labels
            labels_pred = toNumpy(model(features_val))

            Y_predf_np += list(labels_pred)

        Y_predf_np = np.asarray(Y_predf_np).flatten()
        Y_valf_np = Y_valf.flatten()

        if verbose > 2:
            plt.scatter(Y_valf_np, Y_predf_np, s=15, alpha=0.3)
            x = np.linspace(np.min(Y_valf_np), np.max(Y_valf_np), 100)
            y = x
            plt.plot(x, y, '-r')
            plt.title('Fold {}'.format(j))
            plt.xlabel('True Cell Counts')
            plt.ylabel('Predicted Cell Counts')
            plt.show()

        if np.var(Y_predf_np) == 0:
            corrs.append(0)
        else:
            corrs.append(np.corrcoef(Y_predf_np, Y_valf_np)[0, 1])
        rmses.append(math.sqrt(mean_squared_error(Y_predf_np, Y_valf_np)))
        r2s.append(r2_score(Y_predf_np, Y_valf_np))

    # Calculate average validation metrics
    corr_val = np.mean(corrs)
    rmse_val = np.mean(rmses)
    r2_val = np.mean(r2s)
    metrics_val = {'0- Mean Val Corr Coeff': corr_val,
                   '1- Mean Val RMSE': rmse_val,
                   '2- Mean Val R2 Score': r2_val}

    if verbose > 0:
        for key, value in metrics_val.items():
            print(key, ':', value)

    if verbose > 0:
        fig, ax = plt.subplots(1, 1)
        plt.title('Loss')
        ax.plot([i for i in range(max_epochs)],
                av_loss_train_epochs[:max_epochs], label='training')
        ax.plot([i for i in range(max_epochs)],
                av_loss_val_epochs[:max_epochs], label='validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()

    return metrics_val, av_loss_train_epochs, av_loss_val_epochs, params


if __name__ == '__main__':

    # Load X, P and Y from NPZ files
    loaded = np.load('./data/breast.npz')
    X = loaded['X']
    Y = loaded['Y']
    P = loaded['P']

    # Select features
    ntrain = 5841
    X_train = X
    Y_train = Y

    # Choose invariant hyperparamenters
    nfolds = 1
    num_epochs = 20
    batch_size = 120
    learning_rate = 5e-5

    params = {'num_epochs': num_epochs,
              'batch_size': batch_size,
              'learning_rate': learning_rate}

    metrics, loss_train, loss_val, params = calc_optimal_hyperparameters_cnn(
        X_train, Y_train, params, 'cnn',
        nfolds=nfolds, verbose=3, load_all_data=False, ntrain=ntrain)
