import torch
from torch import nn, optim, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import visdom
from torch.nn import functional as F
import pandas as pd
import os, glob
import random, csv

batchsz = 12000
lr = 6e-5
epochs = 500

cuda_use = torch.cuda.is_available()
device = torch.device('cuda:1')
torch.manual_seed(1234)
viz = visdom.Visdom()

class Read_data(Dataset):

    def __init__(self, root, mode):
        """
        :param root: the path of the dataset
        :param resize: the shape of the image
        :param mode: the use of the dataset (train / validatation / test)
        """
        super(Read_data, self).__init__()

        self.root = root

        # to initialize the label to the picture
        self.name2label = {}
        for name in sorted(os.listdir(root)):
            # os.listdir(): to get the name of the file of the path given
            if not os.path.isdir(os.path.join(root, name)):
                # os.path.isdir(): to decide whether the certain root is a file folder
                # os.path.join( , ): to concatenate the path
                continue

            self.name2label[name] = len(self.name2label.keys())
        self.signals, self.labels = self.load_csv('signals.csv')

        # to split the dataset
        if mode == 'train':  # 60%
            self.signals = self.signals[:int(0.6 * len(self.signals))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':  # 20% = 60% -> 80%
            self.signals = self.signals[int(0.6 * len(self.signals)):int(0.8 * len(self.signals))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:  # 20% = 80% -> end
            self.signals = self.signals[int(0.8 * len(self.signals)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            # os.path.exists(): to check if there is the  certain file
            # to create the csv file
            signals = []
            for name in self.name2label.keys():
                for i in range(51):
                    signals += glob.glob(os.path.join(self.root, name, str(i - 20) + 'dB', '*.csv'))

            random.shuffle(signals)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for sig in signals:
                    name = sig.split(os.sep)[-3]
                    # os.sep: the break like '/' in MAC OS operation system
                    label = self.name2label[name]
                    writer.writerow([sig, label])
                print("write into csv file:", filename)

        # read the csv file
        signals, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                sig, label = row
                label = int(label)
                signals.append(sig)
                labels.append(label)

        assert len(signals) == len(labels)

        return signals, labels

    def __len__(self):
        # this function enable we use len() to get the length of the dataset
        return len(self.signals)


    def __getitem__(self, idx):
        # this function enable we use p[key] to get the value
        # idx - [0 : len(self.images)]
        if idx < 0 or idx > len(self.signals) - 1:
            print("the idx is wrong!")
            os._exit(1)
        sig, label = self.signals[idx], self.labels[idx]

        data = torch.from_numpy(pd.read_csv(sig).values).float()
        label = torch.tensor(label)

        return data, label


h_dim_1_g = 512
h_dim_2_g = 256
h_dim_3_g = 512


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1023 * 2, h_dim_1_g),
            nn.BatchNorm1d(h_dim_1_g),
            nn.ReLU(True),
            nn.Linear(h_dim_1_g, h_dim_2_g),
            nn.ReLU(True),
            nn.Linear(h_dim_2_g, h_dim_3_g),
            nn.ReLU(True),
            nn.Linear(h_dim_3_g, 1023 * 2),
        )

    def forward(self, x):
        # Generator
        x = x.view(x.size(0), -1)
        x_hat = self.net(x)
        x_hat = x_hat.view(x_hat.size(0), 1023, 2)

        return x_hat


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(32)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(64)
        )
        self.line1 = nn.Linear(64*7, 128)
        self.dp1 = nn.Dropout(0.5)
        self.line2 = nn.Linear(128, 64)
        self.dp2 = nn.Dropout(0.5)
        self.out = nn.Linear(64, 4)

    def forward(self, x):
        """

        :param x:
        :return:
        """

        x = F.relu(self.conv1(x))
        x = x.squeeze()
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # x = F.adaptive_max_pool2d(x, [1, 1])

        # flaten operation
        x = x.view(x.size(0), -1)
        # [b, 32*3*3] => [b, 10]
        x = F.relu(self.dp1(self.line1(x)))
        x = F.relu(self.dp2(self.line2(x)))
        x = self.out(x)

        return x


def weights_init(m):
    if isinstance(m, nn.Linear):  # True if the object is an instance or subclass of a class or any element of the tuple
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def gradient_penalty(D, xr, xf):
    """

    :param D:
    :param xr:
    :param xf:
    :return:
    """
    LAMBDA = 1

    # only constrait for Discriminator
    xf = xf.detach()
    # xr = xr.detach()

    # [b, 1] => [b, 2]
    if cuda_use:
        alpha = torch.rand(xf.shape[0], 1, 1, 1).to(
            device)  # Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)
    else:
        alpha = torch.rand(xf.shape[0], 1, 1, 1)
    alpha = alpha.expand_as(xr)  # to keep the same transformation on two axis to each point

    interpolates = alpha * xr + ((1 - alpha) * xf)
    interpolates.requires_grad_()

    disc_interpolates = D(interpolates)
    # print(disc_interpolates.shape)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              # grad_outputs like the weigth matrix times the outputs
                              create_graph=True)[
        0]  # we need to set create_grad True cause we want to caculate the gradient of the norm of the gradient
    # print(gradients.shape)

    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gp


def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        if cuda_use:
            x, y = x.to(device), y.to(device)
        with torch.no_grad():
            x = x.unsqueeze(1)
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y.squeeze()).sum().float().item()

    return correct / total

def main():
    train_db = Read_data('data', mode='train')
    val_db = Read_data('data', mode='val')

    # set up a DataLoader object
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True)
    val_loader = DataLoader(val_db, batch_size=batchsz, shuffle=False)

    torch.manual_seed(1234)
    np.random.seed(23)
    if cuda_use:
        D = Discriminator().to(device)
        G = Generator().to(device)
    else:
        D = Discriminator()
        G = Generator()

    criterion_class = nn.CrossEntropyLoss()
    criterion_rof = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()

    optim_G = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))

    acc_best = 0
    loss_best = 100
    for epoch in range(10000):

        # 1. train discriminator for k steps
        D.train()
        G.train()
        for _ in range(5):

            xr, class_true = next(iter(train_loader))
            if cuda_use:
                xr, class_true = xr.to(device), class_true.to(device)
            xr = xr.unsqueeze(1)
            class_pred_r = D(xr)

            predr = torch.logsumexp(class_pred_r, 1)
            if cuda_use:
                lossr = criterion_rof(predr, torch.ones(predr.shape[0]).to(device))
            else:
                lossr = criterion_rof(predr, torch.ones(predr.shape[0]))
            loss_class = criterion_class(class_pred_r, class_true.squeeze())

            # [b, 2]
            if cuda_use:
                z = torch.randn(xr.shape[0], 1023 * 2).to(device)
            else:
                z = torch.randn(xr.shape[0], 1023 * 2)
            # stop gradient on G
            # [b, 2]
            xf = G(z).detach()
            # [b]
            xf = xf.unsqueeze(1)
            class_pred_f = D(xf)
            predf = torch.logsumexp(class_pred_f, 1)
            if cuda_use:
                lossf = criterion_rof(predf, torch.zeros(predf.shape[0]).to(device))
            else:
                lossf = criterion_rof(predf, torch.zeros(predf.shape[0]))
            # min predf
            # lossf = predf.mean()

            # gradient penalty
            gp = gradient_penalty(D, xr, xf)

            loss_D = lossr + lossf + gp + loss_class
            optim_D.zero_grad()
            loss_D.backward()
            # for p in D.parameters():
            #     print(p.grad.norm())
            optim_D.step()

        # 2. train Generator
        if cuda_use:
            z = torch.randn(batchsz, 1023 * 2).to(device)
        else:
            z = torch.randn(batchsz, 1023 * 2)
        z = z.unsqueeze(1)
        # print(z.shape)
        xf = G(z)
        xf = xf.unsqueeze(1)
        predf = D(xf)
        # max predf
        predf = torch.logsumexp(predf, 1)
        if cuda_use:
            loss_G = criterion_rof(predf, torch.ones(predf.shape[0]).to(device))
        else:
            loss_G = criterion_rof(predf, torch.ones(predf.shape[0]))
        # loss_G = - (predf.mean())
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
        if epoch % 2 == 0:
            D.eval()
            acc = evaluate(D, val_loader)

            viz.line([[float(acc)]], [epoch], win='acc', opts=dict(title='acc', legend=['acc']), update='append')
            print('epoch:', epoch, 'acc:', acc)

if __name__ == '__main__':
    main()