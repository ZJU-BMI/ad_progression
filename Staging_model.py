import itertools
import os.path

import numpy
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import StagingDataset, StagingDataset2
import utils
from model import Staging_VAE
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
eps = 1e-8


class StagingModel:
    def __init__(self, x, y, time, label, input_dims, nClusters=3,
                 batch_size=32, epochs=100, lr=1e-4, save_path='train_data'):
        self.n_subj = x.shape[0]
        self.x = torch.from_numpy(x).float().to(device)
        self.y = torch.from_numpy(y).float().to(device)
        self.time = torch.from_numpy(time).float().to(device)
        self.max_time = torch.max(self.time)
        self.label = torch.from_numpy(label).float().to(device)
        self.nClusters = nClusters
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_times = input_dims['num_times']
        self.z_dim = input_dims['z_dim']
        self.save_path = save_path

        staging_dataset = StagingDataset(x, y, time, label)
        staging_dataset_pretrain = StagingDataset(x, y, time, label)
        self.data_loader = DataLoader(dataset=staging_dataset, batch_size=self.batch_size, shuffle=True)
        self.data_loader_pretrain = DataLoader(dataset=staging_dataset_pretrain, batch_size=self.batch_size,
                                               shuffle=True)
        self.model = Staging_VAE(input_dims=input_dims).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.lr_s = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.6)
        self.fitted = False

    def pre_train(self, filepath, pre_epoch=100, pre_lr=1e-4):
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath))
        else:
            opt = torch.optim.Adam(itertools.chain(self.model.encoder.parameters(),
                                                   self.model.decoder.parameters(),
                                                   self.model.decoder_y.parameters(),
                                                   self.model.trans_layer.parameters(),
                                                   self.model.cs_layer.parameters()),
                                   lr=pre_lr, weight_decay=5e-4)

            epoch_bar = tqdm(range(pre_epoch))
            epoch_bar.set_description('Pretraining')
            for i in epoch_bar:
                losses = 0
                l_recon_total = 0
                l_nll_total = 0
                for bh_idx, data in enumerate(self.data_loader_pretrain):
                    (bh_x, bh_y, bh_time, bh_label) = data
                    bh_x, bh_y, bh_time, bh_label = self.to_device(bh_x, bh_y, bh_time, bh_label)

                    xt = torch.cat([bh_x, bh_time.view(-1, 1) / self.max_time], dim=1)

                    z, _ = self.model.encoder(xt)
                    y_recon = self.model.decoder_y(z)
                    hz, h = self.model.transform(bh_y, z)
                    x_recon = self.model.decoder(h)
                    pred_risk = self.model.cs_out(hz)

                    l_recon = F.mse_loss(y_recon, bh_y) + F.mse_loss(x_recon, bh_x)
                    l_nll = self.neg_log_likelihood(pred_risk, bh_time, bh_label)
                    loss = l_recon + l_nll
                    losses += loss.item()
                    l_recon_total += l_recon.item()
                    l_nll_total += l_nll.item()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                epoch_bar.set_postfix({
                    'Loss': '{:.4f}'.format(losses / len(self.data_loader_pretrain)),
                    'Loss recon': '{:.4f}'.format(l_recon_total / len(self.data_loader_pretrain)),
                    'Loss NLL': '{:.4f}'.format(l_nll_total / len(self.data_loader_pretrain)),
                })

            # self.model.encoder.log_var.load_state_dict(self.model.encoder.mu.state_dict())

            Z = []

            with torch.no_grad():
                for bh_idx, data in enumerate(self.data_loader_pretrain):
                    (bh_x, bh_y, bh_time, bh_label) = data
                    bh_x, bh_y, bh_time, bh_label = self.to_device(bh_x, bh_y, bh_time, bh_label)
                    xt = torch.cat([bh_x, bh_time.view(-1, 1) / torch.max(bh_time)], dim=1)
                    z1, z2 = self.model.encoder(xt)
                    Z.append(z1)

            Z = torch.cat(Z, 0).detach().cpu().numpy()
            gmm = GaussianMixture(n_components=self.nClusters, covariance_type='diag')
            gmm.fit_predict(Z)
            self.model.pi.data = torch.log(torch.from_numpy(gmm.weights_).float().to(device))
            # self.model.pi.data = torch.from_numpy(gmm.weights_).float().to(device)
            self.model.mu_k.data = torch.from_numpy(gmm.means_).float().to(device)
            self.model.log_var_k.data = torch.log(torch.from_numpy(gmm.covariances_).float().to(device))
            # print(np.sum(pred))
            torch.save(self.model.state_dict(), filepath)

    def fit(self, pre_train=True, pre_train_epochs=100, save_model=True, pre_lr=1e-4):

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if pre_train:
            self.pre_train(filepath=self.save_path + '/pre_train_weights.pk',
                           pre_epoch=pre_train_epochs, pre_lr=pre_lr)

        if os.path.exists(self.save_path + '/trained_model.pth'):
            state_dict = torch.load(self.save_path + '/trained_model.pth')
            self.model.load_state_dict(state_dict['model'])
            print('load model from local disk...')
        else:
            self.model.train()
            epoch_bar = tqdm(range(self.epochs))
            epoch_bar.set_description('Training')
            for epoch in epoch_bar:
                # self.model.train()
                # print('epoch: ' + str(epoch))
                for bh_idx, data in enumerate(self.data_loader):
                    (bh_x, bh_y, bh_time, bh_label) = data
                    bh_x, bh_y, bh_time, bh_label = self.to_device(bh_x, bh_y, bh_time, bh_label)
                    xt = torch.cat([bh_x, bh_time.view(-1, 1) / self.max_time], dim=1)

                    x_recon, y_recon, pred_risk, z, mu, log_var = self.model(xt, bh_y)

                    l_recon = F.mse_loss(y_recon, bh_y) + F.mse_loss(x_recon, bh_x)
                    l_nll = self.neg_log_likelihood(pred_risk, bh_time, bh_label)

                    kld = self.kld_loss(mu, log_var, z)

                    loss = l_recon + l_nll + kld

                    if torch.isnan(loss):
                        print('nan detected, abort training...')
                        return

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.lr_s.step()

                with torch.no_grad():
                    self.model.eval()
                    xt = torch.cat([self.x, self.time.view(-1, 1) / self.max_time], dim=1)
                    x_recon, y_recon, pred_risk, z, mu, log_var = self.model(xt, self.y)
                    pred, _ = self.predict(xt)
                    # total losses
                    l_recon = F.mse_loss(y_recon, self.y) + F.mse_loss(x_recon, self.x)
                    l_nll = self.neg_log_likelihood(pred_risk, self.time, self.label)
                    kld = self.kld_loss(mu, log_var, z)
                    loss_total = l_recon + kld + l_nll

                    epoch_bar.set_postfix({
                        # 'epoch': epoch,
                        'Total Loss': '{:.4f}'.format(loss_total.item()),
                        'Recon loss': '{:.4f}'.format(l_recon.item()),
                        'NLL loss': '{:.4f}'.format(l_nll.item()),
                        'KLD': '{:.4f}'.format(kld.item()),
                        'cluster0': len(np.where(pred == 0)[0]),
                        'cluster1': len(np.where(pred == 1)[0])
                        # 'cluster3': len(np.where(pred == 2)[0])
                    })
            if save_model:
                torch.save({'model': self.model.state_dict()}, self.save_path + '/trained_model.pth')
        self.fitted = True
        print('training competed....')

    def evaluate(self):
        if not self.fitted:
            print('Model Not trained. Please train the model first before evaluation.')
        else:
            with torch.no_grad():
                self.model.eval()
                xt = torch.cat([self.x, self.time.view(-1, 1) / self.max_time], dim=1)
                pred_pattern, z = self.predict(xt)
                z = z.detach().cpu().numpy()

                fig = plt.figure(figsize=(8, 6))

                ax2 = fig.add_subplot(1, 1, 1)
                plt.scatter(z[:, 0], z[:, 1], c=pred_pattern, marker='o',
                            edgecolor='none', cmap=discrete_cmap(self.nClusters, 'jet'))
                plt.colorbar(ticks=range(self.nClusters))
                ax2.set_xlim([np.min(z[:, 0]) - 0.1, np.max(z[:, 0]) + 0.2])
                ax2.set_ylim([np.min(z[:, 1]) - 0.2, np.max(z[:, 1]) + 0.2])
                # ax2.set_xlim([- 0.1, 0.3])
                # ax2.set_ylim([0, 0.3])
                plt.grid(True)

                result_path = self.save_path + '/result'
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                plt.savefig('result/PMLR_map_epoch_{}2.png'.format(self.epochs))
                plt.show()

    def to_device(self, x, y, time, label):
        return x.to(device), y.to(device), time.to(device), label.to(device)

    def predict(self, xt):
        return self.model.predict(xt)

    def get_z(self):
        xt = torch.cat([self.x, self.time.view(-1, 1) / self.max_time], dim=1)
        mu, log_var = self.model.encoder(xt)
        z = self.model.sampling(mu, log_var)
        return z.detach().cpu().numpy()

    def pred_time(self, x):
        n_subj = x.shape[0]
        x = torch.from_numpy(x).float().to(device)
        zs = np.zeros([n_subj, self.z_dim, self.num_times])
        z_probs = torch.zeros([n_subj, self.nClusters, self.num_times])
        for t in range(self.num_times):
            t_tensor = torch.from_numpy(np.repeat(t / self.max_time, n_subj).reshape([-1, 1])).float().to(device)
            xt = torch.cat([x, t_tensor], dim=1)
            _1, _2, pred_risk, z, _3, _4 = self.model(xt)
            z_prob = self.model.pred_prob(xt)
            z_probs[:, :, t] = z_prob



    def gaussian_pdfs_log(self, x, mus, log_var):
        G = []
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_var[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_var):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_var + (x - mu).pow(2) / torch.exp(log_var), 1))

    def kld_loss(self, mu, log_var, z):
        return self.model.kld_loss(mu, log_var, z)

    def get_pred_cluster(self):
        if not self.fitted:
            print('Model Not trained. Please train the model first before prediction.')
        else:
            with torch.no_grad():
                xt = torch.cat([self.x, self.time.view(-1, 1) / self.max_time], dim=1)
                pred_pattern, _ = self.predict(xt)
                return pred_pattern

    # calculate log partial likelihood loss for cause-specific sub-network
    def neg_log_likelihood(self, pred_risk, true_time, true_label):
        I_1 = torch.sign(true_label)
        mask = utils.get_bh_mask(true_time, true_label, self.num_times)
        mask_tensor = torch.from_numpy(mask).float().to(device)
        # for uncensored: log P(T=t,K=k|x)
        loss1 = I_1 * torch.log(torch.sum(mask_tensor * pred_risk, dim=1) + eps)
        # for censored: log \sum P(T>t|x)
        loss2 = (1. - I_1) * torch.log(torch.sum(mask_tensor * pred_risk, dim=1) + eps)
        likelihood_loss = -torch.mean(loss1 + loss2)
        return likelihood_loss


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
