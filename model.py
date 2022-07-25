import numpy as np
import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class Encoder(nn.Module):
    def __init__(self, input_dim, h_dim1, h_dim2, z_dim=2):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim1),
            nn.LeakyReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.LeakyReLU()
        )

        self.mu = nn.Linear(h_dim2, z_dim)
        self.log_var = nn.Linear(h_dim2, z_dim)
        nn.init.trunc_normal_(self.log_var.weight, std=0.01)

    def forward(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)


class Decoder(nn.Module):
    def __init__(self, out_dim, h_dim1, h_dim2, h_dim3):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(h_dim3, h_dim2),
            nn.LeakyReLU(),
            nn.Linear(h_dim2, h_dim1),
            nn.LeakyReLU(),
            nn.Linear(h_dim1, out_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class CS_layer(nn.Module):
    def __init__(self, h_dim3, cs_dim1, cs_dim2, out_dim):
        super(CS_layer, self).__init__()
        self.cs_layer = nn.Sequential(
            nn.Linear(h_dim3, cs_dim1),
            nn.LeakyReLU(),
            nn.Linear(cs_dim1, cs_dim2),
            nn.LeakyReLU(),
            nn.Linear(cs_dim2, out_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, hz):
        return self.cs_layer(hz)


class TransLayer(nn.Module):
    def __init__(self, x_dim, z_dim, tr_dim1, tr_dim2, out_dim):
        super(TransLayer, self).__init__()
        self.trans_layer_x = nn.Sequential(
            nn.Linear(x_dim, tr_dim1),
            nn.LeakyReLU(),
            nn.Linear(tr_dim1, tr_dim2),
            nn.LeakyReLU(),
            nn.Linear(tr_dim2, out_dim)
        )
        self.trans_layer_z = nn.Linear(z_dim, out_dim)

    def forward(self, y, z):
        hy = self.trans_layer_x(y)
        hz = self.trans_layer_z(z)
        return hz, hy * hz


class Staging_VAE(nn.Module):
    def __init__(self, input_dims):
        super(Staging_VAE, self).__init__()

        self.input_dim = input_dims['input_dim']
        self.h_dim1 = input_dims['h_dim1']
        self.h_dim2 = input_dims['h_dim2']
        self.h_dim3 = input_dims['h_dim3']

        self.x_dim = input_dims['x_dim']
        self.z_dim = input_dims['z_dim']

        self.nClusters = input_dims['nClusters']

        self.trans_dim1 = input_dims['trans_dim1']
        self.trans_dim2 = input_dims['trans_dim2']

        self.cs_dim1 = input_dims['cs_dim1']
        self.cs_dim2 = input_dims['cs_dim2']

        self.num_times = input_dims['num_times']

        # encoder & decoder
        self.encoder = Encoder(input_dim=self.input_dim, h_dim1=self.h_dim1, h_dim2=self.h_dim2, z_dim=self.z_dim)
        self.decoder_y = Decoder(out_dim=self.x_dim, h_dim1=self.h_dim1, h_dim2=self.h_dim2, h_dim3=self.z_dim)
        self.decoder = Decoder(out_dim=self.x_dim, h_dim1=self.h_dim1, h_dim2=self.h_dim2, h_dim3=self.h_dim3)

        # parameters for GMM:
        self.pi = nn.Parameter(torch.FloatTensor(self.nClusters, ).fill_(1) / self.nClusters, requires_grad=True)
        self.mu_k = nn.Parameter(torch.FloatTensor(self.nClusters, self.z_dim).fill_(0), requires_grad=True)
        self.log_var_k = nn.Parameter(torch.FloatTensor(self.nClusters, self.z_dim).fill_(0), requires_grad=True)

        # layers for transform x and z
        self.trans_layer = TransLayer(self.x_dim, self.z_dim, self.trans_dim1, self.trans_dim2, self.h_dim3)

        # self.bn_layer = nn.BatchNorm1d(self.h_dim3, eps=1e-8)
        self.cs_layer = CS_layer(self.h_dim3, self.cs_dim1, self.cs_dim2, self.num_times)

        self.param_init()

    def param_init(self):
        self.encoder.apply(weights_init_normal)
        self.decoder.apply(weights_init_normal)
        self.decoder_y.apply(weights_init_normal)
        self.cs_layer.apply(weights_init_normal)
        self.trans_layer.apply(weights_init_normal)

    def transform(self, y, z):
        return self.trans_layer(y, z)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return mu + std * eps

    # survival analysis cause-specific sub-network
    def cs_out(self, hz):
        out = self.cs_layer(hz)
        return out

    def forward(self, xt, y):
        mu, log_var = self.encoder(xt)
        z = self.sampling(mu, log_var)
        y_recon = self.decoder_y(z)
        hz, h = self.transform(y, z)
        x_recon = self.decoder(h)
        pred_risk = self.cs_out(hz)
        return x_recon, y_recon, pred_risk, z, mu, log_var

    def cal_loss(self, xt, y):
        mu, log_var = self.encoder(xt)
        z = self.sampling(mu, log_var)
        y_recon = self.decoder(z)
        hc, h = self.transform(y, z)
        x_recon = self.decoder(h)
        pred_risk = self.cs_out(hc)
        return x_recon, y_recon, pred_risk, z, mu, log_var

    def predict(self, xt):
        mu, log_var = self.encoder(xt)
        z = self.sampling(mu, log_var)
        pi = self.pi
        mu_k = self.mu_k
        log_var_k = self.log_var_k
        pi = torch.exp(pi) / torch.sum(torch.exp(pi))
        yita_k = torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_k, log_var_k)
        # = torch.exp(tmp1)

        yita = yita_k.detach().cpu().numpy()
        return np.argmax(yita, axis=1), z

    def pred_prob(self, xt):
        mu, log_var = self.encoder(xt)
        z = self.sampling(mu, log_var)
        pi = self.pi
        mu_k = self.mu_k
        log_var_k = self.log_var_k
        pi = torch.exp(pi) / torch.sum(torch.exp(pi))
        yita_k = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_k, log_var_k))
        yita_k = yita_k / torch.sum(yita_k, dim=1)

        yita = yita_k.detach().cpu().numpy()
        return yita

    def kld_loss(self, mu, log_var, z):
        # pi -> [nCluster]
        pi = self.pi
        pi = torch.exp(pi) / torch.sum(torch.exp(pi))
        # log_var_k -> [nCluster, 2]
        log_var_k = self.log_var_k
        # mu_k -> [nCluster, 2]
        mu_k = self.mu_k
        # z -> [N, 2]
        z = self.sampling(mu, log_var)
        # yita_k -> [N, nCluster] = [1, nCluster] + [N, nCluster]
        yita_k = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_k, log_var_k)) + 1e-8
        yita_k = yita_k / (yita_k.sum(1).view(-1, 1))
        # L1 -> [N, nCluster, 2]  [1, nCluster, 2] + [N, 1, 2] - [1, nCluster, 2]
        L1 = log_var_k.unsqueeze(0) + torch.exp(log_var.unsqueeze(1) - log_var_k.unsqueeze(0))
        # L2 -> [N, nCluster, 2] ([N, 1, 2] - [1, nCluster, 2]) / [1, nCluster, 2]
        L2 = (mu.unsqueeze(1) - mu_k.unsqueeze(0)).pow(2) / torch.exp(log_var_k.unsqueeze(0))
        # L3 ->
        L3 = torch.mean(torch.sum(yita_k * torch.log(pi.unsqueeze(0) / yita_k), 1))
        L4 = 0.5 * torch.mean(torch.sum(1 + log_var, 1))
        kld = 0.5 * torch.mean(torch.sum(yita_k * torch.sum(L1 + L2, dim=2), dim=1)) - (L3 + L4)
        return kld

    def gaussian_pdfs_log(self, x, mu, log_var):
        G = []
        # x -> [N, 2], mu -> [nCluster, 2], log_var -> [nCluster, 2]
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x, mu[c:c + 1, :], log_var[c:c + 1, :]).view(-1, 1))
        # return [N, nCluster]
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_var):
        # x -> [N, 2], mu -> [2], log_var -> [2]
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_var + (x - mu).pow(2) / torch.exp(log_var), 1))
