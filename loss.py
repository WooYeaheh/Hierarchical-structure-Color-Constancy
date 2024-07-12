import math
import torch
import numpy as np
import torch.nn as nn

class Angular_loss(torch.nn.Module):
    def __init__(self):
        super(Angular_loss, self).__init__()
        self.threshold = 0.999999

    def forward(self, pred, target):
        return torch.mean(self.angulor_loss(pred, target))

    def angulor_loss(self, pred, target):
        pred = nn.functional.normalize(pred, dim=1)
        target = nn.functional.normalize(target, dim=1)

        arccos_num = torch.sum(pred * target, dim=1)
        arccos_num = torch.clamp(arccos_num, -self.threshold, self.threshold)
        angle = torch.acos(arccos_num) * (180 / math.pi)
        return angle

    def batch_angular_loss(self, pred, target):
        return self.angulor_loss(pred, target)

def angular_loss(pred, gt):
    pred.cuda()
    gt.cuda()
    a = (pred[:, 0] * gt[:, 0] + pred[:, 1] * gt[:, 1] + pred[:, 2] * gt[:, 2])
    b = pow(torch.abs(pred[:, 0] * pred[:, 0] + pred[:, 1] * pred[:, 1] + pred[:, 2] * pred[:, 2] + 1e-06), 1 / 2)
    c = pow(torch.abs(gt[:, 0] * gt[:, 0] + gt[:, 1] * gt[:, 1] + gt[:, 2] * gt[:, 2] + 1e-06), 1 / 2)

    acos = torch.acos(a / (b * c + 1e-05))
    output = torch.mean(acos)
    output = output * 180 / math.pi

    return output

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag (sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class SimCLR_3_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 3 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, 2*batch_size + i] = 0
            mask[2*batch_size + i, i] = 0
        for i in range(2*batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j, z_k):
        N = 3 * self.batch_size
        z = torch.cat((z_i, z_j, z_k), dim=0) # 27,3
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_1 = torch.diag (sim, self.batch_size)
        sim_2 = torch.diag(sim, -self.batch_size)
        sim_3 = torch.diag(sim, self.batch_size*2)
        sim_4 = torch.diag(sim, -self.batch_size*2)

        positive_samples = torch.cat((sim_1,sim_2,sim_3,sim_4), dim=0).reshape(N, 2)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()
        logits_1 = torch.cat((positive_samples[:,0].unsqueeze(1), negative_samples), dim=1)
        logits_2 = torch.cat((positive_samples[:,1].unsqueeze(1), negative_samples), dim=1)

        loss_1 = self.criterion(logits_1, labels)
        loss_2 = self.criterion(logits_2, labels)
        loss = loss_1 + loss_2
        loss /= 2*N

        return loss