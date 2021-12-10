import torch
from torch.nn import functional as F
import numpy as np

def d_lsgan_loss(discriminator, trues, fakes, labels, alpha):
    d_trues = discriminator.forward(trues, labels, alpha)
    d_fakes = discriminator.forward(fakes, labels, alpha)

    loss = F.mse_loss(d_trues, torch.ones_like(d_trues)) + F.mse_loss(d_fakes, torch.zeros_like(d_fakes))
    loss /= 2
    return loss


def g_lsgan_loss(discriminator, fakes, labels, alpha):
    d_fakes = discriminator.forward(fakes, labels, alpha)
    loss = F.mse_loss(d_fakes, torch.ones_like(d_fakes)) / 2
    return loss


def d_wgan_loss(discriminator, d_trues, d_fakes, before,  trues, fakes,  teachers, alpha, phase, useGradient = True):
    epsilon_drift = 1e-3
    lambda_gp = 1 # 10

    batch_size = fakes.size()[0]

    loss_wd =  torch.nn.LeakyReLU(0.002)(1- d_trues).mean() + torch.nn.LeakyReLU(0.002)(1+d_fakes).mean()

    # gradient penalty
    loss_gp = 0
    if(phase == "train" and useGradient):
        epsilon = torch.rand(batch_size, 1, 1, 1, dtype=fakes.dtype, device=fakes.device)
        intpl = epsilon * fakes + (1 - epsilon) * trues
        intpl.requires_grad_()
        f = discriminator.forward(before, intpl, teachers,  alpha)
        grad = torch.autograd.grad(f.sum(), intpl, create_graph=True)[0]
        del intpl, epsilon, f
        grad_norm = grad.view(batch_size, -1).norm(dim=1)
        loss_gp = lambda_gp * ((grad_norm - 1) ** 2).mean()

    # drift
    loss_drift = epsilon_drift * (d_trues ** 2).mean()

    loss = loss_wd + loss_gp + loss_drift

    wd = loss_wd.item()

    return loss, wd

def d_wgan_loss2(discriminator,  before,  trues, fakes,  teachers, alpha, phase, useGradient = True, useBefore = True):
    epsilon_drift = 1e-3
    lambda_gp = 1e-2 # 10
    loss_list = []

    batch_size = fakes.size()[0]

    if(useBefore):
        d_trues = discriminator(before, trues, teachers, alpha)
        d_fakes = discriminator(before, fakes, teachers, alpha)
    else:
        d_trues = discriminator(trues, teachers, alpha)
        d_fakes = discriminator(fakes, teachers, alpha)
    loss_wd =  (torch.nn.LeakyReLU(0.002)(1- d_trues)).mean() +\
         (torch.nn.LeakyReLU(0.002)(1 + d_fakes)).mean()
    TCorrectN = (d_trues > 0).sum().item()
    FCorrectN =  (d_fakes <= 0).sum().item()
    correctN = TCorrectN + FCorrectN
    loss_list.append(loss_wd.item())

    # drift
    # loss_wd += epsilon_drift * (d_trues ** 2).mean()
    # loss_wd += epsilon_drift * (d_fakes ** 2).mean() # 不安定なため追加

    del d_fakes, d_trues
    loss_list.append(loss_wd.item() - loss_list[0])

    # gradient penalty
    loss_gp = 0
    if(phase == "train" and useGradient):
        epsilon = torch.rand(batch_size, 1, 1, 1, dtype=fakes.dtype, device=fakes.device)
        intpl = epsilon * fakes + (1 - epsilon) * trues
        intpl.requires_grad_()
        if(useBefore):
            f = discriminator.forward(before, intpl, teachers,  alpha)
        else:
            f = discriminator.forward(intpl, teachers,  alpha)
        grad = torch.autograd.grad(f.sum(), intpl, create_graph=True)[0]
        del intpl, epsilon, f
        grad_norm = grad.view(batch_size, -1).norm(dim=1)
        loss_gp = lambda_gp * ((grad_norm - 1) ** 2).mean()
        loss_list.append(loss_gp.item())
    else:
        loss_list.append(loss_gp)


    loss = loss_wd  + loss_gp

    
    # wd = loss_wd.item()

    return loss, correctN, np.array(loss_list), TCorrectN, FCorrectN



def g_wgan_loss(d_fakes):
    loss = torch.nn.LeakyReLU(0.002)(1 -d_fakes).mean()
    return loss


def d_logistic_loss(discriminator, trues, fakes, labels, alpha, r1gamma=10):
    d_fakes = discriminator.forward(fakes, labels, alpha)
    trues.requires_grad_()
    d_trues = discriminator.forward(trues, labels, alpha)
    loss = F.softplus(d_fakes).mean() + F.softplus(-d_trues).mean()

    if r1gamma > 0:
        grad = torch.autograd.grad(d_trues.sum(), trues, create_graph=True)[0]
        loss += r1gamma/2 * (grad**2).sum(dim=(1, 2, 3)).mean()

    return loss


def g_logistic_loss(discriminator, fakes, labels, alpha):
    d_fakes = discriminator.forward(fakes, labels, alpha)
    return F.softplus(-d_fakes).mean()

