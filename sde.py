import torch
import torch.nn as nn
from absl import logging
import numpy as np
import math
from tqdm import tqdm
from pytorch_msssim import ssim


def get_sde(name, **kwargs):
    if name == 'vpsde':
        return VPSDE(**kwargs)
    elif name == 'vpsde_cosine':
        return VPSDECosine(**kwargs)
    else:
        raise NotImplementedError


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1, weight=None):  # mean of square
    if weight != None:
        return (weight*a).pow(2).flatten(start_dim=start_dim).mean(dim=-1)
    else:
        return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)

def mos_layer_wise(preds, ltes, noise):
    loss = 0.0
    layer = len(preds)
    w = (layer + 1) * layer / 2
    weights = torch.zeros(preds[0].shape[0], layer).to(preds[0].device)
    for i, pred in enumerate(preds):
        u_true = 1 - torch.tanh(torch.abs(pred - noise))
        # weights[:, i] = mos(ltes[i] - u_true).view(u_true.shape[0], -1).mean(dim=-1)
        weights[:, i] = ltes[i].view(u_true.shape[0], -1).mean()
    weights = 1 / weights.sum(dim=-1, keepdim=True) * weights
    for i, pred in enumerate(preds):
        # interval_ratio = (torch.abs(pred - noise) / torch.abs(noise)).view(
        #                                 pred.shape[0], -1).mean(1).unsqueeze(1).clone().detach().float()
        # interval_ratio = interval_ratio / interval_ratio.sum(0)
        # interval_ratio = (interval_ratio < 0.5).clone().detach().half()
        # labels = torch.concat([interval_ratio, 1-interval_ratio], dim=1).detach()
        # print(ltes[i].shape)
        # print(mos(noise - pred).shape)
        loss +=   weights[:, i] * mos(noise - pred)
        # loss += 1 / (torch.abs((1 - weights[:, i]) - 0.1) + 0.001)  * mos(noise - pred)
    # print("layer-wise loss:", weights)
    return loss

def mos_lte(preds, ltes, noise):
    loss = 0.0
    layer = len(preds)
    w = (layer + 1) * layer / 2
    for i, pred in enumerate(preds):
        u_true = 1 - torch.tanh(torch.abs(pred - noise)).detach()
        loss += 1 / len(preds) * mos(ltes[i] - u_true)
        # interval_ratio = (torch.abs(pred - noise) / noise).view(
        #                                 pred.shape[0], -1).mean(1).unsqueeze(1)
        # interval_ratio = (interval_ratio < 0.1).clone().detach().half()
        # labels = torch.concat([interval_ratio, 1-interval_ratio], dim=1).detach()
        # pos_num = torch.count_nonzero(interval_ratio.view(-1))
        # neg_num = interval_ratio.shape[0] - pos_num

        # pos_weight = 1 - (pos_num / interval_ratio.shape[0])
        # neg_weight = 1 - (neg_num / interval_ratio.shape[0])
        # weight = torch.concat([pos_weight.view(-1), neg_weight.view(-1)], dim=0)
        # loss += torch.nn.functional.cross_entropy(ltes[i], labels, weight=weight)
    # print("lte loss:", ltes[i])
    return loss

def mos_local_lte(preds, ltes, global_ltes):
    loss = 0.0
    layer = len(preds)
    w = (layer + 1) * layer / 2
    for i, pred in enumerate(preds):
        if i == len(preds) - 1:
            break
        u_true = 1 - torch.tanh(global_ltes[i].view(global_ltes[i].shape[0], -1).mean() * torch.abs(pred - preds[-1])).detach()
        loss += 1 / len(preds) * mos(ltes[i] - u_true)
        # interval_ratio = (torch.abs(pred - noise) / noise).view(
        #                                 pred.shape[0], -1).mean(1).unsqueeze(1)
        # interval_ratio = (interval_ratio < 0.1).clone().detach().half()
        # labels = torch.concat([interval_ratio, 1-interval_ratio], dim=1).detach()
        # pos_num = torch.count_nonzero(interval_ratio.view(-1))
        # neg_num = interval_ratio.shape[0] - pos_num

        # pos_weight = 1 - (pos_num / interval_ratio.shape[0])
        # neg_weight = 1 - (neg_num / interval_ratio.shape[0])
        # weight = torch.concat([pos_weight.view(-1), neg_weight.view(-1)], dim=0)
        # loss += torch.nn.functional.cross_entropy(ltes[i], labels, weight=weight)
    # print("lte loss:", ltes[i])
    return loss


def mos_normal_layer_wise(preds, ltes, noise):
    loss = 0.0
    layer = len(preds)
    w = (layer + 1) * layer / 2
    # weights = torch.zeros(preds[0].shape[0], layer).to(preds[0].device)
    # for i, pred in enumerate(preds):
    #     u_true = 1 - torch.tanh(torch.abs(pred - noise))
    #     weights[:, i] = mos(ltes[i] - u_true).view(u_true.shape[0], -1).mean(dim=-1)
    # # print(weights.sum(dim=-1).shape)
    # # print(weights.shape)
    # weights = 1 / weights.sum(dim=-1, keepdim=True) * weights
    for i, pred in enumerate(preds):
        loss += 1 / len(preds) * mos(noise - pred)
    return loss

def mos_calm_layer_wise(preds, ltes, noise):
    loss = 0.0
    layer = len(preds)
    w = (layer + 1) * layer / 2
    # weights = torch.zeros(preds[0].shape[0], layer).to(preds[0].device)
    # for i, pred in enumerate(preds):
    #     u_true = 1 - torch.tanh(torch.abs(pred - noise))
    #     weights[:, i] = mos(ltes[i] - u_true).view(u_true.shape[0], -1).mean(dim=-1)
    # # print(weights.sum(dim=-1).shape)
    # # print(weights.shape)
    # weights = 1 / weights.sum(dim=-1, keepdim=True) * weights
    for i, pred in enumerate(preds):
        loss += i / len(preds) * mos(noise - pred)
    return loss


def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)


class SDE(object):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """
    def drift(self, x, t):
        raise NotImplementedError

    def diffusion(self, t):
        raise NotImplementedError

    def cum_beta(self, t):  # the variance of xt|x0
        raise NotImplementedError

    def cum_alpha(self, t):
        raise NotImplementedError

    def snr(self, t):  # signal noise ratio
        raise NotImplementedError

    def nsr(self, t):  # noise signal ratio
        raise NotImplementedError

    def marginal_prob(self, x0, t):  # the mean and std of q(xt|x0)
        alpha = self.cum_alpha(t)
        beta = self.cum_beta(t)
        mean = stp(alpha ** 0.5, x0)  # E[xt|x0]
        std = beta ** 0.5  # Cov[xt|x0] ** 0.5
        return mean, std

    def sample(self, x0, t_init=0):  # sample from q(xn|x0), where n is uniform
        t = torch.rand(x0.shape[0], device=x0.device) * (1. - t_init) + t_init
        mean, std = self.marginal_prob(x0, t)
        eps = torch.randn_like(x0)
        xt = mean + stp(std, eps)
        return t, eps, xt


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20):
        # 0 <= t <= 1
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def drift(self, x, t):
        return -0.5 * stp(self.squared_diffusion(t), x)

    def diffusion(self, t):
        return self.squared_diffusion(t) ** 0.5

    def squared_diffusion(self, t):  # beta(t)
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def squared_diffusion_integral(self, s, t):  # \int_s^t beta(tau) d tau
        return self.beta_0 * (t - s) + (self.beta_1 - self.beta_0) * (t ** 2 - s ** 2) * 0.5

    def skip_beta(self, s, t):  # beta_{t|s}, Cov[xt|xs]=beta_{t|s} I
        return 1. - self.skip_alpha(s, t)

    def skip_alpha(self, s, t):  # alpha_{t|s}, E[xt|xs]=alpha_{t|s}**0.5 xs
        x = -self.squared_diffusion_integral(s, t)
        return x.exp()

    def cum_beta(self, t):
        return self.skip_beta(0, t)

    def cum_alpha(self, t):
        return self.skip_alpha(0, t)

    def nsr(self, t):
        return self.squared_diffusion_integral(0, t).expm1()

    def snr(self, t):
        return 1. / self.nsr(t)

    def __str__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'

    def __repr__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'


class VPSDECosine(SDE):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """
    def __init__(self, s=0.008):
        self.s = s
        self.F = lambda t: torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        self.F0 = math.cos(s / (1 + s) * math.pi / 2) ** 2

    def drift(self, x, t):
        ft = - torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi / 2
        return stp(ft, x)

    def diffusion(self, t):
        return (torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi) ** 0.5

    def cum_beta(self, t):  # the variance of xt|x0
        return 1 - self.cum_alpha(t)

    def cum_alpha(self, t):
        return self.F(t) / self.F0

    def snr(self, t):  # signal noise ratio
        Ft = self.F(t)
        return Ft / (self.F0 - Ft)

    def nsr(self, t):  # noise signal ratio
        Ft = self.F(t)
        return self.F0 / Ft - 1

    def __str__(self):
        return 'vpsde_cosine'

    def __repr__(self):
        return 'vpsde_cosine'


class ScoreModel(object):
    r"""
        The forward process is q(x_[0,T])
    """

    def __init__(self, nnet: nn.Module, pred: str, sde: SDE, T=1):
        assert T == 1
        self.nnet = nnet
        self.pred = pred
        self.sde = sde
        self.T = T
        print(f'ScoreModel with pred={pred}, sde={sde}, T={T}')

    def predict(self, xt, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.to(xt.device)
        if t.dim() == 0:
            t = duplicate(t, xt.size(0))
        return self.nnet(xt, t * 999, **kwargs)  # follow SDE

    def noise_pred(self, xt, t, **kwargs):
        kwargs["nsr"] = self.sde.nsr(t)
        kwargs["cum_alpha"] = self.sde.cum_alpha(t).rsqrt()
        pred, u_pred, quality, lte, i = self.predict(xt, t, **kwargs)
        if self.pred == 'noise_pred':
            noise_pred = pred
            x_0 = stp(self.sde.cum_alpha(t).rsqrt(), xt) - stp(self.sde.nsr(t).sqrt(), pred)
        elif self.pred == 'x0_pred':
            noise_pred = - stp(self.sde.snr(t).sqrt(), pred) + stp(self.sde.cum_beta(t).rsqrt(), xt)
        else:
            raise NotImplementedError
        return noise_pred, u_pred, x_0, quality, lte, i

    def x0_pred(self, xt, t, **kwargs):
        pred = self.predict(xt, t, **kwargs)
        if self.pred == 'noise_pred':
            x0_pred = stp(self.sde.cum_alpha(t).rsqrt(), xt) - stp(self.sde.nsr(t).sqrt(), pred)
        elif self.pred == 'x0_pred':
            x0_pred = pred
        else:
            raise NotImplementedError
        return x0_pred

    def score(self, xt, t, **kwargs):
        cum_beta = self.sde.cum_beta(t)
        noise_pred, inner_state, _, _, _, i = self.noise_pred(xt, t, **kwargs)
        return stp(-cum_beta.rsqrt(), noise_pred), i, inner_state


class ReverseSDE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt + g(t) dw
    """
    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, t, **kwargs):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score, i, inner_state = self.score_model.score(x, t, **kwargs)
        return drift - stp(diffusion ** 2, score), inner_state

    def diffusion(self, t):
        return self.sde.diffusion(t)


class ODE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt
    """

    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, t, **kwargs):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score, i, _, = self.score_model.score(x, t, **kwargs)
        return drift - 0.5 * stp(diffusion ** 2, score), i 

    def diffusion(self, t):
        return 0


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


@ torch.no_grad()
def euler_maruyama(rsde, x_init, sample_steps, eps=1e-3, T=1, trace=None, verbose=False, **kwargs):
    r"""
    The Euler Maruyama sampler for reverse SDE / ODE
    See `Score-Based Generative Modeling through Stochastic Differential Equations`
    """
    assert isinstance(rsde, ReverseSDE) or isinstance(rsde, ODE)
    print(f"euler_maruyama with sample_steps={sample_steps}")
    timesteps = np.append(0., np.linspace(eps, T, sample_steps))
    timesteps = torch.tensor(timesteps).to(x_init)
    x = x_init
    if trace is not None:
        trace.append(x) 
    
    import matplotlib.pyplot as plt
    loss =  []
    l = torch.tensor([0.0], device=x.device)
    j = 0
    error = np.zeros(20)
    for s, t in tqdm(list(zip(timesteps, timesteps[1:]))[::-1], disable=not verbose, desc='euler_maruyama'):
        if j <= 200:
            kwargs["layer"] = 1
        elif 200 < j <= 400 :
            kwargs["layer"] = 3
        elif 400 < j < 600 :
            kwargs["layer"] = 6
        else:
            kwargs["layer"] = 13
        
        j += 1
        drift, inner_states  = rsde.drift(x, t, **kwargs)
        # print(inner_states)
        # for i, state in enumerate(inner_states[:-1]):
        #     error[i] += mos(state-inner_states[-1])
        # kwargs["is_train"] = False
        # drift2, i  = rsde.drift(x, t, **kwargs)
        # loss.append(mos(drift-drift2).cpu().numpy()[0])
        
        # l = torch.cat((l, i))
        diffusion = rsde.diffusion(t)
        dt = s - t
        mean = x + drift * dt
        sigma = diffusion * (-dt).sqrt()
        x = mean + stp(sigma, torch.randn_like(x)) if s != 0 else mean
        if trace is not None:
            trace.append(x)
        statistics = dict(s=s, t=t, sigma=sigma.item())
        logging.debug(dct2str(statistics))
    print(error/1000)
    # s = np.arange(0,1000)
    # print(loss)
    # plt.plot(s, loss)
    # plt.plot(s, layer)
    # plt.show()
    # plt.savefig("6.png")
    return x

mse = []
def LSimple(score_model: ScoreModel, x0, _step, pred='noise_pred', **kwargs):
    t, noise, xt = score_model.sde.sample(x0)
    if pred == 'noise_pred':
        # for i in range(1000):
        #     t, noise, xt = score_model.sde.sample(x0, t_init=i)
            # t = torch.tensor(i)
        noise_pred, inner_pred, x_0, local_lte, lte, _ = score_model.noise_pred(xt, t, **kwargs)
        ssim_value = ssim(x_0, x0, data_range=1, size_average=False)
        # print(t.item(), "   ",  ssim_value.item())
        #     global mse
        #     mse.append(str(mos(noise - noise_pred).item())) 
        # with open("mse.txt", "w") as f:
        #     # for m in mse:
        #     f.writelines(mse)
        # if _step % 2 == 0:
        #     return  mos(noise - noise_pred) + mos_lte(inner_pred, lte, noise)
        # else:
        #     return mos(noise - noise_pred) + mos_layer_wise(inner_pred, lte, noise)
        # return mos_local_lte(inner_pred, local_lte, lte)
        return ssim_value.detach() * (mos(noise - noise_pred) + mos_lte(inner_pred, lte, noise) + mos_layer_wise(inner_pred, lte, noise))
    elif pred == 'x0_pred':
        x0_pred = score_model.x0_pred(xt, t, **kwargs)
        return mos(x0 - x0_pred)
    else:
        raise NotImplementedError(pred)
