import torch.nn as nn
import torch.optim

from .inn_architecture import build_inn
from .cond_resnet import ResNet18

class CINN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.inn = build_inn(args)
        self.cond_net = ResNet18(args)

        self.trainable_params = list(self.inn.parameters()) + list(self.cond_net.parameters())

        optim = eval(args['training']['optimizer'])
        lr    = eval(args['training']['lr'])

        self.optimizer = optim(self.trainable_params, lr=lr)

    def nll(self, x, y):
        conditions = self.cond_net(y)
        z, jac = self.inn(x, c=conditions)

        ndim = z.shape[1] * z.shape[2] * z.shape[3]
        neg_log_pz = 0.5 * torch.mean(z**2)

        return neg_log_pz - torch.mean(jac) / ndim

    def z(self, x, y):
        conditions = self.cond_net(y)
        z, jac = self.inn(x, c=conditions)

        return z

    def save(self, fname, save_optim=False):
        data_dict = {'inn': self.inn.state_dict(),
                     'cond_net': self.cond_net.state_dict()}
        if save_optim:
            data_dict['optim'] = self.optimizer.state_dict()
        torch.save(data_dict, fname)

    def load(self, fname):
        data_dict = torch.load(fname)

        self.inn.load_state_dict(data_dict['inn'])
        self.cond_net.load_state_dict(data_dict['cond_net'])

        try:
            self.optimizer.load_state_dict(data_dict['optim'])
        except KeyError:
            pass
