import torch.nn as nn
import torch.optim

from .inn_architecture import build_inn
from .cond_resnet import ResNet18, build_entry_flow

class CINN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.inn = build_inn(args)
        self.cond_net = ResNet18(args)
        self.nice_entry = build_entry_flow(args)

        self.trainable_params = (list(self.nice_entry.parameters())
                               + list(self.inn.parameters())
                               + list(self.cond_net.parameters()))

        optim = args['training']['optimizer']
        lr    = eval(args['training']['lr'])

        if optim == 'ADAM':
            self.optimizer = torch.optim.Adam(self.trainable_params, lr=lr)
        elif optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.trainable_params, lr=lr, momentum=0.5)

        else:
            raise ValueError('ADAM or SGD')

    def forward(self, x, y):
        return self.nll(x, y)

    def nll(self, x, y):
        conditions = self.cond_net(y)
        t0 = self.nice_entry(y)

        x = x - t0
        z, jac = self.inn(x, c=conditions)

        ndim = z.shape[1] * z.shape[2] * z.shape[3]
        neg_log_pz = 0.5 * torch.mean(z**2)

        return neg_log_pz - torch.mean(jac) / ndim

    def z(self, x, y):
        conditions = self.cond_net(y)
        t0 = self.nice_entry(y)

        x = x - t0
        z, jac = self.inn(x, c=conditions)

        return z

    def generate(self, z, y):
        conditions = self.cond_net(y)
        t0 = self.nice_entry(y)

        x, j = self.inn(z, c=conditions, rev=True)
        x = x + t0

        return x

    def save(self, fname, save_optim=False):
        data_dict = {'inn': self.inn.state_dict(),
                     'cond_net': self.cond_net.state_dict(),
                     'entry_flow': self.nice_entry.state_dict()}
        if save_optim:
            data_dict['optim'] = self.optimizer.state_dict()
        torch.save(data_dict, fname)

    def load(self, fname):
        data_dict = torch.load(fname)

        self.inn.load_state_dict(data_dict['inn'])
        self.cond_net.load_state_dict(data_dict['cond_net'])
        self.nice_entry.load_state_dict(data_dict['entry_flow'])

        try:
            self.optimizer.load_state_dict(data_dict['optim'])
        except KeyError:
            pass
