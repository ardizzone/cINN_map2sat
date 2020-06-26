import torch.nn as nn
import torch
from all_in_one_block import AllInOneBlock
from FrEIA.modules import HaarDownsampling

class ReversibleSequential(nn.Module):

    def __init__(self, *dims):
        super().__init__()

        self.shapes = [tuple(dims)]
        self.conditions = []
        self.module_list = nn.ModuleList()

    def append(self, module_class, cond=None, cond_shape=None, **kwargs):

        dims_in = [self.shapes[-1]]
        self.conditions.append(cond)

        if cond is not None:
            kwargs['dims_c'] = [cond_shape]

        module = module_class(dims_in, **kwargs)
        self.module_list.append(module)
        ouput_dims = module.output_dims(dims_in)
        assert len(ouput_dims) == 1, "Module has more than one output"
        self.shapes.append(ouput_dims[0])


    def forward(self, x, c=None, rev=False, intermediate_outputs=False):

        iterator = range(len(self.module_list))
        jac = 0

        if rev:
            iterator = reversed(iterator)

        for i in iterator:
            if self.conditions[i] is None:
                x, j = (self.module_list[i]([x], rev=rev)[0],
                        self.module_list[i].jacobian(x, rev=rev))
            else:
                x, j = (self.module_list[i]([x], c=[c[self.conditions[i]]], rev=rev)[0],
                        self.module_list[i].jacobian(x, c=[c[self.conditions[i]]], rev=rev))
            jac = j + jac

        return x, jac


if __name__ == '__main__':

    inn = ReversibleSequential(3, 32, 32)

    def subnet(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, 16, 3, padding=1),
                             nn.ReLU(),
                             nn.Conv2d(16, c_out, 3, padding=1))

    cond_shapes = [(16, 32, 32), (32, 16, 16), (64, 8, 8)]
    for j in range(3):
        for k in range(3):
            inn.append(AllInOneBlock, subnet_constructor=subnet, permute_soft=True, cond=j, cond_shape=cond_shapes[j])
        if j < 2:
            inn.append(HaarDownsampling)

    for s in inn.shapes:
        print(s)

    x = torch.FloatTensor(8, 3, 32, 32).normal_(0,1)
    c = [torch.FloatTensor(8, *s).normal_(0,1) for s in cond_shapes]

    out, jac = inn(x, c)
    x_inv, jac_inv = inn(out, c, rev=True)

    err = torch.abs(x - x_inv)
    err_j = torch.abs(jac + jac_inv)

    print(err.max().item())
    print(err.mean().item())
    print(err_j.max().item())
    print(err_j.mean().item())

