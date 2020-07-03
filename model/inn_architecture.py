import torch.nn as nn
import FrEIA.modules as Fm

from all_in_one_block import AllInOneBlock
from reversible_sequential_net import ReversibleSequential

def _xavier_init(m, gain=1.):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.zero_().add_(gain)
        m.bias.data.zero_()

def build_inn(args):
    data_dims       = eval(args['data']['crop_to'])
    data_channels   = eval(args['data']['base_channels'])

    cond_ch         = eval(args['model']['cond_net_channels'])
    cinn_c_blocks   = eval(args['model']['inn_coupling_blocks'])
    cinn_tot_levels = len(cinn_c_blocks)
    cinn_channels   = eval(args['model']['inn_subnet_channels'])
    cinn_clamps     = eval(args['model']['affine_clamp'])
    cinn_actnorm    = eval(args['model']['global_affine_init'])
    cinn_total_resolution_levels = len(cinn_c_blocks)

    def subnet_conv(ch_hidden):

        def f(ch_in, ch_out):
            net = nn.Sequential(
                nn.Conv2d(ch_in, ch_hidden, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch_hidden),
                nn.ReLU(),

                nn.Conv2d(ch_hidden, ch_hidden, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch_hidden),
                nn.ReLU(),

                nn.Conv2d(ch_hidden, ch_out, 3, padding=1))

            net.apply(lambda m: _xavier_init(m, gain=1.)) #max(1., ch_hidden / 64.)**1.5))
            return net

        return f


    cond_shapes = [(cond_ch[i], data_dims//(2**i), data_dims//(2**i)) for i in range(cinn_tot_levels)]

    inn = ReversibleSequential(data_channels, data_dims, data_dims)

    for i in range(cinn_total_resolution_levels):
        block_args = {'cond':               i,
                      'cond_shape':         cond_shapes[i],
                      'subnet_constructor': subnet_conv(cinn_channels[i]),
                      'affine_clamping':    cinn_clamps[i],
                      'global_affine_init': cinn_actnorm[i],
                      'permute_soft':       (i < 3)
                     }

        for k in range(cinn_c_blocks[i]):
            inn.append(AllInOneBlock, **block_args)

        # dont need HaarDownsampling for the last layer
        if i != cinn_tot_levels - 1:
            inn.append(Fm.HaarDownsampling)

    return inn
