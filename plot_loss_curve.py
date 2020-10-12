'''
Saves the loss curve to the same directory as the losses.dat file is in.
'''
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

BATCHES_PER_EPOCH = 450
FIG_WIDTH = 6
Y_MAX = -2.5

assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <losses.dat file>"

ll = np.loadtxt(sys.argv[1], skiprows=1).T
folder = os.path.dirname(sys.argv[1])

ll[0] *= BATCHES_PER_EPOCH / 1000.

plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH / 1.618))

plt.plot(ll[0], ll[2], color='seagreen', linestyle='--', label='Training')
plt.plot(ll[0], ll[3], color='royalblue', label='Validation')

ylims = plt.ylim()
xlims = plt.xlim()
plt.ylim(ylims[0], Y_MAX)
plt.xlim(0, np.max(ll[0]))

plt.ylabel('Loss  $\\mathcal{L}$', fontsize=12)
plt.xlabel('Iteration ($\cdot 10^3$)', fontsize=12)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.grid(True, alpha=0.4)

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder, 'loss_curve.pdf'))
