"""Check two GLoW transforms on the same PM-plus-shear time response.

DirectFT is sensitive to the sampled response at high w. GLoW's regularized
FFT analytically restores the image singularities and is the reference used
in compare.py. This diagnostic keeps lens, Nt, and tmax identical.
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bgo_matplotlib')

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from glow import lenses, time_domain_c, freq_domain_c
from expansion import amplification
from compare import COLORS  # applies the project plot style

HERE = Path(__file__).resolve().parent
w = np.geomspace(10, 100, 101)
lens = lenses.CombinedLens({'lenses': [
    lenses.Psi_PointLens({'psi0': 1}),
    lenses.Psi_Ext({'gamma1': 0.05, 'gamma2': 0.0})]})
response = time_domain_c.It_MultiContour_C(lens, 0.3,
                    {'Nt': 10000, 'tmax': 1e4, 'parallel': False})
direct = freq_domain_c.Fw_DirectFT_C(response, {'parallel': False})(w)
regularized = freq_domain_c.Fw_FFT_C(response,
                    {'wmin': 10, 'wmax': 120, 'eval_mode': 'exact',
                     'parallel': False})(w)
asymptotic = amplification(w, 'PM', [0.3, 0], shear=(0.05, 0), order=6)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
axes[0].semilogx(w, abs(direct), label='GLoW direct transform', color='coral')
axes[0].semilogx(w, abs(regularized), label='GLoW regularized FFT', color='black')
axes[0].semilogx(w, abs(asymptotic), label='sixth order', color='navy', linestyle='--')
axes[0].set_ylabel(r'$|F(w)|$')
axes[0].legend()
axes[1].loglog(w, abs(direct-asymptotic), label='direct transform', color='coral')
axes[1].loglog(w, abs(regularized-asymptotic), label='regularized FFT', color='black')
axes[1].set_ylabel(r'$|F_{\rm GLoW}-F^{(6)}|$')
axes[1].set_xlabel(r'dimensionless frequency $w$')
axes[1].legend()
for ax in axes:
    ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(HERE/'reference_diagnostic.jpg', dpi=200)
plt.close(fig)

def pair(z):
    return [float(z.real), float(z.imag)]

samples = {}
for target in (20, 40, 80):
    i = int(np.argmin(abs(w-target)))
    samples[str(target)] = {
        'w': float(w[i]), 'direct_F': pair(direct[i]),
        'regularized_F': pair(regularized[i]),
        'sixth_order_F': pair(asymptotic[i]),
        'direct_difference': float(abs(direct[i]-asymptotic[i])),
        'regularized_difference': float(abs(regularized[i]-asymptotic[i])),
    }
(HERE/'reference_diagnostic.json').write_text(json.dumps(samples, indent=2))
print(json.dumps(samples, indent=2))
