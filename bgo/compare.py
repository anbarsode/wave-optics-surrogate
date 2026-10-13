"""Compare orders zero through six with GLoW and measure cumulative costs.

Run with the gw Python. Isolated PM/SIS use GLoW's analytic/semi-analytic
frequency solvers. Sheared lenses use GLoW's full multi-contour time response
and regularized FFT at three resolutions.
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bgo_matplotlib')

import json
import shutil
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
from glow import lenses, time_domain_c, freq_domain_c
from expansion import amplification, _image_data

COLORS = ['navy', 'coral', 'c', 'crimson', 'springgreen', 'gold', 'dodgerblue',
          'hotpink', 'forestgreen', 'moccasin', 'darkred', 'teal']
missing_tex_tools = [name for name in ('latex', 'dvipng') if shutil.which(name) is None]
if missing_tex_tools:
    print('For the requested TeX plot style, please install: ' + ', '.join(missing_tex_tools))
use_tex = not missing_tex_tools
rcParams.update({'axes.labelsize': 14, 'axes.titlesize': 14, 'legend.fontsize': 10,
                 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                 'text.usetex': use_tex, 'text.latex.preamble': r'\usepackage{txfonts}',
                 'font.size': 14, 'font.family': 'serif',
                 'font.serif': 'times' if use_tex else 'DejaVu Serif',
                 'mathtext.default': 'sf', 'lines.markeredgewidth': 1,
                 'lines.linewidth': 2, 'axes.prop_cycle': cycler(color=COLORS)})

HERE = Path(__file__).resolve().parent
GAMMA = (0.05, 0.03)
CASES = {
    'PM': [dict(key='isolated_two', name='isolated, two images', y=0.3, shear=(0., 0.)),
           dict(key='shear_two', name='shear, two images', y=0.3, shear=(0.05, 0.)),
           dict(key='shear_four', name='shear, four images', y=0.03, shear=GAMMA)],
    'SIS': [dict(key='isolated_two', name='isolated, two images', y=0.3, shear=(0., 0.)),
            dict(key='isolated_one', name='isolated, one image', y=1.5, shear=(0., 0.)),
            dict(key='shear_two', name='shear, two images', y=0.3, shear=GAMMA),
            dict(key='shear_four', name='shear, four images', y=0.03, shear=GAMMA),
            dict(key='shear_one', name='shear, one image', y=1.5, shear=GAMMA)],
}


def glow_reference(model, y, shear, w, tmax):
    """Return reference, resolution spread, and complete GLoW wall cost."""
    times = {}
    if shear == (0., 0.):
        start = time.perf_counter()
        if model == 'PM':
            solver = freq_domain_c.Fw_AnalyticPointLens_C(y, {'parallel': False})
        else:
            solver = freq_domain_c.Fw_SemiAnalyticSIS_C(y, {'parallel': False})
        answer = solver(w)
        times['analytic_or_semianalytic'] = time.perf_counter()-start
        return answer, np.zeros_like(w), times
    base = (lenses.Psi_PointLens if model == 'PM' else lenses.Psi_SIS)({'psi0': 1})
    lens = lenses.CombinedLens({'lenses': [base,
        lenses.Psi_Ext({'gamma1': shear[0], 'gamma2': shear[1]})]})
    answers = {}
    for nt in (5000, 10000, 20000):
        start = time.perf_counter()
        response = time_domain_c.It_MultiContour_C(lens, y,
                    {'Nt': nt, 'tmax': tmax, 'parallel': False})
        transform = freq_domain_c.Fw_FFT_C(response, {'wmin': float(min(w)),
                    'wmax': float(1.2*max(w)), 'eval_mode': 'exact',
                    'parallel': False})
        answers[nt] = transform(w)
        times[str(nt)] = time.perf_counter()-start
    spread = abs(answers[20000]-answers[10000])
    return answers[20000], spread, times


def benchmark(model, y, shear, w):
    """Median wall time for each cumulative order, including image solving."""
    predictions, times = [], []
    for order in range(7):
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            prediction = amplification(w, model, [y, 0], shear=shear, order=order)
            samples.append(time.perf_counter()-start)
        predictions.append(prediction)
        times.append(float(np.median(samples)))
    return predictions, times


def complex_pair(z):
    return [float(z.real), float(z.imag)]


def main():
    results = {}
    for model, cases in CASES.items():
        for case in cases:
            y, shear = case['y'], case['shear']
            w = np.geomspace(3, 160 if shear == (0., 0.) else 100, 121)
            tmax = 1e4
            reference, uncertainty, glow_times = glow_reference(model, y, shear, w, tmax)
            predictions, times = benchmark(model, y, shear, w)
            scale = np.maximum(abs(reference), 0.2)
            errors = [abs(p-reference)/scale for p in predictions]
            spread = uncertainty/scale
            count = len(_image_data(model, [y, 0], 1, shear, order=0))
            fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [1, 1.2]})
            axes[0].semilogx(w, abs(reference), color='black', linewidth=2.3,
                             label='GLoW wave optics')
            for order in range(7):
                color = COLORS[order]
                name = 'geometric' if order == 0 else f'order {order}'
                axes[0].semilogx(w, abs(predictions[order]), color=color,
                                 linewidth=1.2, alpha=0.85,
                                 linestyle='--' if order else ':', label=name)
                axes[1].loglog(w, errors[order], color=color, linewidth=1.8,
                               label=name)
            if shear != (0., 0.):
                axes[1].loglog(w, spread, color='gray', linewidth=1.5,
                               label='GLoW resolution spread')
            axes[0].set_title(f'{model}: {case["name"]}; $y={y}$, '
                              f'$\\gamma=({shear[0]},{shear[1]})$; {count} images')
            axes[0].set_yscale('log')
            axes[0].set_ylabel(r'$|F(w)|$')
            axes[1].set_ylabel(r'$|F_{\rm approx}-F_{\rm GLoW}|/\max(|F_{\rm GLoW}|,0.2)$')
            axes[1].set_xlabel(r'dimensionless frequency $w=2\pi f t_0$')
            axes[1].set_ylim(1e-8, 1e7)
            for axis in axes:
                axis.grid(True, which='both', alpha=0.25)
            axes[0].legend(ncol=4, fontsize=9)
            axes[1].legend(ncol=4, fontsize=9)
            fig.tight_layout()
            fig.savefig(HERE/f'{model.lower()}_{case["key"]}_orders_0_to_6.jpg', dpi=200)
            plt.close(fig)
            samples = {}
            for target in (10, 20, 40, 80):
                index = int(np.argmin(abs(w-target)))
                samples[str(target)] = dict(w=float(w[index]),
                    glow=complex_pair(reference[index]),
                    predictions=[complex_pair(p[index]) for p in predictions],
                    normalized_error=[float(e[index]) for e in errors],
                    glow_resolution_spread=float(spread[index]))
            thresholds = {}
            if shear == (0., 0.):
                for tolerance in (0.01, 0.001):
                    thresholds[str(tolerance)] = []
                    for error in errors:
                        # A conservative onset: every higher sampled frequency passes.
                        failing = np.flatnonzero(error > tolerance)
                        index = int(failing[-1]+1) if len(failing) else 0
                        thresholds[str(tolerance)].append(
                            float(w[index]) if index < len(w) else None)
            results[f'{model} / {case["name"]}'] = dict(
                model=model, y=y, shear=list(shear), image_count=count,
                plot=f'{model.lower()}_{case["key"]}_orders_0_to_6.jpg',
                reference_method=('analytic PM' if model == 'PM' else 'semi-analytic SIS')
                    if shear == (0., 0.) else 'regularized FFT, Nt=20000',
                frequencies=len(w), reference_cost_seconds=glow_times,
                cumulative_order_cost_seconds=times, thresholds=thresholds,
                frequency_grid=[float(z) for z in w],
                glow_values=[complex_pair(z) for z in reference],
                analytic_values=[[complex_pair(z) for z in p] for p in predictions],
                normalized_errors=[[float(z) for z in e] for e in errors],
                glow_resolution_spread=[float(z) for z in spread], samples=samples)
            print(model, case['name'], 'GLoW', glow_times,
                  'orders 0..6', np.round(times, 4))
    (HERE/'comparison_metrics.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
