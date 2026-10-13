"""Compare orders zero through ten at matched source position and phase cycles."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
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

HERE = Path(__file__).resolve().parent
COLORS = ['navy', 'coral', 'c', 'crimson', 'springgreen', 'gold',
          'dodgerblue', 'hotpink', 'forestgreen', 'moccasin', 'darkred']
use_tex = all(shutil.which(name) for name in ('latex', 'dvipng'))
rcParams.update({'axes.labelsize': 13, 'axes.titlesize': 12, 'legend.fontsize': 9,
                 'xtick.labelsize': 11, 'ytick.labelsize': 11,
                 'text.usetex': use_tex, 'text.latex.preamble': r'\usepackage{txfonts}',
                 'font.size': 12, 'font.family': 'serif',
                 'font.serif': 'times' if use_tex else 'DejaVu Serif',
                 'mathtext.default': 'sf', 'lines.linewidth': 1.8,
                 'axes.prop_cycle': cycler(color=COLORS)})
GAMMA = (0.05, 0.03)
GAMMA_FOUR = (0.40, 0.02)
CASES = {
 'PM': [
  dict(key='isolated_two', name='isolated, two images', y=.5, strength=1., convergence=0., shear=(0.,0.)),
  dict(key='convergence_two', name='convergence, two images', y=.5, strength=1., convergence=.2, shear=(0.,0.)),
  dict(key='shear_two', name='shear, two images', y=.5, strength=1., convergence=0., shear=(.05,0.)),
  dict(key='convergence_shear_two', name='convergence and shear, two images', y=.5, strength=1., convergence=.2, shear=GAMMA),
  dict(key='convergence_shear_four', name='convergence and shear, four images', y=.5, strength=1., convergence=.2, shear=GAMMA_FOUR)],
 'SIS': [
  dict(key='isolated_two', name='isolated, two images', y=.5, strength=1., convergence=0., shear=(0.,0.)),
  dict(key='isolated_one', name='isolated, one image', y=.5, strength=.4, convergence=0., shear=(0.,0.)),
  dict(key='convergence_two', name='convergence, two images', y=.5, strength=1., convergence=.2, shear=(0.,0.)),
  dict(key='convergence_one', name='convergence, one image', y=.5, strength=.4, convergence=.2, shear=(0.,0.)),
  dict(key='shear_four', name='shear, four images', y=.5, strength=1., convergence=0., shear=GAMMA_FOUR),
  dict(key='convergence_shear_two', name='convergence and shear, two images', y=.5, strength=1., convergence=.2, shear=GAMMA),
  dict(key='convergence_shear_four', name='convergence and shear, four images', y=.5, strength=1., convergence=.2, shear=GAMMA_FOUR),
  dict(key='convergence_shear_one', name='convergence and shear, one image', y=.5, strength=.4, convergence=.2, shear=GAMMA)]}


def glow_reference(model, y, strength, shear, convergence, w):
    """Use GLoW analytic routines under exact mass-sheet scaling when gamma=0."""
    start = time.perf_counter()
    if shear == (0., 0.):
        scale = 1-convergence
        yy = y/np.sqrt(scale)
        if model == 'PM':
            if strength != 1:
                raise ValueError('The analytic PM reference currently expects unit strength')
            if scale != 1:
                # PM strength stays unity under the mass-sheet coordinate change.
                solver = freq_domain_c.Fw_AnalyticPointLens_C(yy, {'parallel': False})
            else:
                solver = freq_domain_c.Fw_AnalyticPointLens_C(y, {'parallel': False})
        else:
            solver = freq_domain_c.Fw_SemiAnalyticSIS_C(
                yy, {'parallel': False}, psi0=strength/np.sqrt(scale))
        exact = solver(w)/scale
        return exact, np.zeros_like(w), {'analytic_or_semianalytic': time.perf_counter()-start}
    scale = 1-convergence
    transformed_y = y/np.sqrt(scale)
    transformed_shear = tuple(z/scale for z in shear)
    transformed_strength = strength if model == 'PM' else strength/np.sqrt(scale)
    base = (lenses.Psi_PointLens if model == 'PM' else lenses.Psi_SIS)(
        {'psi0': transformed_strength})
    lens = lenses.CombinedLens({'lenses': [base,
        lenses.Psi_Ext({'gamma1': transformed_shear[0],
                        'gamma2': transformed_shear[1]})]})
    values, times = {}, {}
    transformed_images = _image_data(model, [transformed_y, 0.], transformed_strength,
                                     transformed_shear, 0, 0.)
    if model == 'PM' and len(transformed_images) == 4:
        # GLoW's multi-contour initialization is unreliable for this PM
        # four-image configuration. Its slower area integral is robust. Supply
        # the independently solved stationary points so the FFT can regularize
        # the image singularities in I(tau).
        for n_rho in (40000, 80000):
            start = time.perf_counter()
            response = time_domain_c.It_AreaIntegral_C(lens, transformed_y,
                {'n_rho': n_rho, 'n_theta': 4000, 'Nt': 20000, 'tmax': 1000})
            response.p_crits = [{
                'type': {0: 'min', 1: 'saddle', 2: 'max'}[image['morse']],
                't': image['tau'], 'x1': image['x'][0], 'x2': image['x'][1],
                'mag': abs(1/image['determinant'])} for image in transformed_images]
            transform = freq_domain_c.Fw_FFT_C(response,
                {'wmin': float(min(w)), 'wmax': float(1.2*max(w)),
                 'eval_mode': 'exact', 'parallel': False})
            values[n_rho] = transform(w)/scale
            times[f'area_{n_rho}'] = time.perf_counter()-start
        return values[80000], abs(values[80000]-values[40000]), times
    for nt in (16000, 20000):
        start = time.perf_counter()
        response = time_domain_c.It_MultiContour_C(lens, transformed_y,
                    {'Nt': nt, 'tmax': 1e4, 'parallel': False})
        transform = freq_domain_c.Fw_FFT_C(response,
                    {'wmin': float(min(w)), 'wmax': float(1.2*max(w)),
                     'eval_mode': 'exact', 'parallel': False})
        values[nt] = transform(w)/scale
        times[str(nt)] = time.perf_counter()-start
    return values[20000], abs(values[20000]-values[16000]), times


def benchmark(model, y, strength, shear, convergence, w):
    """Median complete evaluation wall time per cumulative order."""
    predictions, costs = [], []
    for order in range(11):
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            prediction = amplification(w, model, [y, 0.], shear=shear,
                                       convergence=convergence, strength=strength,
                                       order=order)
            elapsed.append(time.perf_counter()-start)
        predictions.append(prediction)
        costs.append(float(np.median(elapsed)))
    return predictions, costs


def pair(z):
    return [float(z.real), float(z.imag)]


def delay_span(model, y, strength, shear, convergence):
    """Largest stationary phase separation used to count oscillation cycles.

    For SIS, include its conical-center phase because it is the second phase
    in a one-image configuration. With two or four images this definition is
    the fastest beat among all image and center contributions.
    """
    data = _image_data(model, [y, 0.], strength, shear, 0, convergence)
    phases = [image['tau'] for image in data]
    if model == 'SIS':
        phases.append(y*y/2)
    return max(phases)-min(phases), len(data)


def plot_ten_cycle_summary(output):
    """Compare every cumulative order at the same ten-cycle phase range."""
    names = list(output)
    errors = np.array([output[name]['ten_cycle_errors'] for name in names])
    fig, axis = plt.subplots(figsize=(11, 7.5))
    image = axis.imshow(np.log10(np.maximum(errors, 1e-12)), aspect='auto',
                        cmap='viridis', vmin=-6, vmax=2)
    axis.set_xticks(range(11))
    axis.set_xlabel('cumulative order $K$')
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels([name.replace(' / ', ': ') for name in names], fontsize=8)
    axis.set_title(r'Errors at matched source $y=0.5$ and ten phase cycles')
    fig.colorbar(image, ax=axis, label=r'$\log_{10}$ relative complex error')
    fig.tight_layout()
    fig.savefig(HERE/'matched_y_error_at_ten_cycles.jpg', dpi=200)
    plt.close(fig)


def main():
    output = {}
    selected = (0, 1, 2, 4, 6, 8, 10)
    for model, cases in CASES.items():
        for case in cases:
            y, strength = case['y'], case['strength']
            shear, convergence = case['shear'], case['convergence']
            phase_delay, image_count = delay_span(model, y, strength, shear, convergence)
            phase_cycles = np.geomspace(.5, 20, 121)
            phase_cycles[np.argmin(abs(phase_cycles-10))] = 10
            w = 2*np.pi*phase_cycles/phase_delay
            exact, spread, glow_cost = glow_reference(
                model, y, strength, shear, convergence, w)
            approx, costs = benchmark(model, y, strength, shear, convergence, w)
            scale = np.maximum(abs(exact), .2)
            error = np.array([abs(p-exact)/scale for p in approx])
            uncertainty = spread/scale
            fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True,
                                      gridspec_kw={'height_ratios':[1,1.1,.65]})
            axes[0].semilogx(phase_cycles, abs(exact), 'k', linewidth=2.4, label='GLoW')
            for k in selected:
                label = 'geometric' if k == 0 else f'order {k}'
                axes[0].semilogx(phase_cycles, abs(approx[k]), color=COLORS[k],
                                 linestyle=':' if k == 0 else '-', linewidth=1.3, label=label)
                axes[1].loglog(phase_cycles, error[k], color=COLORS[k], label=label)
            if any(shear):
                axes[1].loglog(phase_cycles, uncertainty, color='gray', label='GLoW resolution spread')
            mesh = axes[2].pcolormesh(phase_cycles, np.arange(11),
                        np.log10(np.maximum(error,1e-12)), shading='nearest',
                        cmap='viridis', vmin=-6, vmax=2)
            fig.colorbar(mesh, ax=axes[2], label=r'$\log_{10}$ relative error')
            strength_name = 'a' if model == 'PM' else 'b'
            axes[0].set_title(
                f'{model}: {case["name"]}; $y={y}$, ${strength_name}={strength}$; '
                f'{image_count} images\n'
                f'$\\kappa={convergence}$, $\\gamma=({shear[0]},{shear[1]})$; '
                f'$w={w[0]:.1f}$--${w[-1]:.1f}$')
            axes[0].set_yscale('log')
            # Keep the full-wave signal visible when high-order terms blow up;
            # their full divergence remains visible in the error panel.
            upper = 20*max(abs(exact))
            lower = max(min(abs(exact))/10, 1e-5)
            axes[0].set_ylim(lower, upper)
            if max(max(abs(p)) for p in approx) > upper:
                axes[0].text(.02, .03, 'large high-order values clipped; see error panel',
                             transform=axes[0].transAxes, fontsize=9)
            axes[0].set_ylabel(r'$|F(w)|$')
            axes[1].set_ylabel(r'$|F_K-F_{\rm GLoW}|/\max(|F_{\rm GLoW}|,0.2)$')
            axes[2].set_ylabel('order $K$')
            axes[2].set_yticks([0,2,4,6,8,10])
            axes[2].set_xlabel(r'phase cycles $N_\Delta=w\Delta\tau/(2\pi)$')
            for ax in axes[:2]:
                ax.grid(True, which='both', alpha=.25)
                ax.legend(ncol=4, fontsize=8)
            axes[1].set_ylim(1e-8,1e8)
            fig.tight_layout()
            filename = f'{model.lower()}_{case["key"]}_orders_0_to_10.jpg'
            fig.savefig(HERE/filename, dpi=200)
            plt.close(fig)
            ten = int(np.argmin(abs(phase_cycles-10)))
            output[f'{model} / {case["name"]}'] = {
                'model':model, 'y':y, 'strength':strength,
                'convergence':convergence, 'shear':list(shear),
                'image_count':image_count, 'plot':filename,
                'delay_span':float(phase_delay),
                'phase_cycle_grid':phase_cycles.tolist(),
                'reference_method':('mass-sheet transformed exact GLoW' if not any(shear)
                    else 'GLoW area integral and regularized FFT, n_rho=80000'
                    if model == 'PM' and image_count == 4
                    else 'mass-sheet transformed GLoW regularized FFT, Nt=20000'),
                'reference_cost_seconds':glow_cost,
                'cumulative_order_cost_seconds':costs,
                'frequency_grid':w.tolist(),
                'glow_values':[pair(z) for z in exact],
                'analytic_values':[[pair(z) for z in p] for p in approx],
                'normalized_errors':error.tolist(),
                'glow_resolution_spread':uncertainty.tolist(),
                'minimum_error_order':[int(np.argmin(error[:,i])) for i in range(len(w))],
                'ten_cycle_frequency':float(w[ten]),
                'ten_cycle_errors':error[:,ten].tolist(),
                'ten_cycle_best_order':int(np.argmin(error[:,ten])),
                'ten_cycle_glow_resolution_spread':float(uncertainty[ten])}
            print(model, case['name'], 'images', image_count, 'GLoW', glow_cost,
                  'order10', round(costs[10],3), flush=True)
            (HERE/'comparison_metrics.json').write_text(json.dumps(output,indent=2))
    plot_ten_cycle_summary(output)
    return output


if __name__ == '__main__':
    main()
