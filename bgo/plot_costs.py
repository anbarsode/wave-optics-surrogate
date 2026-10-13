"""Plot measured cumulative-order costs and rough power-law extrapolations.

The timing data come from compare.py: median of three one-thread evaluations
of 121 frequencies, including image finding and coefficient construction.
For the extrapolation, fit T_K = A K**p to orders K=3,...,6 in log space.
Orders 2,...,6 supply a fit-window sensitivity range. These are empirical
fits, not a complexity theorem; the partition count grows faster than a
power law beyond sufficiently high order.
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
from compare import COLORS  # applies the project plot style

HERE = Path(__file__).resolve().parent
measurements = json.loads((HERE/'comparison_metrics.json').read_text())
cases = list(measurements.items())


def fit_power(cost, first_order):
    """Least-squares fit to log(T_K) = log(A) + p log(K)."""
    k = np.arange(first_order, 7, dtype=float)
    log_a, p = np.polynomial.polynomial.polyfit(np.log(k), np.log(cost[first_order:]), 1)
    fitted = log_a+p*np.log(k)
    residual = np.sum((np.log(cost[first_order:])-fitted)**2)
    total = np.sum((np.log(cost[first_order:])-np.mean(np.log(cost[first_order:])))**2)
    return float(np.exp(log_a)), float(p), float(1-residual/total)


def crossing(cost, glow_cost):
    """Observed crossing interval, or None if the formulas remain faster."""
    for k in range(1, len(cost)):
        if cost[k-1] < glow_cost <= cost[k]:
            return [k-1, k]
    if cost[0] >= glow_cost:
        return [0, 0]
    return None


fig, axes = plt.subplots(4, 2, figsize=(14, 17), sharex=True)
fit_results = {}
for axis, (name, case) in zip(axes.flat, cases):
    cost = np.asarray(case['cumulative_order_cost_seconds'])*1000
    reference_times = case['reference_cost_seconds']
    glow_cost = 1000*reference_times.get(
        'analytic_or_semianalytic', reference_times.get('20000'))
    a3, p3, r3 = fit_power(cost, 3)
    a2, p2, r2 = fit_power(cost, 2)
    cross3 = (glow_cost/a3)**(1/p3)
    cross2 = (glow_cost/a2)**(1/p2)
    observed = crossing(cost, glow_cost)
    is_shear = any(case['shear'])
    model_color = COLORS[0] if case['model'] == 'PM' else COLORS[3]

    measured_k = np.arange(7)
    axis.plot(measured_k+1, cost, 'o-', color=model_color, label='measured analytic',
              linewidth=2, markersize=5)
    fitted_k = np.geomspace(3, 100, 250)
    axis.plot(fitted_k+1, a3*fitted_k**p3, color=model_color, linestyle='--',
              linewidth=1.7, label=rf'fit $K^{{{p3:.2f}}}$')
    axis.axhline(glow_cost, color='black', linewidth=1.8, label='GLoW')
    if is_shear:
        axis.axvspan(min(cross2, cross3)+1, max(cross2, cross3)+1,
                     color='gray', alpha=0.18, label='fit-window range')
        axis.plot(cross3+1, glow_cost, marker='D', markersize=6, color=model_color)
        caption = rf'break-even $K\approx {cross3:.0f}$'
    elif observed == [0, 0]:
        caption = 'GLoW faster from order 0'
    else:
        axis.plot(cross3+1, glow_cost, marker='D', markersize=6, color=model_color)
        caption = rf'break-even $K\approx {cross3:.1f}$'
    axis.set_title(name.replace(' / ', ': ')+'\n'+caption, fontsize=12)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlim(1, 105)
    axis.set_ylim(0.25, 12000)
    axis.set_xticks(np.array([0, 1, 2, 3, 4, 6, 10, 20, 40, 80])+1)
    axis.set_xticklabels(['0', '1', '2', '3', '4', '6', '10', '20', '40', '80'])
    axis.grid(True, which='both', alpha=0.25)
    axis.legend(loc='upper left', fontsize=8)
    fit_results[name] = {
        'model': case['model'], 'shear': case['shear'],
        'image_count': case['image_count'],
        'glow_cost_ms': glow_cost,
        'analytic_cost_ms_orders_0_to_6': cost.tolist(),
        'fit_orders_3_to_6': {
            'A_ms': a3, 'power': p3, 'log_space_R2': r3,
            'break_even_order': cross3},
        'fit_orders_2_to_6': {
            'A_ms': a2, 'power': p2, 'log_space_R2': r2,
            'break_even_order': cross2},
        'observed_crossing_interval': observed,
    }

for axis in axes[-1, :]:
    axis.set_xlabel(r'cumulative order $K$ beyond geometric')
for axis in axes[:, 0]:
    axis.set_ylabel('wall time for 121 frequencies (ms)')
fig.tight_layout()
fig.savefig(HERE/'computational_costs.jpg', dpi=200)
plt.close(fig)
(HERE/'cost_fit_metrics.json').write_text(json.dumps(fit_results, indent=2))
for name, result in fit_results.items():
    main = result['fit_orders_3_to_6']
    alternative = result['fit_orders_2_to_6']
    print(name, 'power', round(main['power'], 2),
          'break-even', round(main['break_even_order'], 1),
          'fit-window alternative', round(alternative['break_even_order'], 1),
          'observed', result['observed_crossing_interval'])
