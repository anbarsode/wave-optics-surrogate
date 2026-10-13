"""Plot cumulative measured costs through order ten and empirical power-law extrapolations.

The fit is descriptive only. Partition growth and multiprecision arithmetic
make distant extrapolations unreliable.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bgo_matplotlib')

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from compare import COLORS

HERE = Path(__file__).resolve().parent
cases = list(json.loads((HERE/'comparison_metrics.json').read_text()).items())


def fit(cost, first):
    k = np.arange(first, len(cost), dtype=float)
    p, intercept = np.polyfit(np.log(k), np.log(cost[first:]), 1)
    fitted = intercept+p*np.log(k)
    r2 = 1-np.sum((np.log(cost[first:])-fitted)**2)/np.sum(
        (np.log(cost[first:])-np.mean(np.log(cost[first:])))**2)
    return float(np.exp(intercept)), float(p), float(r2)


def observed_crossing(cost, glow):
    for k in range(1, len(cost)):
        if cost[k-1] < glow <= cost[k]:
            return [k-1,k]
    return [0,0] if cost[0] >= glow else None


ncols = 2
nrows = int(np.ceil(len(cases)/ncols))
fig, axes = plt.subplots(nrows,ncols,figsize=(14,3.5*nrows),squeeze=False)
results = {}
for axis,(name,case) in zip(axes.flat,cases):
    cost = np.array(case['cumulative_order_cost_seconds'])*1000
    reference = case['reference_cost_seconds']
    glow = 1000*reference.get('20000', reference.get(
        'analytic_or_semianalytic', max(reference.values())))
    a,p,r2 = fit(cost,7)
    a5,p5,r25 = fit(cost,6)
    crossing = (glow/a)**(1/p) if p>0 else None
    crossing5 = (glow/a5)**(1/p5) if p5>0 else None
    x = np.arange(len(cost))
    color = COLORS[0] if case['model']=='PM' else COLORS[3]
    axis.plot(x,cost,'o-',color=color,label='measured cumulative order')
    fit_k = np.geomspace(7, max(30,min(150,(crossing or 30)*1.5)),200)
    axis.plot(fit_k,a*fit_k**p,'--',color=color,label=rf'fit $K^{{{p:.2f}}}$')
    axis.axhline(glow,color='black',label='GLoW')
    if crossing and 10 < crossing < 150:
        axis.plot(crossing,glow,'D',color=color)
    axis.set_title(name.replace(' / ',': ') +
         (f'; break even K≈{crossing:.1f}' if crossing and crossing>10 else
          '; GLoW already faster' if glow<=cost[10] else '; extrapolation uncertain'),
          fontsize=11)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlim(.8,150)
    axis.grid(True,which='both',alpha=.25)
    axis.legend(fontsize=8)
    results[name] = {
        'model':case['model'],'convergence':case['convergence'],
        'shear':case['shear'],'image_count':case['image_count'],
        'glow_cost_ms':glow,
        'analytic_cost_ms_orders_0_to_10':cost.tolist(),
        'fit_orders_7_to_10':{'A_ms':a,'power':p,'log_space_R2':r2,
                              'break_even_order':crossing},
        'fit_orders_6_to_10':{'A_ms':a5,'power':p5,'log_space_R2':r25,
                              'break_even_order':crossing5},
        'observed_crossing_interval':observed_crossing(cost,glow)}
for axis in axes.flat[len(cases):]:
    axis.axis('off')
for axis in axes[-1,:]:
    axis.set_xlabel('cumulative order $K$')
for row in axes:
    row[0].set_ylabel('wall time for 121 frequencies (ms)')
fig.tight_layout()
fig.savefig(HERE/'computational_costs_orders_0_to_10.jpg',dpi=200)
plt.close(fig)
(HERE/'cost_fit_metrics.json').write_text(json.dumps(results,indent=2))
for name,value in results.items():
    print(name,'power',round(value['fit_orders_7_to_10']['power'],2),
          'break even',value['fit_orders_7_to_10']['break_even_order'])
