"""Build and compare canonical PM and SIS surrogates across source position.

Run with ``micromamba run -n gw python -m decomposition.surrogate_study``.
The saved .npz files contain only spline coefficients, source positions, and
BGO transition frequencies. Evaluation does not call the exact lens model.
"""

import os
import shutil

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from cycler import cycler
import numpy as np

from decomposition.isolated import isolated_lens
from decomposition.surrogate import (CanonicalBasis, bgo_start, build_surrogate,
                                     error_measures, geometry)


def plot_style():
    """Apply the project's publication-style settings."""
    my_colors = ['navy', 'coral', 'c', 'crimson', 'springgreen', 'gold',
                 'dodgerblue', 'hotpink', 'forestgreen', 'moccasin', 'darkred', 'teal']
    rc_params = {'axes.labelsize': 14, 'axes.titlesize': 14, 'legend.fontsize': 14,
                 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                 'text.usetex': shutil.which('dvipng') is not None,
                 'text.latex.preamble': r'\usepackage{txfonts}',
                 'font.size': 14, 'font.family': 'serif',
                 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                 'mathtext.default': 'sf', 'lines.markeredgewidth': 1,
                 'lines.linewidth': 2, 'axes.prop_cycle': cycler(color=my_colors)}
    rcParams.update(rc_params)
    if not rc_params['text.usetex']:
        print('Install dvipng to enable full LaTeX rendering in these figures.', flush=True)


def make_grids(model: str):
    """Shared frequency and source grids; validation uses unseen values."""
    y_min, y_max = ((0.01, 10.0) if model == 'PM' else (0.03, 0.9))
    y_train = np.geomspace(y_min, y_max, 24)
    if model == 'SIS':
        # The center term changes rapidly near y=0.12. A pilot comparison at
        # source midpoints identified this band as needing more source samples.
        y_train = np.sort(np.r_[y_train, np.geomspace(0.09, 0.16, 12)])
    y_test = np.sqrt(y_train[:-1]*y_train[1:])
    w_train = np.unique(np.r_[np.geomspace(0.01, 20, 220), np.linspace(0.01, 20, 180)])
    w_test = np.unique(np.r_[np.geomspace(0.0101, 19.9, 180),
                             np.linspace(0.071, 19.93, 180)])
    w_bgo = np.geomspace(0.01, 100, 420)
    return y_train, y_test, w_train, w_test, w_bgo


def plot_curves(model, surrogates, y_train, w, output):
    """Real and imaginary parts of both normalized G curves over source position."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    sample_y = np.sqrt(y_train[:-1]*y_train[1:])[::2]
    cmap = plt.get_cmap('plasma')
    norm = LogNorm(y_train[0], y_train[-1])
    for row, (label, surrogate) in enumerate(surrogates.items()):
        for y in sample_y:
            components, _ = surrogate.components(w, float(y))
            magnitudes, _ = geometry(model, float(y))
            normalized = components/magnitudes[:, None]
            color = cmap(norm(y))
            for image in range(2):
                axes[row, 2*image].plot(w, normalized[image].real,
                                        color=color, linewidth=1)
                axes[row, 2*image+1].plot(w, normalized[image].imag,
                                          color=color, linewidth=1)
        for col in range(4):
            axes[row, col].set_xscale('log')
            axes[row, col].grid(alpha=0.2)
            if row == 1:
                axes[row, col].set_xlabel(r'$w$')
        axes[row, 0].set_ylabel(label)
    for image, name in enumerate(('minimum', 'saddle')):
        axes[0, 2*image].set_title(rf'${{\rm Re}}(G_{{\rm {name}}}/\sqrt{{|\mu|}})$')
        axes[0, 2*image+1].set_title(rf'${{\rm Im}}(G_{{\rm {name}}}/\sqrt{{|\mu|}})$')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                 label=r'source position $y$', fraction=0.02, pad=0.02)
    fig.suptitle(f'{model}: canonical image curves over source position')
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.11, top=0.86,
                        wspace=0.28, hspace=0.25)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_errors(model, y_test, measurements, output, rms_limit):
    """Validation errors at source positions excluded from fitting."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    panels = [('complex_rms', r'RMS $|\Delta F|/|F|$'),
              ('complex_max', r'maximum $|\Delta F|/|F|$'),
              ('amplitude_rms', r'RMS $|\Delta |F||/|F|$'),
              ('amplitude_max', r'maximum $|\Delta |F||/|F|$'),
              ('phase_rms', r'RMS $|\Delta\arg F|$ [rad]'),
              ('phase_max', r'maximum $|\Delta\arg F|$ [rad]'),
              ('high_complex_rms', r'RMS $|\Delta F|/|F|$, $w\geq5$'),
              ('high_complex_max', r'maximum $|\Delta F|/|F|$, $w\geq5$')]
    colors = {'geometric baseline': 'navy', 'two BGO orders': 'coral'}
    for ax, (key, label) in zip(axes.ravel(), panels):
        for name, rows in measurements.items():
            ax.plot(y_test, [row[key] for row in rows], marker='o', markersize=3,
                    label=name, color=colors[name])
        if key == 'complex_rms':
            ax.axhline(rms_limit, color='gray', linestyle=':', label='training limit')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel(r'$y$')
    axes[0, 0].legend(fontsize=10)
    fig.suptitle(f'{model}: errors at unseen source positions, $0.01\leq w\leq20$')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_example(model, surrogates, output):
    """Amplification, residual, components, amplitude, phase at unseen y=0.5."""
    w = np.geomspace(0.01, 20, 480)
    y = 0.5
    data = isolated_lens(model, w, y)
    predictions = {name: surrogate.amplification(w, y)
                   for name, surrogate in surrogates.items()}
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    colors = {'geometric baseline': 'navy', 'two BGO orders': 'coral'}
    ax = axes[0, 0]
    ax.plot(data.wave.real, data.wave.imag, color='black', label='wave optics')
    ax.plot(data.geometric.real, data.geometric.imag, color='gray', linestyle=':',
            label='geometric optics')
    for name, prediction in predictions.items():
        ax.plot(prediction.real, prediction.imag, color=colors[name],
                linestyle='--' if name == 'geometric baseline' else '-.', label=name)
    ax.set(xlabel=r'${\rm Re}\,F$', ylabel=r'${\rm Im}\,F$', title='Complex amplification')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for name, prediction in predictions.items():
        residual = prediction-data.wave
        ax.plot(residual.real, residual.imag, color=colors[name], label=name)
    ax.scatter([0], [0], marker='+', color='black')
    ax.set(xlabel=r'${\rm Re}\,\Delta F$', ylabel=r'${\rm Im}\,\Delta F$',
           title='Complex residual')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=9)
    for name, prediction in [('wave optics', data.wave),
                             ('geometric optics', data.geometric), *predictions.items()]:
        color = {'wave optics': 'black', 'geometric optics': 'gray'}.get(name,
                                                                          colors.get(name))
        style = ':' if name == 'geometric optics' else '-'
        axes[1, 0].plot(w, abs(prediction), color=color, linestyle=style, label=name)
        axes[1, 1].plot(w, np.unwrap(np.angle(prediction)), color=color,
                        linestyle=style, label=name)
    axes[1, 0].set(ylabel=r'$|F|$', title='Amplitude')
    axes[1, 1].set(ylabel=r'unwrapped $\arg F$ [rad]', title='Phase')
    for name, prediction in predictions.items():
        axes[2, 0].plot(w, abs(prediction)/abs(data.wave)-1,
                        color=colors[name], label=name)
        axes[2, 1].plot(w, np.angle(prediction*np.conj(data.wave)),
                        color=colors[name], label=name)
    axes[2, 0].set(ylabel=r'$|F_{\rm sur}|/|F_{\rm wave}|-1$',
                   title='Signed fractional amplitude error')
    axes[2, 1].set(ylabel=r'$\arg(F_{\rm sur}/F_{\rm wave})$ [rad]',
                   title='Signed phase error')
    for ax in axes[1:].ravel():
        ax.set_xscale('log')
        ax.set_xlabel(r'$w$')
        ax.grid(alpha=0.2)
    axes[1, 0].legend(fontsize=9)
    fig.suptitle(f'{model}: surrogate at an unseen source position $y=0.5$')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_component_plane(model, surrogates, output):
    """Both fitted G curves in the complex plane at y=0.5."""
    w = np.geomspace(0.01, 20, 400)
    y = 0.5
    magnitudes, _ = geometry(model, y)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, (name, surrogate) in enumerate(surrogates.items()):
        components, _ = surrogate.components(w, y)
        for image in range(2):
            ax = axes[row, image]
            curve = components[image]
            ax.plot(curve.real, curve.imag, color='gray', linewidth=1)
            ax.scatter(curve.real, curve.imag, c=np.log10(w), cmap='viridis', s=8)
            ax.scatter([magnitudes[image]], [0], color='crimson', marker='x', s=50)
            ax.set(xlabel=r'${\rm Re}\,G_i$', ylabel=r'${\rm Im}\,G_i$',
                   title=f'{name}: {"minimum" if image == 0 else "saddle"}')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=0.2)
    fig.suptitle(f'{model}: image factors at $y=0.5$; cross is geometric limit')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def run_study(output_dir='decomposition/surrogate_results'):
    """Build two surrogates per lens, validate them, and save figures and data."""
    plot_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    basis = CanonicalBasis(0.01, 20, knot_count=14)
    rms_limit = 0.005
    summary = {}
    for model in ('PM', 'SIS'):
        print(f'Building {model} surrogates', flush=True)
        y_train, y_test, w_train, w_test, w_bgo = make_grids(model)
        train_waves = np.array([isolated_lens(model, w_train, float(y)).wave
                                for y in y_train])
        starts = np.array([
            bgo_start(w_bgo, model, float(y), isolated_lens(model, w_bgo, float(y)).wave)
            for y in y_train
        ])
        test_waves = np.array([isolated_lens(model, w_test, float(y)).wave
                               for y in y_test])
        surrogates, measurements, train_errors = {}, {}, {}
        for order, label, stem in ((0, 'geometric baseline', 'geo'),
                                   (2, 'two BGO orders', 'bgo2')):
            surrogate, errors, _ = build_surrogate(
                model, order, y_train, w_train, basis, rms_limit, train_waves, starts)
            surrogate.save(output/f'{model.lower()}_{stem}.npz')
            surrogates[label] = surrogate
            train_errors[label] = errors
            measurements[label] = []
            high = w_test >= 5
            for y, truth in zip(y_test, test_waves):
                prediction = surrogate.amplification(w_test, float(y))
                row = error_measures(truth, prediction)
                high_row = error_measures(truth[high], prediction[high])
                row['high_complex_rms'] = high_row['complex_rms']
                row['high_complex_max'] = high_row['complex_max']
                measurements[label].append(row)
        rows = []
        for label, result in measurements.items():
            for y, metrics in zip(y_test, result):
                rows.append({'lens': model, 'method': label, 'y': y, **metrics})
        with (output/f'{model.lower()}_errors.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        plot_curves(model, surrogates, y_train, np.geomspace(0.01, 20, 300),
                    output/f'{model.lower()}_curves.jpg')
        plot_errors(model, y_test, measurements, output/f'{model.lower()}_errors.jpg', rms_limit)
        plot_example(model, surrogates, output/f'{model.lower()}_example.jpg')
        plot_component_plane(model, surrogates, output/f'{model.lower()}_components.jpg')
        summary[model] = {
            'y_range': [float(y_train[0]), float(y_train[-1])],
            'training_y_count': len(y_train), 'validation_y_count': len(y_test),
            'frequency_range': [0.01, 20.0], 'knots_per_image': basis.knot_count,
            'rms_training_limit': rms_limit,
            'bgo_reference_range': [0.01, 100.0],
            'bgo_start_min_max': [float(starts.min()), float(starts.max())],
            'methods': {
                label: {
                    'maximum_training_complex_rms': float(errors.max()),
                    **{f'validation_{key}_median': float(np.median([r[key] for r in
                                                                      measurements[label]]))
                       for key in measurements[label][0]},
                    **{f'validation_{key}_max': float(np.max([r[key] for r in
                                                               measurements[label]]))
                       for key in measurements[label][0]},
                } for label, errors in train_errors.items()
            }
        }
        print(json.dumps({model: summary[model]}, indent=2), flush=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    run_study()
