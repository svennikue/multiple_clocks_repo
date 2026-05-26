#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polar-plot helpers for visualising single-neuron tuning in the el-gaby
framework.  Each 360-bin trial is laid around the circle as 4 states ×
90 bins, with 3 phase chunks of 30 bins inside each state. State A is
at the top of the polar (12 o'clock); states proceed clockwise to B, C,
D.

These helpers are pure plotting: the caller is responsible for loading
the (n_correct_trials, 360) neuron array per config and looking up
pref_state / pref_phase from Script 1's tuning CSV.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


N_BINS_PER_TRIAL = 360
N_STATES = 4
N_PHASES = 3
N_BINS_PER_STATE = N_BINS_PER_TRIAL // N_STATES         # 90
N_BINS_PER_PHASE = N_BINS_PER_STATE // N_PHASES         # 30
STATE_LABELS = ['A', 'B', 'C', 'D']


def _polar_axes(ax):
    """Apply el-gaby's polar conventions: 0 at top, clockwise, ABCD at quadrants."""
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, N_STATES, endpoint=False))
    ax.set_xticklabels(STATE_LABELS, fontsize=10)
    ax.grid(True, alpha=0.3)


def _shade_pref_phase(ax, pref_phase, r_max, color='red', alpha=0.10):
    """Shade the pref_phase bins of every state (30 bins per state)."""
    if pref_phase is None or pref_phase < 0:
        return
    bin_to_theta = 2 * np.pi / N_BINS_PER_TRIAL
    for s in range(N_STATES):
        b0 = s * N_BINS_PER_STATE + pref_phase * N_BINS_PER_PHASE
        b1 = b0 + N_BINS_PER_PHASE
        thetas = np.linspace(b0, b1, 50) * bin_to_theta
        ax.fill_between(thetas, 0, r_max, color=color, alpha=alpha,
                        linewidth=0)


def _mark_pref_state(ax, pref_state, r_max, color='red'):
    """Draw a tick at the centre of the preferred-state quadrant."""
    if pref_state is None or pref_state < 0:
        return
    theta_centre = (pref_state + 0.5) * (2 * np.pi / N_STATES)
    ax.plot([theta_centre, theta_centre], [r_max * 0.95, r_max * 1.05],
            color=color, lw=2)


def smooth_circular(trace, sigma):
    """Gaussian-smooth a 1D trace with circular boundary handling."""
    if sigma is None or sigma <= 0:
        return trace
    trace = np.asarray(trace, dtype=float)
    if not np.isfinite(trace).any():
        return trace
    # Replace NaNs with the mean of the finite values before smoothing, then
    # propagate the original NaN mask back to avoid 0-padding edges.
    finite = np.isfinite(trace)
    if not finite.all():
        filled = trace.copy()
        filled[~finite] = np.nanmean(trace)
    else:
        filled = trace
    smoothed = gaussian_filter1d(filled, sigma=sigma, mode='wrap')
    if not finite.all():
        smoothed[~finite] = np.nan
    return smoothed


def plot_cell_polar(traces_per_config, configs,
                    pref_phase_per_config=None,
                    pref_state_per_config=None,
                    state_tuned_per_config=None,
                    phase_tuned_per_config=None,
                    r_per_config=None,
                    n_trials_per_config=None,
                    smooth_sigma=10,
                    title=None, out_path=None):
    """Polar overview of one neuron across configs.

    Parameters
    ----------
    traces_per_config : list of (360,) arrays
        Trial-averaged firing rate per config (in `configs` order). Use
        np.nan-padded zeros for empty configs; that subplot will just
        render empty.
    configs : list of str
        Config labels (e.g. '3-6-1-9').
    pref_phase_per_config : list of int or None
        per-config preferred phase (0..2) for shading; -1 to skip.
    pref_state_per_config : list of int or None
        per-config preferred state (0..3) marker; -1 to skip.
    state_tuned_per_config, phase_tuned_per_config : list of bool or None
        if provided, the subplot title gets 'S'/'P' badges.
    r_per_config : list of float or None
        if provided, the subplot title shows r per config.
    n_trials_per_config : list of int or None
        if provided, the subplot title shows # correct trials.
    title : str
        figure suptitle (e.g. cell label + ROI + summary stats).
    out_path : str or None
        if given, savefig there; otherwise return the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_configs = len(configs)
    n_cols = 3
    n_rows = int(np.ceil((n_configs + 1) / n_cols))   # +1 for mean subplot
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             subplot_kw=dict(projection='polar'),
                             squeeze=False)

    theta = np.linspace(0, 2 * np.pi, N_BINS_PER_TRIAL, endpoint=False)
    # Apply circular gaussian smoothing once so r_max and per-config plots
    # share the same scale.
    smoothed_traces = [smooth_circular(np.asarray(t, dtype=float), smooth_sigma)
                       for t in traces_per_config]
    finite_traces = [t for t in smoothed_traces
                     if np.isfinite(t).any()]
    if finite_traces:
        r_max_global = float(np.nanmax(np.stack(finite_traces, axis=0)))
        r_max_global = max(r_max_global, 1e-6)
    else:
        r_max_global = 1.0

    for i in range(n_configs):
        ax = axes.ravel()[i]
        trace = smoothed_traces[i]
        if not np.isfinite(trace).any():
            ax.text(0, 0, 'no data', ha='center', va='center',
                    transform=ax.transAxes)
            _polar_axes(ax)
            continue

        ax.plot(theta, trace, color='steelblue', lw=1.0)
        ax.fill_between(theta, 0, np.where(np.isfinite(trace), trace, 0),
                        color='steelblue', alpha=0.15, linewidth=0)
        ax.set_rmax(r_max_global * 1.05)

        # Pref-phase shading + pref-state tick.
        if pref_phase_per_config is not None:
            _shade_pref_phase(ax, pref_phase_per_config[i], r_max_global)
        if pref_state_per_config is not None:
            _mark_pref_state(ax, pref_state_per_config[i], r_max_global)
        _polar_axes(ax)

        # Build subplot title.
        bits = [configs[i]]
        if n_trials_per_config is not None:
            bits.append(f"n={n_trials_per_config[i]}")
        if r_per_config is not None and np.isfinite(r_per_config[i]):
            bits.append(f"r={r_per_config[i]:+.2f}")
        badges = []
        if state_tuned_per_config is not None and state_tuned_per_config[i]:
            badges.append('S')
        if phase_tuned_per_config is not None and phase_tuned_per_config[i]:
            badges.append('P')
        if badges:
            bits.append('[' + '/'.join(badges) + ']')
        ax.set_title('  '.join(bits), fontsize=10)

    # Cross-config mean (uses any finite trace).
    if finite_traces:
        mean_trace = np.nanmean(np.stack(finite_traces, axis=0), axis=0)
        ax = axes.ravel()[n_configs]
        ax.plot(theta, mean_trace, color='darkred', lw=1.8)
        ax.fill_between(theta, 0,
                        np.where(np.isfinite(mean_trace), mean_trace, 0),
                        color='darkred', alpha=0.15, linewidth=0)
        ax.set_rmax(r_max_global * 1.05)
        _polar_axes(ax)
        ax.set_title(f'mean across {len(finite_traces)} configs',
                     fontsize=11)

    # Hide the leftover empty axes.
    for ax in axes.ravel()[n_configs + 1:]:
        ax.axis('off')

    if title is not None:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return None
    return fig
