"""Method-schematic figures: unfolding code vs concurrent (DSR) code.

Builds the panels that explain, for one example task configuration, how the
two competing representational formats are simulated and how they turn into
a representational dissimilarity matrix:

  a) the ABCD loop cut into equal bins = equal angles into the future
  b) the example task on the 3x3 grid, with the executed trajectory
  c) the encodings across bins: position in sequence (A-D) and the physical
     locations actually visited
  d) the CONCURRENT (DSR) code read out from two different bins - the whole
     remaining trajectory, rolled so the present always comes first
  e) the UNFOLDING code read out from the same two bins - only the current
     location / current position in sequence
  f) similarity = counting how many encoded elements overlap. Two examples:
     one within task, and one across tasks where the current location is
     completely different yet the same locations occupy the same future
     lags, so the concurrent code is similar while the unfolding code is not
  g) the resulting model RDMs, for the single example task and across all
     tasks (the matrices that actually enter the RSA)

Two frameworks are supported, both faithful to the analysis they illustrate:

  'fmri'  - 8 bins per loop (A_path, A_reward, ... D_reward), each bin holding
            the modal path resampled to 12 steps -> 96-element DSR vector.
            Mirrors ``scripts/create_fMRI_model_RDMs_on_clean_beh.py``.
  'seeg'  - 12 bins per loop (4 states x 3 subgoal phases), each bin holding a
            single location -> 12-element DSR vector.
            Mirrors ``scripts/RSA_DSR_ROIs_simple.py``.

@author: Svenja Kuechenhoff
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Wedge, Circle

import mc


# ── project-wide colour conventions (see CLAUDE.md) ──────────────────────
STATE_COLORS = {'A': '#F15A29', 'B': '#F7931E', 'C': '#C7C6E2', 'D': '#6B60AA'}
LOCATION_COLORS = {
    1: '#0a607a', 2: '#7eb1c4', 3: '#b6d4e0',
    4: '#175e62', 5: '#5b9b8d', 6: '#c8e0d0',
    7: '#0e3d3a', 8: '#3d8b7d', 9: '#a7d9b2',
}
# Future lag / task angle: the cyclic "rainbow" used for the mPFC gradient
# figures — 0 deg yellow, 90 red, 180 blue, 270 green, wrapping to yellow.
FUTURE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    'future_angle',
    [(0.00, '#f7e34a'), (0.17, '#e8622c'), (0.25, '#c0272d'),
     (0.42, '#6b3f8f'), (0.50, '#3b53a4'), (0.67, '#2b8cbe'),
     (0.75, '#3fa34d'), (0.92, '#9ccb5a'), (1.00, '#f7e34a')])


def lag_colors(n_bins):
    """One colour per future lag, evenly spaced around the task circle."""
    return [FUTURE_CMAP(k / n_bins) for k in range(n_bins)]
MATCH_COLOR = '#0e3d3a'      # dark green: elements that overlap
MISMATCH_COLOR = '#e8e8e8'
OFF_COLOR = '#f5f5f5'

FONT_TITLE = 11     # panel headings
FONT_LABEL = 9      # axis labels — the floor for anything on an A4 page
FONT_TICK = 9

CM = 1 / 2.54       # matplotlib works in inches; every layout below is in cm

# Physical panel sizes (cm). These are the sizes of the PLOTTED BOX; tick
# labels, titles and colourbars live in the margins around them, so a panel
# dropped into an A4 figure keeps exactly these dimensions and its 9 pt text
# stays 9 pt.
RDM_CM = 4.0            # one model RDM
ENCODING_CM = 4.0       # panel c block (position-in-sequence + location)
CLOCK_CM = 4.0
GRID_CM = 4.0
W_CONCURRENT = 7.0      # the three code columns shared by panels d/e and f
W_LOCATION = 2.0
W_POSITION = 3.0
GAP_COL = 0.55
W_CODE_ROW = W_CONCURRENT + GAP_COL + W_LOCATION + GAP_COL + W_POSITION
X_LOCATION = W_CONCURRENT + GAP_COL
X_POSITION = X_LOCATION + W_LOCATION + GAP_COL
LEFT_LABEL_CM = 3.4     # room for row labels like "5-9-4-3  A reward"

FMRI_BIN_ORDER = ['A_path', 'A_reward', 'B_path', 'B_reward',
                  'C_path', 'C_reward', 'D_path', 'D_reward']
STATES = ['A', 'B', 'C', 'D']
PHASES = ['early', 'middle', 'late']


def rc_context():
    return plt.rc_context({
        'font.family': 'Arial',
        'font.size': FONT_LABEL,
        'axes.titlesize': FONT_TITLE,
        'axes.labelsize': FONT_LABEL,
        'xtick.labelsize': FONT_TICK,
        'ytick.labelsize': FONT_TICK,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


# ══════════════════════════════════════════════════════════════════════════
#  DATA: build the example task(s)
# ══════════════════════════════════════════════════════════════════════════

def _resample_path(path, target_len=12):
    """Repeat each step so a path of any length becomes ``target_len`` long.

    Identical to ``resample_locations`` in create_fMRI_model_RDMs_on_clean_beh.py.
    """
    n = len(path)
    reps = [target_len // n + (i < target_len % n) for i in range(n)]
    return np.repeat(path, reps)


def _downsample_mode(x, target_len):
    """Mode-downsample to ``target_len`` slots, using every input bin."""
    x = np.asarray(x, dtype=object)
    n = len(x)
    return np.array([
        Counter(x[(i * n) // target_len:((i + 1) * n) // target_len])
        .most_common(1)[0][0]
        for i in range(target_len)
    ])


def _example_from_bin_locs(name, bin_locs, bin_labels, bin_states,
                           framework, source, rewards=None):
    """Assemble the example-task dict shared by all panels."""
    bin_locs = np.asarray(bin_locs, dtype=int)
    n_bins, len_per_bin = bin_locs.shape
    traj = bin_locs.reshape(-1)
    dsr = np.stack([np.roll(traj, -pos * len_per_bin) for pos in range(n_bins)])
    if rewards is None:
        rewards = {}
        for st in STATES:
            idx = [i for i, (s, lab) in enumerate(zip(bin_states, bin_labels))
                   if s == st and ('reward' in lab or lab.endswith('early'))]
            if idx:
                rewards[st] = int(Counter(bin_locs[idx[0]]).most_common(1)[0][0])
    return dict(name=name, framework=framework, source=source,
                n_bins=n_bins, len_per_bin=len_per_bin,
                bin_labels=list(bin_labels), bin_states=list(bin_states),
                bin_locs=bin_locs, traj=traj, dsr=dsr, rewards=dict(rewards),
                deg_per_bin=360.0 / n_bins)


def align_to_state_A(locs, rewards, n_phases):
    """Rotate a binned loop so bin 0 is the bin the subject sits at reward A.

    The raw 360-bin sEEG loops are not all anchored at the same reward: the
    example configuration below starts on reward D. Labelling bin 0 as 'A'
    regardless is what made earlier schematics show locations that did not
    match where A-D actually were. We therefore pick the rotation that best
    lines the four state-onset bins up with the four rewarded locations, and
    report it, rather than assuming the anchor.
    """
    locs = np.asarray(locs).reshape(-1)
    n_bins = len(locs)
    targets = [rewards[s] for s in STATES]
    scores = []
    for r in range(n_bins):
        rolled = np.roll(locs, -r)
        hit = sum(int(rolled[k * n_phases] == t) for k, t in enumerate(targets))
        scores.append(hit)
    best = int(np.argmax(scores))
    return np.roll(locs, -best), best, int(scores[best])


# ── fMRI ────────────────────────────────────────────────────────────────
def _fmri_beh_path(sub, source_dir):
    return f"{source_dir}/data/derivatives/{sub}/beh/{sub}_beh_fmri_clean.csv"


def _fmri_modal_bin_locs(beh_df, task_key, len_per_bin=12):
    """Per bin: the most frequent path across repeats, resampled to 12.

    Exactly the ``EVs['location']`` construction of
    create_fMRI_model_RDMs_on_clean_beh.py.
    """
    out = []
    for b in FMRI_BIN_ORDER:
        d = beh_df[beh_df['unique_time_bin_type'] == f"{task_key}_{b}"]
        if len(d) == 0:
            return None
        paths = [d[d['repeat'] == r]['curr_loc'].to_numpy()
                 for r in range(0, int(d['repeat'].max()) + 1)]
        paths = [p for p in paths if len(p)]
        modal = np.array(Counter(map(tuple, paths)).most_common(1)[0][0])
        out.append(_resample_path(modal, len_per_bin))
    return np.array(out, dtype=int)


def build_fmri_examples(sub, source_dir, len_per_bin=12):
    """All task-halves-1 configurations of one fMRI subject, keyed by goal label."""
    beh = pd.read_csv(_fmri_beh_path(sub, source_dir))
    examples, by_key = {}, {}
    for task_key in sorted(beh['task_config_ex'].unique()):
        bin_locs = _fmri_modal_bin_locs(beh, task_key, len_per_bin)
        if bin_locs is None:
            continue
        bin_states = [b.split('_')[0] for b in FMRI_BIN_ORDER]
        rew_idx = [i for i, b in enumerate(FMRI_BIN_ORDER) if b.endswith('reward')]
        label = '-'.join(str(int(Counter(bin_locs[i]).most_common(1)[0][0]))
                         for i in rew_idx)
        ex = _example_from_bin_locs(label, bin_locs, FMRI_BIN_ORDER, bin_states,
                                    'fmri', f"{sub} / {task_key}")
        ex['task_key'] = task_key
        by_key[task_key] = ex
        examples.setdefault(label, ex)
    return examples, by_key


# ── sEEG ────────────────────────────────────────────────────────────────
SEEG_BEH_COLUMNS = ['rep_correct', 't_A', 't_B', 't_C', 't_D',
                    'loc_A', 'loc_B', 'loc_C', 'loc_D',
                    'rep_overall', 'new_grid_onset', 'session_no', 'grid_no',
                    'correct']


def _seeg_trials_for_config(config_str, ephys_dir):
    """All correct 360-bin location traces recorded for one configuration."""
    traces = []
    for sub_dir in sorted(d for d in os.listdir(ephys_dir)
                          if d.startswith('s') and d[1:].isdigit()):
        beh_dir = os.path.join(ephys_dir, sub_dir, 'cells_and_beh')
        beh_p = os.path.join(beh_dir, f"all_trial_times_{sub_dir[1:]}.csv")
        loc_p = os.path.join(beh_dir, 'locations.csv')
        if not (os.path.exists(beh_p) and os.path.exists(loc_p)):
            continue
        beh = pd.read_csv(beh_p, header=None)
        beh.columns = SEEG_BEH_COLUMNS
        cfg = (beh[['loc_A', 'loc_B', 'loc_C', 'loc_D']]
               .astype(int).astype(str).agg('-'.join, axis=1))
        idx = beh.index[(cfg == config_str) & (beh['correct'] == 1)]
        if len(idx) == 0:
            continue
        locs = pd.read_csv(loc_p, header=None)
        traces.append(locs.iloc[idx].to_numpy())
    if not traces:
        return None
    return np.vstack(traces)


def build_seeg_example(config_str, ephys_dir, n_bins=12):
    """Modal 12-bin trajectory of one sEEG configuration, pooled over trials."""
    stacked = _seeg_trials_for_config(config_str, ephys_dir)
    if stacked is None:
        return None
    mode_360 = np.array([Counter(col[~pd.isna(col)]).most_common(1)[0][0]
                         for col in stacked.T])
    locs = _downsample_mode(mode_360, target_len=n_bins).astype(int)
    rewards = {s: int(v) for s, v in zip(STATES, config_str.split('-'))}
    n_phases = n_bins // len(STATES)
    locs, shift, hits = align_to_state_A(locs, rewards, n_phases)
    bin_labels = [f"{s}_{p}" for s in STATES for p in PHASES[:n_phases]]
    bin_states = [s for s in STATES for _ in range(n_phases)]
    ex = _example_from_bin_locs(
        config_str, locs[:, None], bin_labels, bin_states, 'seeg',
        f"sEEG, modal path over {len(stacked)} correct trials",
        rewards=rewards)
    ex['bin_shift'] = shift
    ex['state_onsets_matched'] = hits
    return ex


# ══════════════════════════════════════════════════════════════════════════
#  small drawing helpers
# ══════════════════════════════════════════════════════════════════════════

def _grid_rc(loc):
    loc = int(loc)
    return (loc - 1) // 3, (loc - 1) % 3


def non_adjacent_steps(ex):
    """Trajectory steps that are not neighbours on the 3x3 grid.

    Binning a loop into a small number of bins can drop an intermediate
    location, which then shows up in panel b as a jump. Reported rather than
    silently smoothed, so the schematic is never prettier than the binning.
    """
    seq = [int(ex['traj'][0])]
    for v in ex['traj'][1:]:
        if int(v) != seq[-1]:
            seq.append(int(v))
    seq.append(seq[0])
    bad = []
    for a, b in zip(seq[:-1], seq[1:]):
        (r0, c0), (r1, c1) = _grid_rc(a), _grid_rc(b)
        if abs(r0 - r1) + abs(c0 - c1) > 1:
            bad.append((a, b))
    return seq, bad


def _strip(ax, vec, y=0.0, height=1.0, block=1, colors=None,
           edge='white', lw=0.4, block_lw=1.0, x0=0.0):
    """One horizontal row of coloured cells, with block separators."""
    vec = np.asarray(vec)
    colors = colors or LOCATION_COLORS
    if len(vec) > 24:
        lw = 0.18
    for i, v in enumerate(vec):
        ax.add_patch(Rectangle((x0 + i, y), 1, height,
                               facecolor=colors.get(int(v), OFF_COLOR),
                               edgecolor=edge, linewidth=lw))
    if block > 1:
        for i in range(0, len(vec) + 1, block):
            ax.plot([x0 + i, x0 + i], [y, y + height], color='#333333',
                    lw=block_lw, solid_capstyle='butt', zorder=5)
    ax.add_patch(Rectangle((x0, y), len(vec), height, fill=False,
                           edgecolor='#333333', lw=block_lw, zorder=5))


def _match_row(ax, a, b, y=0.0, height=0.45, block=1, x0=0.0):
    """Row marking, element by element, whether the two codes agree."""
    a, b = np.asarray(a), np.asarray(b)
    same = a == b
    for i, s in enumerate(same):
        ax.add_patch(Rectangle((x0 + i, y), 1, height,
                               facecolor=MATCH_COLOR if s else MISMATCH_COLOR,
                               edgecolor='white', linewidth=0.3))
    if block > 1:
        for i in range(0, len(a) + 1, block):
            ax.plot([x0 + i, x0 + i], [y, y + height], color='#333333',
                    lw=0.8, solid_capstyle='butt', zorder=5)
    return int(same.sum()), len(same)


def _lag_bar(ax, n_bins, block, y, height=0.28):
    """Colour bar coding how far into the future each block sits."""
    for k in range(n_bins):
        ax.add_patch(Rectangle((k * block, y), block, height,
                               facecolor=LAG_COLORS[min(k, len(LAG_COLORS) - 1)],
                               edgecolor='white', linewidth=0.4))


def _deg_labels(n_bins, deg_per_bin):
    return ['now'] + [f"+{int(round(k * deg_per_bin))}°"
                      for k in range(1, n_bins)]




# ══════════════════════════════════════════════════════════════════════════
#  LAYOUT HELPERS — everything is placed in absolute cm
# ══════════════════════════════════════════════════════════════════════════

def _add_ax_cm(fig, page, x_cm, y_top_cm, w_cm, h_cm):
    """Axes at an absolute position, measured in cm from the page's top-left."""
    fw, fh = page
    return fig.add_axes([x_cm / fw, (fh - y_top_cm - h_cm) / fh,
                         w_cm / fw, h_cm / fh])


def _cm_canvas(ax, w_cm, h_cm):
    """Turn an axes into a blank canvas whose data units ARE centimetres,
    with y running downwards from the top edge."""
    ax.set_xlim(0, w_cm)
    ax.set_ylim(h_cm, 0)
    ax.axis('off')
    ax.set_clip_on(False)


def _new_page(w_cm, h_cm):
    fig = plt.figure(figsize=(w_cm * CM, h_cm * CM))
    return fig, (w_cm, h_cm)


def _strip_cm(ax, vec, x0, w, y, h, block=1, colors=None, outline=True):
    """A row of coloured cells filling ``w`` cm, starting at ``x0`` cm."""
    vec = np.asarray(vec)
    n = len(vec)
    dx = w / n
    colors = colors or LOCATION_COLORS
    lw = 0.3 if n <= 24 else 0.12
    for i, v in enumerate(vec):
        ax.add_patch(Rectangle((x0 + i * dx, y), dx, h,
                               facecolor=colors.get(int(v), OFF_COLOR),
                               edgecolor='white', linewidth=lw))
    if block > 1:
        for i in range(block, n, block):
            ax.plot([x0 + i * dx] * 2, [y, y + h], color='#333333', lw=0.7,
                    solid_capstyle='butt', zorder=5)
    if outline:
        ax.add_patch(Rectangle((x0, y), w, h, fill=False, edgecolor='#333333',
                               lw=0.7, zorder=6))


def rdm_colour(value, lim=None, cmap='coolwarm'):
    """The colour this dissimilarity would have in the model RDMs."""
    lim = lim or {}
    vmin, vmax = lim.get('vmin', 0.0), lim.get('vmax', 1.0)
    span = (vmax - vmin) or 1.0
    return plt.get_cmap(cmap)(float(np.clip((value - vmin) / span, 0, 1)))


def _match_cm(ax, a, b, x0, w, y, h, block=1, lim=None):
    """Row marking element by element whether two codes agree.

    Coloured with the RDM colormap — an element that matches contributes 0 to
    the Hamming distance and gets the RDM's "similar" colour, one that differs
    contributes 1 and gets its "dissimilar" colour — so the bar reads in the
    same currency as panel g.
    """
    a, b = np.asarray(a), np.asarray(b)
    same = a == b
    dx = w / len(a)
    c_hit, c_miss = rdm_colour(0.0, lim), rdm_colour(1.0, lim)
    for i, ok in enumerate(same):
        ax.add_patch(Rectangle((x0 + i * dx, y), dx, h,
                               facecolor=c_hit if ok else c_miss,
                               edgecolor='none'))
    if block > 1:
        for i in range(block, len(a), block):
            ax.plot([x0 + i * dx] * 2, [y, y + h], color='white', lw=0.5,
                    solid_capstyle='butt', zorder=5)
    ax.add_patch(Rectangle((x0, y), w, h, fill=False, edgecolor='#999999',
                           lw=0.5, zorder=6))
    return int(same.sum()), len(same)


def _lag_bar_cm(ax, n_bins, x0, w, y, h):
    """Colour bar coding how far into the future each block sits."""
    cols = lag_colors(n_bins)
    dx = w / n_bins
    for k in range(n_bins):
        ax.add_patch(Rectangle((x0 + k * dx, y), dx, h, facecolor=cols[k],
                               edgecolor='white', linewidth=0.3))


def _ink(color, target=0.42):
    """Darken a colour until it is legible as text on white."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum <= target:
        return (r, g, b)
    f = target / lum
    return (r * f, g * f, b * f)


def _state_cells_cm(ax, state, x0, w, y, h, label=True):
    dx = w / len(STATES)
    for r_i, s in enumerate(STATES):
        on = s == state
        ax.add_patch(Rectangle((x0 + r_i * dx, y), dx, h,
                               facecolor=STATE_COLORS[s] if on else OFF_COLOR,
                               edgecolor='white', linewidth=0.5))
        if label:
            import matplotlib.colors as mcolors
            r, g, b = mcolors.to_rgb(STATE_COLORS[s])
            pale = 0.299 * r + 0.587 * g + 0.114 * b > 0.6
            ax.text(x0 + (r_i + 0.5) * dx, y + h / 2, s, ha='center',
                    va='center', fontsize=FONT_TICK, zorder=7,
                    color=((_ink(STATE_COLORS[s]) if pale else 'white')
                           if on else '#b0b0b0'),
                    fontweight='bold' if on else 'normal')
    ax.add_patch(Rectangle((x0, y), w, h, fill=False, edgecolor='#333333',
                           lw=0.7, zorder=6))


def _deg_labels(n_bins, deg_per_bin):
    return ['now'] + [f"+{int(round(k * deg_per_bin))}°"
                      for k in range(1, n_bins)]


def _bin_short(label):
    """'A_reward' -> 'at A', 'A_path' -> '→A', 'A_early' -> 'A early'."""
    st, kind = label.split('_')
    if kind == 'reward':
        return f"at {st}"
    if kind == 'path':
        return f"→{st}"
    return f"{st} {kind[:3]}"


def _thin(labels, n_slots, box_cm, min_cm=0.32):
    """Blank out tick labels that would sit closer together than 9 pt allows."""
    if box_cm / n_slots >= min_cm:
        return list(labels)
    step = int(np.ceil(min_cm / (box_cm / n_slots)))
    return [l if i % step == 0 else '' for i, l in enumerate(labels)]


# ══════════════════════════════════════════════════════════════════════════
#  PANELS
# ══════════════════════════════════════════════════════════════════════════

def panel_a_clock(ax, ex):
    """The loop cut into equal bins; each bin is an equal angle into the future."""
    n = ex['n_bins']
    cols = lag_colors(n)
    ax.set_aspect('equal')
    ax.axis('off')
    for k in range(n):
        th1 = 90 - (k + 1) * ex['deg_per_bin']      # 0 deg at the top,
        th2 = 90 - k * ex['deg_per_bin']            # running clockwise
        ax.add_patch(Wedge((0, 0), 1.0, th1, th2, facecolor=cols[k],
                           edgecolor='white', linewidth=0.8))
    ax.add_patch(Circle((0, 0), 0.47, facecolor='white', edgecolor='#333333',
                        linewidth=0.7))
    for deg, (x, y, ha, va) in zip(
            [0, 90, 180, 270],
            [(0, 1.10, 'center', 'bottom'), (1.10, 0, 'left', 'center'),
             (0, -1.10, 'center', 'top'), (-1.10, 0, 'right', 'center')]):
        ax.text(x, y, f"{deg}°", ha=ha, va=va, fontsize=FONT_TICK)
    # each state letter sits on the wedge where the subject is AT that reward
    for st in STATES:
        at_bins = [i for i, l in enumerate(ex['bin_labels'])
                   if l in (f'{st}_reward', f'{st}_early')]
        if not at_bins:
            continue
        ang = np.deg2rad(90 - (at_bins[0] + 0.5) * ex['deg_per_bin'])
        ax.text(0.73 * np.cos(ang), 0.73 * np.sin(ang), st, ha='center',
                va='center', fontsize=FONT_LABEL, fontweight='bold',
                color=STATE_COLORS[st])
    ax.text(0, 0, 'future\nlag', ha='center', va='center', fontsize=FONT_TICK)
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.42, 1.42)


def panel_b_grid(ax, ex):
    """The example configuration and the executed trajectory on the 3x3 grid."""
    ax.set_aspect('equal')
    ax.axis('off')
    for loc in range(1, 10):
        r, c = _grid_rc(loc)
        ax.add_patch(Rectangle((c, -r), 1, 1, facecolor=LOCATION_COLORS[loc],
                               edgecolor='white', linewidth=1.2))
        ax.text(c + 0.14, -r + 0.85, str(loc), ha='center', va='center',
                fontsize=FONT_TICK, color='white')
    seq, _ = non_adjacent_steps(ex)
    for a, b in zip(seq[:-1], seq[1:]):
        r0, c0 = _grid_rc(a)
        r1, c1 = _grid_rc(b)
        ax.add_patch(FancyArrowPatch((c0 + 0.5, -r0 + 0.5), (c1 + 0.5, -r1 + 0.5),
                                     arrowstyle='-|>', mutation_scale=8,
                                     linewidth=1.1, color='#333333',
                                     connectionstyle='arc3,rad=0.2',
                                     shrinkA=11, shrinkB=11, zorder=6))
    for st, loc in ex['rewards'].items():
        r, c = _grid_rc(loc)
        ax.add_patch(Circle((c + 0.5, -r + 0.5), 0.28, facecolor='white',
                            edgecolor=STATE_COLORS[st], linewidth=1.8, zorder=7))
        ax.text(c + 0.5, -r + 0.5, st, ha='center', va='center', zorder=8,
                fontsize=FONT_LABEL, fontweight='bold', color=STATE_COLORS[st])
    ax.set_xlim(-0.06, 3.06)
    ax.set_ylim(-2.06, 1.06)


def panel_c_encoding(ax_state, ax_loc, ex, box_cm=ENCODING_CM):
    """Position in sequence and physical location, bin by bin."""
    n, block = ex['n_bins'], ex['len_per_bin']
    for b_i, st in enumerate(ex['bin_states']):
        for r_i, s in enumerate(STATES):
            ax_state.add_patch(Rectangle((b_i * block, r_i), block, 1,
                                         facecolor=STATE_COLORS[s] if s == st else OFF_COLOR,
                                         edgecolor='white', linewidth=0.5))
    ax_state.set_xlim(0, n * block)
    ax_state.set_ylim(4, 0)
    ax_state.set_yticks(np.arange(4) + 0.5)
    ax_state.set_yticklabels(STATES, fontsize=FONT_TICK)
    ax_state.set_xticks([])
    ax_state.set_ylabel('position\nin seq.', fontsize=FONT_LABEL, labelpad=2)

    total = n * block
    for c_i, loc in enumerate(ex['traj']):
        ax_loc.add_patch(Rectangle((c_i, int(loc) - 1), 1, 1,
                                   facecolor=LOCATION_COLORS[int(loc)],
                                   edgecolor='none'))
    for i in range(block, total, block):
        ax_loc.plot([i, i], [0, 9], color='#bbbbbb', lw=0.5, zorder=4)
    ax_loc.set_xlim(0, total)
    ax_loc.set_ylim(9, 0)
    ax_loc.set_yticks(np.arange(9) + 0.5)
    ax_loc.set_yticklabels([str(i) for i in range(1, 10)], fontsize=FONT_TICK)
    ax_loc.set_ylabel('location', fontsize=FONT_LABEL, labelpad=2)
    ax_loc.set_xticks(np.arange(n) * block + block / 2)
    ax_loc.set_xticklabels(
        _thin([l.replace('_', ' ') for l in ex['bin_labels']], n, box_cm),
        fontsize=FONT_TICK, rotation=90)
    for ax in (ax_state, ax_loc):
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color('#333333')
        ax.tick_params(length=1.6, pad=1.5)


# ── d/e: the two codes side by side, read out from two bins ──────────────
DE_HEAD = 1.50          # column headers + the UNFOLDING bracket
DE_DEG = 0.34           # "now / +45 deg / ..." labels
DE_LAG = 0.18           # future-lag colour bar
DE_STRIP = 0.55         # a code strip
DE_BLOCKS = 0.38        # "at A / ->B" labels under a strip
DE_H = DE_HEAD + DE_DEG + 2 * (DE_LAG + DE_STRIP + DE_BLOCKS)


def panel_de_codes(ax, ex, bins):
    """Concurrent vs unfolding code, both read out from the same two bins."""
    n, block = ex['n_bins'], ex['len_per_bin']
    _cm_canvas(ax, W_CODE_ROW, DE_H)

    ax.text(W_CONCURRENT / 2, DE_HEAD - 0.30,
            f"CONCURRENT code  ({n * block} elements)", ha='center', va='bottom',
            fontsize=FONT_LABEL, style='italic')
    ax.text(X_LOCATION + W_LOCATION / 2, DE_HEAD - 0.30,
            f"current\nlocation ({block})", ha='center', va='bottom',
            fontsize=FONT_LABEL)
    ax.text(X_POSITION + W_POSITION / 2, DE_HEAD - 0.30,
            'position in\nsequence (4)', ha='center', va='bottom',
            fontsize=FONT_LABEL)
    ax.plot([X_LOCATION, X_POSITION + W_POSITION], [DE_HEAD - 1.02] * 2,
            color='#666666', lw=0.6, clip_on=False)
    ax.text((X_LOCATION + X_POSITION + W_POSITION) / 2, DE_HEAD - 1.07,
            'UNFOLDING code', ha='center', va='bottom', fontsize=FONT_LABEL,
            style='italic')

    for k, txt in enumerate(_thin(_deg_labels(n, ex['deg_per_bin']), n,
                                  W_CONCURRENT, min_cm=0.62)):
        ax.text((k + 0.5) * W_CONCURRENT / n, DE_HEAD + DE_DEG - 0.05, txt,
                ha='center', va='bottom', fontsize=FONT_TICK)

    y = DE_HEAD + DE_DEG
    for b_i in bins:
        _lag_bar_cm(ax, n, 0, W_CONCURRENT, y, DE_LAG)
        _strip_cm(ax, ex['dsr'][b_i], 0, W_CONCURRENT, y + DE_LAG, DE_STRIP,
                  block=block)
        _strip_cm(ax, ex['bin_locs'][b_i], X_LOCATION, W_LOCATION,
                  y + DE_LAG, DE_STRIP)
        _state_cells_cm(ax, ex['bin_states'][b_i], X_POSITION, W_POSITION,
                        y + DE_LAG, DE_STRIP)
        ax.text(-0.12, y + DE_LAG + DE_STRIP / 2,
                f"bin {b_i}   {ex['bin_labels'][b_i].replace('_', ' ')}",
                ha='right', va='center', fontsize=FONT_TICK, clip_on=False)
        y_lab = y + DE_LAG + DE_STRIP + DE_BLOCKS - 0.07
        per_state = n // len(STATES)
        if per_state == 1:
            # one block per state: the full "at A" / "->B" label fits
            for k in range(n):
                ax.text((k + 0.5) * W_CONCURRENT / n, y_lab,
                        _bin_short(ex['bin_labels'][(b_i + k) % n]),
                        ha='center', va='bottom', fontsize=FONT_TICK,
                        color='#555555')
        else:
            # several blocks per state: one letter, centred over its group
            for g in range(len(STATES)):
                k0 = g * per_state
                st = ex['bin_states'][(b_i + k0) % n]
                ax.text((k0 + per_state / 2) * W_CONCURRENT / n, y_lab, st,
                        ha='center', va='bottom', fontsize=FONT_TICK,
                        color='#555555')
                if g:
                    x_sep = k0 * W_CONCURRENT / n
                    ax.plot([x_sep, x_sep], [y_lab - 0.02, y_lab + 0.22],
                            color='#bbbbbb', lw=0.5)
        y += DE_LAG + DE_STRIP + DE_BLOCKS


# ── f: similarity by counting overlap ────────────────────────────────────
F_HEAD = 2.15
F_TITLE = 0.44
F_DEG = 0.32          # "0deg / 90deg / ..." labels above the lag bar
F_LAG = 0.20          # the rainbow future-lag bar
F_STRIP = 0.55        # a code strip
F_BINS = 0.34         # the task-state labels attached under a strip
F_OVERLAP = 0.26
F_NUMS = 0.46
F_SWATCH = 0.30      # the single RDM cell each comparison produces
F_ROW_H = (F_TITLE + F_DEG + F_LAG + 2 * (F_STRIP + F_BINS)
           + F_OVERLAP + F_NUMS)
F_GAP_ROW = 0.42


def _is_reward_bin(label):
    """True for the bin in which the subject is AT that state's reward."""
    return label.endswith('_reward') or label.endswith('_early')


def _mark_task_states(ax, ex, bin_idx, x0, w, y_strip, h_strip, y_lab, h_lab):
    """Frame the reward blocks in their state colour and label every block.

    This is the second time axis: the rainbow bar above says how far into the
    future a block sits, while these labels say WHICH part of the task that
    is — and they shift with the bin the code is read out from.
    """
    n = ex['n_bins']
    dx = w / n
    labels = [ex['bin_labels'][(bin_idx + k) % n] for k in range(n)]
    for k, lab in enumerate(labels):
        if _is_reward_bin(lab):
            ax.add_patch(Rectangle((x0 + k * dx, y_strip), dx, h_strip,
                                   fill=False,
                                   edgecolor=STATE_COLORS[lab.split('_')[0]],
                                   linewidth=1.6, zorder=8))
    if dx >= 0.70:
        # room for the full "at A" / "->B" label under every block
        for k, lab in enumerate(labels):
            st = lab.split('_')[0]
            ax.text(x0 + (k + 0.5) * dx, y_lab + h_lab - 0.06,
                    _bin_short(lab), ha='center', va='bottom',
                    fontsize=FONT_TICK,
                    color=(_ink(STATE_COLORS[st]) if _is_reward_bin(lab)
                           else '#666666'))
    else:
        # one letter per RUN of blocks belonging to the same state — found
        # from the labels, so it stays correct whatever bin the code starts at
        sts = [l.split('_')[0] for l in labels]
        k0 = 0
        for k in range(1, n + 1):
            if k == n or sts[k] != sts[k0]:
                ax.text(x0 + (k0 + k) / 2 * dx, y_lab + h_lab - 0.06, sts[k0],
                        ha='center', va='bottom', fontsize=FONT_TICK,
                        color=_ink(STATE_COLORS[sts[k0]]), fontweight='bold')
                if k < n:
                    ax.plot([x0 + k * dx] * 2,
                            [y_lab + 0.02, y_lab + h_lab - 0.08],
                            color='#cccccc', lw=0.5)
                k0 = k


def _similarity_row(ax, p, y0, n, block, limits=None):
    """One comparison: the two codes, their task states, and the overlap."""
    ex_a, ex_b = p['ex_a'], p['ex_b']
    ia, ib = p['bin_a'], p['bin_b']
    ax.text(0, y0 + F_TITLE - 0.08, p['title'], ha='left', va='bottom',
            fontsize=FONT_LABEL, fontweight='bold')

    # the rainbow bar: how far into the future each block sits
    y_deg = y0 + F_TITLE
    for k, txt in enumerate(_thin(_deg_labels(n, ex_a['deg_per_bin']), n,
                                  W_CONCURRENT, min_cm=0.62)):
        ax.text((k + 0.5) * W_CONCURRENT / n, y_deg + F_DEG - 0.06, txt,
                ha='center', va='bottom', fontsize=FONT_TICK)
    y = y_deg + F_DEG
    _lag_bar_cm(ax, n, 0, W_CONCURRENT, y, F_LAG)
    y += F_LAG

    for ex_, i_ in ((ex_a, ia), (ex_b, ib)):
        _strip_cm(ax, ex_['dsr'][i_], 0, W_CONCURRENT, y, F_STRIP, block=block)
        _strip_cm(ax, ex_['bin_locs'][i_], X_LOCATION, W_LOCATION, y, F_STRIP)
        _state_cells_cm(ax, ex_['bin_states'][i_], X_POSITION, W_POSITION,
                        y, F_STRIP)
        ax.text(-0.12, y + F_STRIP / 2,
                f"{ex_['name']}   {ex_['bin_labels'][i_].replace('_', ' ')}",
                ha='right', va='center', fontsize=FONT_TICK, clip_on=False)
        _mark_task_states(ax, ex_, i_, 0, W_CONCURRENT, y, F_STRIP,
                          y + F_STRIP, F_BINS)
        y += F_STRIP + F_BINS

    limits = limits or {}
    lim_c = limits.get('concurrent (DSR)')
    lim_l = limits.get('current location')
    lim_s = limits.get('position in sequence')
    m, tot = _match_cm(ax, ex_a['dsr'][ia], ex_b['dsr'][ib], 0, W_CONCURRENT,
                       y, F_OVERLAP, block=block, lim=lim_c)
    ml, _ = _match_cm(ax, ex_a['bin_locs'][ia], ex_b['bin_locs'][ib],
                      X_LOCATION, W_LOCATION, y, F_OVERLAP, lim=lim_l)
    sa, sb = ex_a['bin_states'][ia], ex_b['bin_states'][ib]
    _match_cm(ax, np.array([sa]), np.array([sb]), X_POSITION, W_POSITION,
              y, F_OVERLAP, lim=lim_s)
    ax.text(-0.12, y + F_OVERLAP / 2, 'overlap', ha='right', va='center',
            fontsize=FONT_TICK, color='#555555', clip_on=False)

    # each bar collapses into ONE cell of the model RDM — drawn at its end
    d_c, d_l = 1 - m / tot, 1 - ml / len(ex_a['bin_locs'][ia])
    d_s = 0.0 if sa == sb else 1.0
    for x0_, w_, d, lim in ((0, W_CONCURRENT, d_c, lim_c),
                            (X_LOCATION, W_LOCATION, d_l, lim_l),
                            (X_POSITION, W_POSITION, d_s, lim_s)):
        ax.add_patch(Rectangle((x0_ + w_ + 0.13, y), F_SWATCH, F_OVERLAP,
                               facecolor=rdm_colour(d, lim),
                               edgecolor='#333333', lw=0.5))
    y += F_OVERLAP
    for x_c, txt in (
            (W_CONCURRENT / 2,
             f"{m}/{tot} match = {100 * m / tot:.1f}%  →  d {d_c:.2f}"),
            (X_LOCATION + W_LOCATION / 2,
             f"{100 * ml / len(ex_a['bin_locs'][ia]):.0f}%  →  d {d_l:.2f}"),
            (X_POSITION + W_POSITION / 2,
             f"{'100' if sa == sb else '0'}%  →  d {d_s:.2f}")):
        ax.text(x_c, y + F_NUMS - 0.08, txt, ha='center', va='bottom',
                fontsize=FONT_TICK)


def panel_f_similarity(ax, ex, pairs, limits=None):
    """Similarity = count the overlapping elements, for contrasting pairs."""
    n, block = ex['n_bins'], ex['len_per_bin']
    h = F_HEAD + len(pairs) * F_ROW_H + (len(pairs) - 1) * F_GAP_ROW
    _cm_canvas(ax, W_CODE_ROW + F_SWATCH + 0.2, h)


    ax.text(W_CONCURRENT / 2, F_HEAD - 0.30,
            f"CONCURRENT code  ({n * block} elements)", ha='center',
            va='bottom', fontsize=FONT_LABEL, style='italic')
    ax.text(X_LOCATION + W_LOCATION / 2, F_HEAD - 0.30,
            f"current\nlocation ({block})", ha='center', va='bottom',
            fontsize=FONT_LABEL)
    ax.text(X_POSITION + W_POSITION / 2, F_HEAD - 0.30,
            'position in\nsequence (4)', ha='center', va='bottom',
            fontsize=FONT_LABEL)
    ax.plot([X_LOCATION, X_POSITION + W_POSITION], [F_HEAD - 1.02] * 2,
            color='#666666', lw=0.6, clip_on=False)
    ax.text((X_LOCATION + X_POSITION + W_POSITION) / 2, F_HEAD - 1.07,
            'UNFOLDING code', ha='center', va='bottom', fontsize=FONT_LABEL,
            style='italic')
    # spell out the two time axes the panel carries
    ax.text(0, 0.02,
            'ACROSS a row: how far into the future — the rainbow bar\n'
            'DOWN the rows: which time bin the code is read out from —\n'
            '                      the A–D labels under every strip shift with it\n'
            'The square after a bar = the RDM cell it makes.',
            ha='left', va='top', fontsize=FONT_TICK, color='#555555')
    for i, p in enumerate(pairs):
        _similarity_row(ax, p, F_HEAD + i * (F_ROW_H + F_GAP_ROW), n, block,
                        limits=limits)
    return h


# ── g: the resulting RDMs ────────────────────────────────────────────────
def _draw_rdm(ax, rdm, vmin=None, vmax=None, block=None, cmap='coolwarm',
              mask=None):
    M = np.array(rdm, dtype=float)
    if mask is not None:
        M = np.where(mask, M, np.nan)
    if vmin is None or vmax is None:
        lo, hi = np.nanpercentile(M, [2, 98])
        if hi - lo < 1e-9:
            lo, hi = np.nanmin(M) - 0.01, np.nanmax(M) + 0.01
        vmin, vmax = float(lo), float(hi)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('#ffffff')
    im = ax.imshow(M, cmap=cm, vmin=vmin, vmax=vmax, interpolation='nearest')
    if block:
        # grey, not white: with a mask the excluded cells are already white
        c_line = '#cccccc' if mask is not None else 'white'
        for i in range(block, M.shape[0], block):
            ax.axvline(i - 0.5, color=c_line, lw=0.4)
            ax.axhline(i - 0.5, color=c_line, lw=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color('#333333')
    ax.tick_params(length=1.6, pad=1.5)
    return im


ROUTE_ROW = 0.34
ROUTE_H = 0.36 + 2 * ROUTE_ROW + 0.34


def draw_route_strips(ax, ex_a, ex_b, w_cm, names=('task half 1',
                                                   'task half 2')):
    """The route each half walked, bin by bin, with the differences marked.

    The same reward configuration can be solved by more than one route, and
    the across-halves RDM compares one route against the other — which is why
    its diagonal need not be zero.
    """
    n, block = ex_a['n_bins'], ex_a['len_per_bin']
    _cm_canvas(ax, w_cm, ROUTE_H)
    diff = route_differences(ex_a, ex_b)
    y = 0.36
    for ex_, nm in ((ex_a, names[0]), (ex_b, names[1])):
        _strip_cm(ax, ex_['traj'], 0, w_cm, y, ROUTE_ROW, block=block)
        ax.text(-0.10, y + ROUTE_ROW / 2, nm, ha='right', va='center',
                fontsize=FONT_TICK, clip_on=False)
        y += ROUTE_ROW
    dx = w_cm / n
    for k in diff:
        ax.add_patch(Rectangle((k * dx, 0.36), dx, 2 * ROUTE_ROW, fill=False,
                               edgecolor='#c0392b', lw=1.2, zorder=9))
    if dx >= 0.70:
        for k in range(n):
            ax.text((k + 0.5) * dx, ROUTE_H - 0.02,
                    _bin_short(ex_a['bin_labels'][k]), ha='center',
                    va='bottom', fontsize=FONT_TICK,
                    color='#c0392b' if k in diff else '#666666')
    else:
        sts = [l.split('_')[0] for l in ex_a['bin_labels']]
        k0 = 0
        for k in range(1, n + 1):
            if k == n or sts[k] != sts[k0]:
                ax.text((k0 + k) / 2 * dx, ROUTE_H - 0.02, sts[k0],
                        ha='center', va='bottom', fontsize=FONT_TICK,
                        color=_ink(STATE_COLORS[sts[k0]]), fontweight='bold')
                k0 = k
    ax.text(0, 0.30, ('same route in both halves' if not diff else
                      f"different route at {', '.join(_bin_short(ex_a['bin_labels'][k]) for k in diff)}"),
            ha='left', va='bottom', fontsize=FONT_TICK,
            color='#666666' if not diff else '#c0392b')
    return diff


def used_cell_mask(ex, n_blocks=1, include_diagonal=False,
                   exclude_path_reward=True):
    """Which cells of a model RDM actually enter the searchlight regression.

    Two restrictions, both read off the RSA config
    (``diagonal_included: false``, ``masked_conds: true``):

    * only the upper triangle is regressed — the RDM is symmetric, so the
      lower half repeats the same numbers
      (``compute_hamming_distance`` returns ``np.triu_indices(n, k=1)``);
    * path bins are only compared with path bins and reward bins with reward
      bins (``make_category_masks(..., mask_only_path_rew_combos=True)`` keeps
      the ``same``-type cells), because a path bin and a reward bin differ in
      far more than the model being tested.
    """
    n = ex['n_bins'] * n_blocks
    keep = np.triu(np.ones((n, n), dtype=bool),
                   k=0 if include_diagonal else 1)
    if exclude_path_reward:
        kinds = np.array([l.split('_')[-1] for l in ex['bin_labels']] * n_blocks)
        keep &= kinds[:, None] == kinds[None, :]
    return keep


def _rdm_limits(across, pct=(2, 98)):
    """Colour limits taken once from the all-tasks RDMs and reused everywhere,
    so a given dissimilarity has one colour across the whole figure."""
    out = {}
    for name, d in across.items():
        M = np.asarray(d['rdm'], dtype=float)
        lo, hi = np.nanpercentile(M, pct)
        if hi - lo < 1e-9:
            lo, hi = float(np.nanmin(M)) - 0.01, float(np.nanmax(M)) + 0.01
        out[name] = {'vmin': float(lo), 'vmax': float(hi)}
    return out


def _rdm_colorbar(fig, page, im, x_cm, y_cm, h_cm, label):
    cax = _add_ax_cm(fig, page, x_cm, y_cm, 0.22, h_cm)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=FONT_TICK, labelpad=2)
    cb.ax.tick_params(labelsize=FONT_TICK, length=1.6, pad=1.5)
    cb.outline.set_linewidth(0.5)
    return cb


PARTNER_HALF = {'1_forw': '2_backw', '1_backw': '2_forw',
                '2_forw': '1_backw', '2_backw': '1_forw'}


def partner_example(ex, by_key):
    """The same goal configuration recorded in the OTHER task half.

    The RSA never compares a run with itself: ``X1_forw`` is paired with
    ``X2_backw`` and vice versa, so every similarity is between two
    independent runs. Returns None if the partner is not in ``by_key``.
    """
    tk = ex.get('task_key')
    if not tk:
        return None
    letter, half, direction = tk[0], tk[1], tk.split('_')[1]
    partner = PARTNER_HALF.get(f"{half}_{direction}")
    return by_key.get(f"{letter}{partner}") if partner else None


def route_differences(ex_a, ex_b):
    """Bins in which the two task halves did NOT walk the same route."""
    return [i for i in range(ex_a['n_bins'])
            if not np.array_equal(ex_a['bin_locs'][i], ex_b['bin_locs'][i])]


def within_half_rdms(ex):
    """Model RDMs comparing the example task's bins WITHIN one run.

    This is the didactic version — bin i against bin j of the same run — and
    it is what panel f counts out element by element. The RSA itself never
    does this (see :func:`across_half_rdms`), because two bins of the same run
    share noise; here it only has to explain the arithmetic.
    """
    n = ex['n_bins']
    dsr = 1 - np.array([[(ex['dsr'][i] == ex['dsr'][j]).mean() for j in range(n)]
                        for i in range(n)])
    loc = 1 - np.array([[(ex['bin_locs'][i] == ex['bin_locs'][j]).mean()
                         for j in range(n)] for i in range(n)])
    st = np.array([[0.0 if ex['bin_states'][i] == ex['bin_states'][j] else 1.0
                    for j in range(n)] for i in range(n)])
    return {'concurrent (DSR)': dsr, 'current location': loc,
            'position in sequence': st}


def across_half_rdms(ex_a, ex_b):
    """Model RDMs comparing task half 1 (rows) with task half 2 (columns).

    What the fMRI RSA actually regresses: every cell is a comparison between
    two independent runs, so no cell can be inflated by shared noise — and the
    diagonal is not zero whenever the subject solved the configuration by a
    different route the second time.

    Symmetrised as ``(M + M.T) / 2`` exactly as
    ``my_RSA.compute_hamming_distance`` does, so this is literally the example
    task's block of the all-tasks RDM; a cell therefore averages
    "half 1 bin i vs half 2 bin j" with "half 2 bin i vs half 1 bin j".
    """
    n = ex_a['n_bins']
    sym = lambda M: (M + M.T) / 2
    dsr = sym(1 - np.array([[(ex_a['dsr'][i] == ex_b['dsr'][j]).mean()
                             for j in range(n)] for i in range(n)]))
    loc = sym(1 - np.array([[(ex_a['bin_locs'][i] == ex_b['bin_locs'][j]).mean()
                             for j in range(n)] for i in range(n)]))
    st = sym(np.array([[0.0 if ex_a['bin_states'][i] == ex_b['bin_states'][j]
                        else 1.0 for j in range(n)] for i in range(n)]))
    return {'concurrent (DSR)': dsr, 'current location': loc,
            'position in sequence': st}


single_task_rdms = None  # superseded by within_half_rdms / across_half_rdms


def single_task_rdms_from_across(ex, across):
    """The example task's diagonal block, taken FROM the across-task RDMs.

    Computing the single-task matrix separately is not the same thing: the
    across-halves RDM compares task half 1 against task half 2, so if the
    subject walked a different route in the second half the two versions
    disagree. Slicing the block out guarantees that what panel g shows for the
    example task is literally the block it shows for that task in the
    all-tasks matrix.
    """
    n = ex['n_bins']
    out = {}
    for name, d in across.items():
        labels = d.get('block_labels') or []
        tasks = d.get('task_keys') or []
        idx = None
        for b, (lab, tk) in enumerate(zip(labels, tasks)):
            if lab == ex['name'] or tk == ex.get('task_key'):
                idx = b
                break
        if idx is None:
            continue
        sl = slice(idx * n, (idx + 1) * n)
        out[name] = d['rdm'][sl, sl]
    return out or None


STATE_AS_HAMMING = True
"""Score the state regressor by Hamming on its A-D label rather than by the
correlation distance the RSA uses.

``my_RSA.compute_crosscorr`` demeans each row before the cosine, so on a
4-element one-hot two different states come out at 1 - (-1/3) = 1.33 while two
identical ones come out at 0. That is the same binary same/different matrix
the Hamming version gives, just on a 0..1.33 scale instead of 0..1 — and the
RSA GLM z-scores every regressor, so the rescaling cannot change a result. In
the figure it does matter: on a shared colour scale the 1.0 of the other two
models would render lighter than the 1.33 of this one, so "completely
different" would look different in each panel. Comparing the state LABELS
(A vs B) instead puts all three models on one 0..1 scale.
"""


def _state_label_EV(state_EV):
    """One-hot state vectors -> single-character labels, for Hamming."""
    return {k: np.array([STATES[int(np.argmax(np.asarray(v)))]], dtype=object)
            for k, v in state_EV.items()}


def across_task_rdms_fmri(sub, source_dir, ev_string, label_map=None,
                          n_bins=8):
    """The RDMs that actually enter the searchlight, from the saved EV pickle."""
    import pickle
    p = (f"{source_dir}/data/derivatives/{sub}/beh/modelled_EVs/"
         f"{sub}_modelled_EVs_{ev_string}.pkl")
    with open(p, 'rb') as f:
        EVs = pickle.load(f)
    EV_keys = sorted(EVs['location'].keys())
    out = {}
    for model, nice in (('DSR', 'concurrent (DSR)'),
                        ('location', 'current location'),
                        ('state', 'position in sequence')):
        ev, hamming = EVs[model], None
        if model == 'state' and STATE_AS_HAMMING:
            ev, hamming = _state_label_EV(EVs['state']), ('state',)
        res = mc.analyse.my_RSA.build_across_halves_model_RDM(
            model, ev, EV_keys, hamming_models=hamming)
        if res is None:
            continue
        rdm, th1_keys, method, vrange = res
        task_keys = [th1_keys[b * n_bins].rsplit('_', 2)[0]
                     for b in range(rdm.shape[0] // n_bins)]
        out[nice] = dict(rdm=rdm, keys=th1_keys, method=method, vrange=vrange,
                         task_keys=task_keys,
                         block_labels=[(label_map or {}).get(t, t)
                                       for t in task_keys])
    return out


def across_task_rdms_seeg(examples):
    """Across-configuration model RDMs for the sEEG framework (96 x 96)."""
    order = list(examples)
    dsr = np.vstack([examples[c]['dsr'] for c in order])
    loc = np.vstack([examples[c]['bin_locs'] for c in order])
    st = np.vstack([np.array(examples[c]['bin_states'])[:, None] for c in order])
    out = {}
    for name, M in (('concurrent (DSR)', dsr), ('current location', loc),
                    ('position in sequence', st)):
        vec = mc.analyse.my_RSA.compute_hamming_distance(
            np.vstack([M, M]), include_diagonal=True, model_name=name)[0]
        out[name] = dict(rdm=mc.analyse.my_RSA._expand_triu_to_square(vec),
                         keys=order, method='hamming_distance',
                         vrange={'vmin': 0, 'vmax': 1},
                         task_keys=list(order), block_labels=list(order))
    return out


_METRIC_LABEL = {'hamming_distance': '1 − fraction identical',
                 'crosscorr': '1 − cosine similarity',
                 'categorical': 'same / different',
                 'distance': '|difference|'}


# ══════════════════════════════════════════════════════════════════════════
#  pair selection
# ══════════════════════════════════════════════════════════════════════════

def pick_pair(ex, others, mode, same_bin_type=True):
    """Find the most instructive across-task comparison.

    ``mode='future'``  : current locations do NOT overlap, concurrent code
                         overlaps as much as possible (concurrent says
                         'similar', unfolding says 'different').
    ``mode='present'`` : current locations overlap completely, concurrent code
                         overlaps as little as possible (the mirror case).
    """
    best = None
    for other in others:
        if other['name'] == ex['name']:
            continue
        for i in range(ex['n_bins']):
            for j in range(other['n_bins']):
                if same_bin_type and (i % 2) != (j % 2):
                    continue
                loc_sim = float((ex['bin_locs'][i] == other['bin_locs'][j]).mean())
                dsr_sim = float((ex['dsr'][i] == other['dsr'][j]).mean())
                if mode == 'future':
                    if loc_sim > 0:
                        continue
                    key = (dsr_sim,)
                else:
                    if loc_sim < 1:
                        continue
                    key = (-dsr_sim,)
                if best is None or key > best[0]:
                    best = (key, i, j, other, dsr_sim, loc_sim)
    if best is None:
        return None
    return dict(bin_a=best[1], bin_b=best[2], ex_b=best[3],
                dsr_sim=best[4], loc_sim=best[5])


def build_pairs(ex, others=None, within_pairs=((1, 3), (2, 4)),
                cross_task=False):
    """The comparisons shown in panel f.

    By default two WITHIN-task comparisons, one per comparison type that the
    RSA actually uses: reward-reward and path-path (across-phase pairs are
    excluded from the regression, see the DSR RSA section of the methods).
    Both show the same plan rolled by one state, read out from a different
    time bin. Set ``cross_task=True`` to append the two across-task cases
    found by :func:`pick_pair`.
    """
    kinds = {'reward': 'two reward bins', 'path': 'two path bins'}
    pairs = []
    for i, (a, b) in enumerate(within_pairs):
        lab = ex['bin_labels'][a].split('_')[-1]
        what = kinds.get(lab, f"two {lab} bins")
        n_roll = (b - a) % ex['n_bins']
        pairs.append(dict(
            ex_a=ex, bin_a=a, ex_b=ex, bin_b=b, kind=f'within_{lab}',
            title=f"{i + 1}  {what} of the same task — the same plan, "
                  f"rolled by {n_roll} bins"))
    if cross_task and others:
        for mode, txt in (('present', 'same place now, different plan'),
                          ('future', 'different place now, '
                                     'same future lags')):
            found = pick_pair(ex, others, mode)
            if found is not None:
                pairs.append(dict(
                    ex_a=ex, bin_a=found['bin_a'], ex_b=found['ex_b'],
                    bin_b=found['bin_b'], kind=mode, info=found,
                    title=f"{len(pairs) + 1}  {txt}"))
    return pairs


# ══════════════════════════════════════════════════════════════════════════
#  figures — every plotted box has the physical size declared at the top
# ══════════════════════════════════════════════════════════════════════════

def _save(fig, stem, show=False):
    """Save WITHOUT bbox_inches='tight' — the cm sizes must survive."""
    if stem:
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        fig.savefig(stem + '.pdf', dpi=300)
        fig.savefig(stem + '.png', dpi=400)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _title(fig, page, text, x_cm=0.05, y_cm=0.05):
    fig.text(x_cm / page[0], 1 - y_cm / page[1], text, fontsize=FONT_TITLE,
             ha='left', va='top')


def _rdm_row_figure(rdms, ex, block=None, block_labels=None, tick_labels=None,
                    vmin=None, vmax=None, metric_labels=None, title=None,
                    row_label=None, limits=None, axis_names=None, routes=None,
                    mask=None, show_mask=False):
    """Three RDMs of exactly RDM_CM x RDM_CM, side by side."""
    lab_w = 2.0 if tick_labels or block_labels else 0.35
    cb_w = 1.45
    col_w = lab_w + RDM_CM + cb_w
    x_lab = 1.05 if row_label else 0.0
    bottom = (2.4 if axis_names else 2.0) if tick_labels else 0.35
    top = (0.75 if title else 0.15) + 0.45 + (ROUTE_H + 0.45 if routes else 0)
    if axis_names:
        x_lab += 0.45
    n_panels = len(rdms) + (1 if show_mask else 0)
    page_w = x_lab + n_panels * col_w + 0.25
    page_h = top + RDM_CM + bottom
    fig, page = _new_page(page_w, page_h)
    if title:
        _title(fig, page, title)
    if routes:
        draw_route_strips(
            _add_ax_cm(fig, page, x_lab + lab_w + (col_w if show_mask else 0),
                       0.95, RDM_CM, ROUTE_H),
            routes[0], routes[1], RDM_CM)
    used = {}
    off = 0
    if show_mask and mask is not None:
        axm = _add_ax_cm(fig, page, x_lab + lab_w, top, RDM_CM, RDM_CM)
        axm.imshow(np.where(mask, 1.0, 0.0), interpolation='nearest',
                   cmap=mpl.colors.ListedColormap(['#ffffff', '#4d4d4d']),
                   vmin=0, vmax=1)
        if block:
            for i in range(block, mask.shape[0], block):
                axm.axvline(i - 0.5, color='#bbbbbb', lw=0.4)
                axm.axhline(i - 0.5, color='#bbbbbb', lw=0.4)
        axm.set_xticks([])
        axm.set_yticks([])
        for sp in axm.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color('#333333')
        n_tri = int(np.triu(np.ones_like(mask), k=1).sum())
        axm.set_title(f"cells regressed\n{int(mask.sum())} of {n_tri} "
                      f"upper-triangle cells", fontsize=FONT_LABEL, pad=3)
        if tick_labels is not None:
            k = len(tick_labels)
            axm.set_xticks(np.arange(k))
            axm.set_xticklabels(_thin(tick_labels, k, RDM_CM), rotation=90,
                                fontsize=FONT_TICK)
            axm.set_yticks(np.arange(k))
            axm.set_yticklabels(_thin(tick_labels, k, RDM_CM),
                                fontsize=FONT_TICK)
        if axis_names:
            axm.set_xlabel(axis_names[1], fontsize=FONT_TICK, labelpad=1)
            axm.set_ylabel(axis_names[0], fontsize=FONT_TICK, labelpad=1)
        off = 1
    for i0, (name, M) in enumerate(rdms.items()):
        i = i0 + off
        x = x_lab + i * col_w + lab_w
        ax = _add_ax_cm(fig, page, x, top, RDM_CM, RDM_CM)
        lim = (limits or {}).get(name, {})
        im = _draw_rdm(ax, M['rdm'] if isinstance(M, dict) else M,
                       vmin=lim.get('vmin', vmin), vmax=lim.get('vmax', vmax),
                       block=block, mask=mask)
        used[name] = {'vmin': float(im.get_clim()[0]),
                      'vmax': float(im.get_clim()[1])}
        ax.set_title(name, fontsize=FONT_LABEL, pad=3)
        if tick_labels is not None:
            k = len(tick_labels)
            ax.set_xticks(np.arange(k))
            ax.set_xticklabels(_thin(tick_labels, k, RDM_CM), rotation=90,
                               fontsize=FONT_TICK)
            if i == off:
                ax.set_yticks(np.arange(k))
                ax.set_yticklabels(_thin(tick_labels, k, RDM_CM),
                                   fontsize=FONT_TICK)
        elif block_labels is not None and i == off:
            nb = len(block_labels)
            ax.set_yticks(np.arange(nb) * block + block / 2 - 0.5)
            ax.set_yticklabels(_thin(block_labels, nb, RDM_CM),
                               fontsize=FONT_TICK)
        lbl = (metric_labels or {}).get(name, '1 − fraction identical')
        _rdm_colorbar(fig, page, im, x + RDM_CM + 0.18, top, RDM_CM, lbl)
        if axis_names:
            ax.set_xlabel(axis_names[1], fontsize=FONT_TICK, labelpad=1)
            if i == off:
                ax.set_ylabel(axis_names[0], fontsize=FONT_TICK, labelpad=1)
    if row_label:
        fig.text(0.12 / page_w, 1 - (top + RDM_CM / 2) / page_h, row_label,
                 fontsize=FONT_LABEL, ha='left', va='center', rotation=90)
    return fig, page, used


def save_panels(ex, others, across, label_map=None, within_pair=None,
                out_dir=None, prefix='panel', show=False,
                within_pairs=None, cross_task=False, ex_partner=None):
    """Render every panel at its declared physical size, ready for Illustrator."""
    n, block = ex['n_bins'], ex['len_per_bin']
    if within_pair is None:
        within_pair = (1, 3) if n >= 4 else (0, 1)
    if within_pairs is None:
        within_pairs = ((1, 3), (2, 4)) if n >= 6 else ((0, 1),)
    pairs = build_pairs(ex, others, within_pairs, cross_task=cross_task)
    made = {}

    with rc_context():
        # a — the loop cut into bins
        fig, page = _new_page(CLOCK_CM + 0.9, CLOCK_CM + 1.35)
        _title(fig, page, f"a  loop → {n} bins")
        panel_a_clock(_add_ax_cm(fig, page, 0.45, 0.85, CLOCK_CM, CLOCK_CM), ex)
        _save(fig, os.path.join(out_dir, f'{prefix}_a_bins'), show)
        made['a_bins'] = f"{CLOCK_CM}×{CLOCK_CM} cm"

        # b — the example configuration
        fig, page = _new_page(GRID_CM + 0.5, GRID_CM + 1.1)
        _title(fig, page, f"b  task {ex['name']}")
        panel_b_grid(_add_ax_cm(fig, page, 0.25, 0.85, GRID_CM, GRID_CM), ex)
        _save(fig, os.path.join(out_dir, f'{prefix}_b_task'), show)
        made['b_task'] = f"{GRID_CM}×{GRID_CM} cm"

        # c — what is encoded at each bin (block is ENCODING_CM x ENCODING_CM)
        h_state = (ENCODING_CM - 0.12) * 4 / 13
        h_loc = (ENCODING_CM - 0.12) * 9 / 13
        left, top = 1.65, 0.85
        fig, page = _new_page(left + ENCODING_CM + 0.25,
                              top + ENCODING_CM + 1.60)
        _title(fig, page, 'c  encoding per bin')
        ax_s = _add_ax_cm(fig, page, left, top, ENCODING_CM, h_state)
        ax_l = _add_ax_cm(fig, page, left, top + h_state + 0.12,
                          ENCODING_CM, h_loc)
        panel_c_encoding(ax_s, ax_l, ex)
        _save(fig, os.path.join(out_dir, f'{prefix}_c_encoding'), show)
        made['c_encoding'] = f"{ENCODING_CM}×{ENCODING_CM} cm"

        # d/e — concurrent and unfolding code, same two bins
        fig, page = _new_page(LEFT_LABEL_CM + W_CODE_ROW + 0.3, DE_H + 1.0)
        _title(fig, page, 'd  concurrent code      e  unfolding code')
        panel_de_codes(_add_ax_cm(fig, page, LEFT_LABEL_CM, 0.75,
                                  W_CODE_ROW, DE_H), ex, within_pair)
        _save(fig, os.path.join(out_dir, f'{prefix}_de_codes'), show)
        made['de_codes'] = (f"{W_CONCURRENT}+{W_LOCATION}+{W_POSITION} cm wide, "
                            f"{DE_H:.2f} cm high")

        # f — counting the overlap, in the RDM's own colours
        rdm_limits = _rdm_limits(across)
        f_h = F_HEAD + len(pairs) * F_ROW_H + (len(pairs) - 1) * F_GAP_ROW
        f_w = W_CODE_ROW + F_SWATCH + 0.2
        fig, page = _new_page(LEFT_LABEL_CM + f_w + 0.3, f_h + 1.0)
        _title(fig, page, 'f  similarity = fraction of elements that match')
        panel_f_similarity(_add_ax_cm(fig, page, LEFT_LABEL_CM, 0.75,
                                      f_w, f_h), ex, pairs,
                           limits=rdm_limits)
        _save(fig, os.path.join(out_dir, f'{prefix}_f_similarity'), show)
        made['f_similarity'] = f"{W_CODE_ROW} × {f_h:.2f} cm"

        # g3 — all task configurations (built first: it fixes the colour
        # limits every other RDM panel then reuses)
        nb = max(1, list(across.values())[0]['rdm'].shape[0] // n)
        blabels = (list(across.values())[0].get('block_labels')
                   or [str(k) for k in list(across.values())[0]['keys']])[:nb]
        metrics = {k: _METRIC_LABEL.get(v.get('method'), 'dissimilarity')
                   for k, v in across.items()}
        mask_all = used_cell_mask(ex, n_blocks=nb)
        fig, page, limits = _rdm_row_figure(
            across, ex, block=n, block_labels=blabels, metric_labels=metrics,
            mask=mask_all,
            title='g3  model RDMs — all task configurations, both halves; '
                  'only the regressed cells are shown')
        _save(fig, os.path.join(out_dir, f'{prefix}_g3_rdm_all_tasks'), show)
        made['g3_rdm_all_tasks'] = f"{RDM_CM}×{RDM_CM} cm per RDM"

        ticks = [l.replace('_', ' ') for l in ex['bin_labels']]
        # g1 — WITHIN one run: what panel f counts out. Explains the principle.
        fig, page, _ = _rdm_row_figure(
            within_half_rdms(ex), ex, limits=limits, metric_labels=metrics,
            tick_labels=ticks,
            title=f"g1  how similarity is computed — task {ex['name']}, "
                  f"bins of ONE run")
        _save(fig, os.path.join(out_dir, f'{prefix}_g1_rdm_within_half'), show)
        made['g1_rdm_within_half'] = f"{RDM_CM}×{RDM_CM} cm per RDM"

        # g2 — ACROSS the two runs: what the RSA actually regresses
        if ex_partner is not None:
            # sliced from the all-tasks RDMs so it IS that block, whatever
            # metric the pipeline scored each model with
            g2_mats = (single_task_rdms_from_across(ex, across)
                       or across_half_rdms(ex, ex_partner))
            mask_one = used_cell_mask(ex)
            fig, page, _ = _rdm_row_figure(
                g2_mats, ex, limits=limits,
                metric_labels=metrics, tick_labels=ticks,
                axis_names=('task half 1', 'task half 2'),
                routes=(ex, ex_partner), mask=mask_one, show_mask=True,
                title=f"g2  what the RSA regresses — task {ex['name']}, "
                      f"half 1 vs half 2, path-path and reward-reward only")
            _save(fig, os.path.join(out_dir, f'{prefix}_g2_rdm_across_halves'),
                  show)
            made['g2_rdm_across_halves'] = f"{RDM_CM}×{RDM_CM} cm per RDM"
    return pairs, made


def make_method_figure(ex, others, across, label_map=None, within_pair=None,
                       save_stem=None, show=False, within_pairs=None,
                       cross_task=False, ex_partner=None):
    """One overview page with every panel at its true physical size."""
    n, block = ex['n_bins'], ex['len_per_bin']
    if within_pair is None:
        within_pair = (1, 3) if n >= 4 else (0, 1)
    if within_pairs is None:
        within_pairs = ((1, 3), (2, 4)) if n >= 6 else ((0, 1),)
    pairs = build_pairs(ex, others, within_pairs, cross_task=cross_task)
    f_h = F_HEAD + len(pairs) * F_ROW_H + (len(pairs) - 1) * F_GAP_ROW

    h_state = (ENCODING_CM - 0.12) * 4 / 13
    h_loc = (ENCODING_CM - 0.12) * 9 / 13
    cb_w, lab_w = 1.45, 2.0
    col_w = lab_w + RDM_CM + cb_w
    page_w = max(LEFT_LABEL_CM + W_CODE_ROW + 0.3, 3 * col_w + 0.5)
    ys = dict(top=1.35)
    y = ys['top']
    ys['row1'] = y;            y += ENCODING_CM + 1.9 + 0.8
    ys['de'] = y + 0.55;       y += 0.55 + DE_H + 1.0
    ys['f'] = y + 0.55;        y += 0.55 + f_h + 1.0
    ys['g1'] = y + 0.95;       y += 0.95 + RDM_CM + 2.0 + 0.6
    if ex_partner is not None:
        ys['g2'] = y + 1.05 + ROUTE_H
        y += 1.05 + ROUTE_H + RDM_CM + 2.3 + 0.6
    ys['g3'] = y + 0.75;       y += 0.75 + RDM_CM + 0.9
    page_h = y

    with rc_context():
        fig, page = _new_page(page_w, page_h)
        fig.text(0.05 / page_w, 1 - 0.45 / page_h,
                 f"Unfolding vs concurrent code — example task {ex['name']} "
                 f"({ex['framework']}; {ex['source']})",
                 fontsize=FONT_TITLE, ha='left', va='center')

        # row 1: a, b, c
        x = 0.55
        _title(fig, page, f"a  loop → {n} bins", x, ys['row1'] - 0.55)
        panel_a_clock(_add_ax_cm(fig, page, x, ys['row1'], CLOCK_CM, CLOCK_CM), ex)
        x += CLOCK_CM + 1.4
        _title(fig, page, f"b  task {ex['name']}", x, ys['row1'] - 0.55)
        panel_b_grid(_add_ax_cm(fig, page, x, ys['row1'], GRID_CM, GRID_CM), ex)
        x += GRID_CM + 2.3
        _title(fig, page, 'c  encoding per bin', x - 1.7, ys['row1'] - 0.55)
        panel_c_encoding(
            _add_ax_cm(fig, page, x, ys['row1'], ENCODING_CM, h_state),
            _add_ax_cm(fig, page, x, ys['row1'] + h_state + 0.12,
                       ENCODING_CM, h_loc), ex)

        _title(fig, page, 'd  concurrent code      e  unfolding code',
               0.55, ys['de'] - 0.55)
        panel_de_codes(_add_ax_cm(fig, page, LEFT_LABEL_CM, ys['de'],
                                  W_CODE_ROW, DE_H), ex, within_pair)

        _title(fig, page, 'f  similarity = fraction of elements that match',
               0.55, ys['f'] - 0.55)
        panel_f_similarity(_add_ax_cm(fig, page, LEFT_LABEL_CM, ys['f'],
                                      W_CODE_ROW + F_SWATCH + 0.2, f_h),
                           ex, pairs, limits=_rdm_limits(across))

        _title(fig, page, f"g1  how similarity is computed: bins of ONE run "
                          f"(task {ex['name']})", 0.55, ys['g1'] - 0.95)
        if ex_partner is not None and 'g2' in ys:
            _title(fig, page, 'g2  what the RSA regresses: half 1 vs half 2, '
                              'path-path and reward-reward cells only',
                   0.55, ys['g2'] - ROUTE_H - 0.95)
        _title(fig, page, 'g3  all task configurations', 0.55, ys['g3'] - 0.55)
        nb = max(1, list(across.values())[0]['rdm'].shape[0] // n)
        blabels = (list(across.values())[0].get('block_labels')
                   or [str(k) for k in list(across.values())[0]['keys']])[:nb]
        limits = _rdm_limits(across)
        ticks = [l.replace('_', ' ') for l in ex['bin_labels']]

        mask_one, mask_all = used_cell_mask(ex), used_cell_mask(ex, n_blocks=nb)
        rows = [('g1', ys['g1'], within_half_rdms(ex), False)]
        if ex_partner is not None and 'g2' in ys:
            rows.append(('g2', ys['g2'],
                         single_task_rdms_from_across(ex, across)
                         or across_half_rdms(ex, ex_partner), True))
        for tag, y_row, mats, is_across in rows:
            if is_across:
                draw_route_strips(
                    _add_ax_cm(fig, page, lab_w, y_row - ROUTE_H - 0.35,
                               RDM_CM, ROUTE_H), ex, ex_partner, RDM_CM)
            for i, (name, M) in enumerate(mats.items()):
                xr = lab_w + i * col_w
                ax = _add_ax_cm(fig, page, xr, y_row, RDM_CM, RDM_CM)
                lim = limits.get(name, {'vmin': 0, 'vmax': 1})
                im = _draw_rdm(ax, M, vmin=lim['vmin'], vmax=lim['vmax'],
                               mask=mask_one if is_across else None)
                ax.set_title(name, fontsize=FONT_LABEL, pad=3)
                ax.set_xticks(np.arange(n))
                ax.set_xticklabels(_thin(ticks, n, RDM_CM), rotation=90,
                                   fontsize=FONT_TICK)
                if is_across:
                    ax.set_xlabel('task half 2', fontsize=FONT_TICK, labelpad=1)
                if i == 0:
                    ax.set_yticks(np.arange(n))
                    ax.set_yticklabels(_thin(ticks, n, RDM_CM),
                                       fontsize=FONT_TICK)
                    if is_across:
                        ax.set_ylabel('task half 1', fontsize=FONT_TICK,
                                      labelpad=1)
                _rdm_colorbar(fig, page, im, xr + RDM_CM + 0.18, y_row, RDM_CM,
                              _METRIC_LABEL.get(across[name].get('method'),
                                                'dissimilarity'))

        for i, (name, d) in enumerate(across.items()):
            xr = lab_w + i * col_w
            ax = _add_ax_cm(fig, page, xr, ys['g3'], RDM_CM, RDM_CM)
            lim = limits.get(name, {'vmin': 0, 'vmax': 1})
            im = _draw_rdm(ax, d['rdm'], block=n, mask=mask_all,
                           vmin=lim['vmin'], vmax=lim['vmax'])
            ax.set_title(name, fontsize=FONT_LABEL, pad=3)
            if i == 0:
                ax.set_yticks(np.arange(nb) * n + n / 2 - 0.5)
                ax.set_yticklabels(_thin(blabels, nb, RDM_CM),
                                   fontsize=FONT_TICK)
            _rdm_colorbar(fig, page, im, xr + RDM_CM + 0.18, ys['g3'], RDM_CM,
                          _METRIC_LABEL.get(d.get('method'), 'dissimilarity'))

        _save(fig, save_stem, show)
    return pairs, (page_w, page_h)
