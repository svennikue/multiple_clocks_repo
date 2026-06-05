"""Helpers for sizing matplotlib figures to fit on an A4 publication page.

The core idea
-------------
Final figures end up as small subpanels of a half/full A4 page. When a saved
figure is shrunk to fit a subpanel, its fonts shrink with it. There are two
ways to keep the printed font at a target point size (e.g. 11pt Arial):

  (A) ``actual_size`` — save the figure at the dimensions it will actually
      occupy in print. Fonts stay at the target size; no shrinkage required.
      This is the recommended default and is what ``subpanel_figure`` produces.

  (B) ``full_page``   — save at full page width, then scale up the font so it
      prints at the target size after shrinking. Use ``scaled_font_for``.

Page constants assume an A4 layout (8.27 × 11.69 in) with 1.5 cm margins per
side (usable width ≈ 7.09 in). Override via ``page_width_in`` for other layouts.
"""

import matplotlib.pyplot as plt

A4_WIDTH_IN  = 8.27
A4_HEIGHT_IN = 11.69
CM_PER_IN    = 2.54

DEFAULT_MARGIN_CM     = 1.5
USABLE_PAGE_WIDTH_IN  = A4_WIDTH_IN  - 2 * DEFAULT_MARGIN_CM / CM_PER_IN   # ≈ 7.09
USABLE_PAGE_HEIGHT_IN = A4_HEIGHT_IN - 2 * DEFAULT_MARGIN_CM / CM_PER_IN   # ≈ 10.51

TARGET_FONT_PT     = 11
TARGET_FONT_FAMILY = 'Arial'

PUB_RC = {
    'font.family':     'sans-serif',
    'font.sans-serif': [TARGET_FONT_FAMILY, 'DejaVu Sans'],
    'font.size':       TARGET_FONT_PT,
    'axes.labelsize':  TARGET_FONT_PT,
    'axes.titlesize':  TARGET_FONT_PT + 1,
    'xtick.labelsize': TARGET_FONT_PT - 1,
    'ytick.labelsize': TARGET_FONT_PT - 1,
    'legend.fontsize': TARGET_FONT_PT - 1,
}


def _pub_rc(font_pt):
    return {
        'font.family':     'sans-serif',
        'font.sans-serif': [TARGET_FONT_FAMILY, 'DejaVu Sans'],
        'font.size':       font_pt,
        'axes.labelsize':  font_pt,
        'axes.titlesize':  font_pt + 1,
        'xtick.labelsize': max(font_pt - 1, 6),
        'ytick.labelsize': max(font_pt - 1, 6),
        'legend.fontsize': max(font_pt - 1, 6),
    }


def subpanel_figure(width_fracs, aspect_ratios=None, height_in=None,
                    page_width_in=USABLE_PAGE_WIDTH_IN,
                    target_font_pt=TARGET_FONT_PT,
                    extra_height_in=0.0):
    """Compute figsize & width_ratios so that each subpanel occupies the
    requested fraction of the printed A4 page.

    Drop the saved figure into the document at 100% and the fonts will render
    at ``target_font_pt`` automatically.

    Parameters
    ----------
    width_fracs : sequence of float
        Fraction of usable page width per subpanel (left → right). Need not
        sum to 1; e.g. ``[0.4, 0.3, 0.3]`` for the example fig1.
    aspect_ratios : sequence of float, optional
        height / width per subpanel. If supplied, figure height defaults to
        the max of the implied per-panel heights.
    height_in : float, optional
        Override figure height (inches).
    page_width_in : float
        Usable page width. Default = A4 (8.27 in) minus 1.5 cm margins/side.
    target_font_pt : float
        Desired printed font size.
    extra_height_in : float
        Padding for suptitles, colorbars, x-tick labels, etc.

    Returns
    -------
    dict with:
        figsize         (W, H) inches for ``plt.figure(figsize=...)``
        width_ratios    list of subpanel widths (inches) — pass via
                        ``gridspec_kw={'width_ratios': ...}``
        font_size       recommended base font size (= target_font_pt)
        rc              matplotlib rc dict ready for ``plt.rc_context(rc)``
        panel_widths_in per-subpanel widths in inches (sanity-check)
        page_width_in   usable page width used
    """
    width_fracs = list(width_fracs)
    panel_widths_in = [page_width_in * f for f in width_fracs]
    total_width_in = sum(panel_widths_in)

    if height_in is None:
        if aspect_ratios is None:
            aspect_ratios = [1.0] * len(width_fracs)
        height_in = max(w * a for w, a in zip(panel_widths_in, aspect_ratios))
    height_in = float(height_in) + float(extra_height_in)

    return {
        'figsize':         (total_width_in, height_in),
        'width_ratios':    panel_widths_in,
        'font_size':       target_font_pt,
        'rc':              _pub_rc(target_font_pt),
        'panel_widths_in': panel_widths_in,
        'page_width_in':   page_width_in,
    }


def scaled_font_for(width_frac, target_font_pt=TARGET_FONT_PT):
    """Mode (B): saving at full page width but the figure prints at
    ``width_frac`` of the page. Returns the font size to use in the saved
    figure so the printed font ends up at ``target_font_pt``.

    Example: a subpanel that prints at 40% of the page width needs
    ``scaled_font_for(0.4) == 27.5`` pt in the source figure.
    """
    if not 0 < width_frac <= 1.0:
        raise ValueError(f"width_frac must be in (0, 1], got {width_frac}")
    return target_font_pt / width_frac


def describe_figure(figsize, page_width_in=USABLE_PAGE_WIDTH_IN,
                    target_font_pt=TARGET_FONT_PT):
    """Print how a saved figure will print at 100% page width. Useful for
    sanity-checking ``figsize`` before saving.
    """
    W, H = figsize
    shrink = min(page_width_in / W, 1.0)
    apparent = target_font_pt * shrink
    print(f"  figsize: {W:.2f} × {H:.2f} in  → shrink {shrink:.2f}× "
          f"→ apparent font {apparent:.1f} pt (target {target_font_pt} pt)")
    if apparent < target_font_pt - 0.5:
        print(f"  → too small. Save at "
              f"({page_width_in:.2f}, {H * shrink:.2f}) in instead, "
              f"or set font_size = {target_font_pt / shrink:.1f} pt.")
