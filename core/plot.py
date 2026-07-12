# Copyright (c) 2022 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

from os.path import join
from os import makedirs
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from core.utils import remove_label_duplicated

from numbers import Number

def scattered_boxplot(ax, x, notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, showcaps=None, showbox=None,
                      showfliers="unif",
                      hide_points_within_whiskers=False,
                      boxprops=None, labels=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_ticks=True, autorange=False, zorder=None, *, data=None):
    if showfliers=="classic":
        classic_fliers=True
    else:
        classic_fliers=False
    bplot = ax.boxplot(x, notch=notch, sym=sym, vert=vert, whis=whis, positions=positions, widths=widths, patch_artist=patch_artist, bootstrap=bootstrap, usermedians=usermedians, conf_intervals=conf_intervals, meanline=meanline, showmeans=showmeans, showcaps=showcaps, showbox=showbox,
               showfliers=classic_fliers,
               boxprops=boxprops, labels=labels, flierprops=flierprops, medianprops=medianprops, meanprops=meanprops, capprops=capprops, whiskerprops=whiskerprops, manage_ticks=manage_ticks, autorange=autorange, zorder=zorder,data=data)
    N=len(x)
    datashape_message = ("List of boxplot statistics and `{0}` "
                             "values must have same the length")
    # check position
    if positions is None:
        positions = list(range(1, N + 1))
    elif len(positions) != N:
        raise ValueError(datashape_message.format("positions"))

    positions = np.array(positions)
    if len(positions) > 0 and not isinstance(positions[0], Number):
        raise TypeError("positions should be an iterable of numbers")

    # width
    if widths is None:
        widths = [np.clip(0.15 * np.ptp(positions), 0.15, 0.5)] * N
    elif np.isscalar(widths):
        widths = [widths] * N
    elif len(widths) != N:
        raise ValueError(datashape_message.format("widths"))

    if hide_points_within_whiskers:
        import matplotlib.cbook as cbook
        from matplotlib import rcParams
        if whis is None:
            whis = rcParams['boxplot.whiskers']
        if bootstrap is None:
            bootstrap = rcParams['boxplot.bootstrap']
        bxpstats = cbook.boxplot_stats(x, whis=whis, bootstrap=bootstrap,
                                       labels=labels, autorange=autorange)
    for i in range(N):
        if hide_points_within_whiskers:
            xi=bxpstats[i]['fliers']
        else:
            xi=x[i]
        if showfliers=="unif":
            jitter=np.random.uniform(-widths[i]*0.5,widths[i]*0.5,size=np.size(xi))
        elif showfliers=="normal":
            jitter=np.random.normal(loc=0.0, scale=widths[i]*0.1,size=np.size(xi))
        elif showfliers==False or showfliers=="classic":
            return
        else:
            raise NotImplementedError("showfliers='"+str(showfliers)+"' is not implemented. You can choose from 'unif', 'normal', 'classic' and False")

        plt.scatter(positions[i]+jitter,xi,alpha=0.5,marker="o", facecolors='none', edgecolors="k")
    return bplot

def ratings_vs_value_curve(pain_ratings, values_to_plot, interpolate=False):
    value_group_by_ratings = [0] * 11
    ratings_trials = [0] * 11
    for ratings, value_to_plot in zip(pain_ratings, values_to_plot):
        if value_to_plot == value_to_plot and not isinstance(value_to_plot, str): # avoid nan and string marker
            value_group_by_ratings[ratings] += value_to_plot
            ratings_trials[ratings] += 1

    for i in range(len(value_group_by_ratings)):
        if ratings_trials[i] > 0:
            value_group_by_ratings[i] /= ratings_trials[i]

    value_group_by_ratings = [y if x > 0 else np.nan for x, y in zip(ratings_trials, value_group_by_ratings)]

    #interpolate then normalise
    curve_pd = pd.DataFrame.from_dict({"ratings" : np.arange(11, dtype=np.int32), "rate" : value_group_by_ratings})
    if interpolate:
        curve_pd.interpolate(inplace=True, limit_area="inside")

    #return curve_pd["ratings"].to_list(), (curve_pd["rate"].to_numpy() / np.nanmean(curve_pd["rate"].to_numpy())).tolist()
    return curve_pd["ratings"].to_list(), curve_pd["rate"].to_list()

def all_curves(all_pain_ratings, all_value_to_plot, interpolate=False):
    individual_curves = [ratings_vs_value_curve(x, y, interpolate) for x, y in zip(all_pain_ratings, all_value_to_plot)]
    return individual_curves

def plot_all_subjects_pain_value_curve(subjects, curves, LABEL_MAP={}, COLOR_MAP={}, y_label="Distance effect"):
    total_curve = [0] * 11
    total_count = [0] * 11
    curve_val = [[] for i in range(11)]
    for name, curve in zip(subjects, curves):
        # if (LABEL_MAP[name] == '500ms stimulation time'):
        #     continue
        for r, c in zip(curve[0], curve[1]):
            if c == c:
                total_curve[r] += c
                total_count[r] += 1
                curve_val[r].append(c)
        if LABEL_MAP != {}:
            plt.plot(curve[0], curve[1], alpha=0.8, linestyle='dashed', label=LABEL_MAP[name], c=COLOR_MAP[name])

    average_curve = [tcu / tco for tcu, tco in zip(total_curve, total_count) if tco != 0][:11]
    plt.plot(average_curve, c='black', label='Average Curve Up to Pain Rating 10')
    scattered_boxplot(plt.gca(), curve_val, positions=[x for x in range(11)])

    plt.ylabel(y_label, fontsize=20)
    plt.xlabel("Visual Analog Pain Ratings", fontsize=20)

    X = np.array([x for x in range(len(average_curve))]).T
    X = sm.add_constant(X)
    results = sm.OLS(endog=average_curve, exog=X).fit()
    print(results.summary())

    handles, labels = remove_label_duplicated(plt.gca())
    plt.legend(handles, labels)
    plt.show()

def average_curves(curves, max_len=50):
    total_count = [0] * max_len
    total_curve = [[] for i in range(max_len)]

    for curve in curves:
        try:
            for i, val in enumerate(curve):
                total_count[i] += 1
                total_curve[i].append(val)
        except TypeError:
            print(curve)
            continue
    
    final_curve = [ np.mean(v) for v, c in zip(total_curve, total_count) if c != 0 ]
    standard_error_curve = [ np.std(v) for v, c in zip(total_curve, total_count) if c != 0 ]

    return np.array(final_curve), np.array(standard_error_curve)

import math

def significance_converter(data, raw=False):
    # * is p < 0.05
    # ** is p < 0.005
    # *** is p < 0.0005
    # etc.
    if raw:
        return 'p=' + "{:.3f}".format(data)
    if data < 0.001:
        text = '***'
    elif data < 0.01:
        text = '**'
    elif data < 0.05:
        text = '*'
    elif data < 0.1:
        text = '.'
    else:
        text = 'p=' + "{:.3f}".format(data)
    return text

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    """

    if type(data) is str:
        text = data
    else:
        text = significance_converter(data)

    print(center, height)
    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)
    # plt.text(mid[0], mid[1] + dh, 'p < ' + "{0:.4f}".format(math.ceil(data * 10000) / 10000), ha='center')

def generate_map_visualization(memory, player_location, pineapple_map, choice, expected,
                               gaze_direction, plot_no=0, save_dir='figures/realtime_visualization',
                               pain_value=-99999, choice_value=-99999, choice_values={}):
    x_coor = []
    z_coor = []
    colour = []

    for pineapple, location in pineapple_map.items():
        x_coor.append(location['x'])
        z_coor.append(location['z'])
        if pineapple == choice:
            if choice == expected:
                colour.append('yellow')
            else:
                colour.append('orange')
        elif pineapple in memory.keys():
            if pineapple == expected:
                colour.append('purple')
            else:
                colour.append('green')
        else:
            if pineapple == expected:
                colour.append('pink')
            else:
                if pineapple.endswith('G'):
                    colour.append('red')
                else:
                    colour.append('blue')
    
    x_coor.append(player_location[0])
    z_coor.append(player_location[2])
    colour.append('black')

    plt.figure(figsize=(10, 10))
    plt.scatter(x_coor, z_coor, color=colour)

    second_point = (player_location[0] + 10, player_location[2] + (gaze_direction[2] / gaze_direction[0]) * 10)
    plt.xlim((105, 135))
    plt.ylim((100, 130))
    plt.plot([player_location[0], second_point[0]], [player_location[2], second_point[1]], color='k', linestyle='dashed')

    makedirs(save_dir, exist_ok=True)

    plt.figtext(0.5, 0.9, "Pain value: " + str(pain_value),  fontsize=12, transform=plt.gcf().transFigure)
    plt.figtext(0.5, 0.8, "Chosen: " + choice + ' = ' + str(choice_value), fontsize=12, transform=plt.gcf().transFigure)
    for i, (p, v, opt) in enumerate(choice_values):
        plt.figtext(0.5, 0.7 - i * 0.1, p + ' = ' + str(v) + (' SUB' if opt else ' OPT'), fontsize=12, transform=plt.gcf().transFigure)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.5, right=0.5)
    
    plt.savefig(join(save_dir, str(plot_no) + '.png'))
    plt.clf()

def get_x_positions(ax, n_groups, n_hues, hue_width=0.5):
    # This calculation is an approximation based on the plot structure
    step = 1 / (n_hues + 1)
    positions = []
    for i in range(n_groups):
        for j in range(n_hues):
            positions.append(i + step * ((j - 1) + 0.5))
    return positions

def add_significance_bar_hue(ax, start_pos, end_pos, y, h, text, p_val, text_font=24, color='black', show_insignificance=False):
    if p_val > .1 and not show_insignificance:
        return
    ax.plot([start_pos, start_pos, end_pos, end_pos], [y, y+h, y+h, y], lw=1.5, c=color)
    ax.text((start_pos + end_pos) / 2, y + 1.5 * h, text, ha='center', va='bottom', fontsize=text_font, color=color)

def add_significance_bar_hue_inverted(ax, start_pos, end_pos, y, h, text, p_val, text_font=24, color='black', show_insignificance=False):
    if p_val > .1 and not show_insignificance:
        return
    ax.plot([start_pos, start_pos, end_pos, end_pos], [y, y-h, y-h, y], lw=1.5, c=color)
    ax.text((start_pos + end_pos) / 2, y - 1.5 * h, text, ha='center', va='bottom', fontsize=text_font, color=color)

def mne_mri_volume_plot(vol_estimate, fwd, fig, ax, title='', view='x', mni_coors=[-60, -40, -20, 0, 20, 40, 60]):
    lims = [2, 3, 4]
    clim = dict(kind="value", pos_lims=lims)

    # Modified from MNE-Python 1.7.0 mne.viz._3d.py:plot_volume_source_estimates
    from mne.viz._3d import _load_subject_mri, _process_clim, _separate_map, _get_map_ticks, _linearize_map, _crop_colorbar

    colormap = 'auto'
    transparent = 'auto'
    src = fwd['src']
    subject = src._subject
    img = vol_estimate.as_volume(src, mri_resolution=False)

    bg_img = _load_subject_mri("T1.mgz", vol_estimate, subject, None, "bg_img")

    print(bg_img)

    time_sl = slice(0, None)
    loc_idx, time_idx = np.unravel_index(
        np.abs(vol_estimate.data[:, time_sl]).argmax(), vol_estimate.data[:, time_sl].shape
    )

    from nilearn.image import index_img

    img_idx = index_img(img, time_idx)
    assert img_idx.shape == img.shape[:3]
    ydata = vol_estimate.data[loc_idx]

    mapdata = _process_clim(clim, colormap, transparent, vol_estimate.data, True)
    _separate_map(mapdata)
    diverging = "pos_lims" in mapdata["clim"]
    ticks = _get_map_ticks(mapdata)
    colormap, scale_pts = _linearize_map(mapdata)
    del mapdata

    ylim = [min((scale_pts[0], ydata.min())), max((scale_pts[-1], ydata.max()))]
    ylim = np.array(ylim) + np.array([-1, 1]) * 0.05 * np.diff(ylim)[0]
    dup_neg = False
    if vol_estimate.data.min() < 0:
        dup_neg = not diverging  # glass brain with signed data
    yticks = list(ticks)
    if dup_neg:
        yticks += [0] + list(-np.array(ticks))
    yticks = np.unique(yticks)
    del yticks

    vmax = scale_pts[-1]

    plot_kwargs = dict(
        threshold=None,
        axes=ax,
        resampling_interpolation="nearest",
        vmax=vmax,
        figure=fig,
        colorbar=True,
        bg_img=bg_img,
        cmap=colormap,
        black_bg=True,
        symmetric_cbar=True,
        display_mode=view
    )

    import warnings
    from functools import partial
    from nilearn.plotting import plot_stat_map

    def plot_and_correct(*args, **kwargs):
        ax.clear()
        if params.get("fig_anat") is not None and plot_kwargs["colorbar"]:
            params["fig_anat"]._cbar.ax.clear()
        with warnings.catch_warnings(record=True):  # nilearn bug; ax recreated
            warnings.simplefilter("ignore", DeprecationWarning)
            params["fig_anat"] = partial(plot_stat_map, **plot_kwargs)(*args, **kwargs)
        params["fig_anat"]._cbar.outline.set_visible(False)
        # for key in "xyz":
        #     params.update({"ax_" + key: params["fig_anat"].axes[key].ax})
        # Fix nilearn bug w/cbar background being white
        if plot_kwargs["colorbar"]:
            params["fig_anat"]._cbar.ax.set_facecolor("0.5")
            # adjust one-sided colorbars
            if not diverging:
                _crop_colorbar(params["fig_anat"]._cbar, *scale_pts[[0, -1]])
            params["fig_anat"]._cbar.set_ticks(params["cbar_ticks"])

    params = dict(
        img_idx=img_idx,
        fig=fig,
        cbar_ticks=ticks,
    )

    plot_and_correct(stat_map_img=params["img_idx"], cut_coords=mni_coors)
    params["fig_anat"].title(title, color='#ffffff', bgcolor='#000000')