import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import SpanSelector
from matplotlib.ticker import MaxNLocator


def plot_grt_profile(em, **kwargs):
    """Plot garnet profiles.

    Note:
        Endmembers have to be properly ordered.

    Args:
        em (pandas.DataFrame): garnet endmembers
        use_index (bool): When True, xticks are derived from DataFrame index and
            SpanSelector is activated, otherwise ticks
            are sequential. Default False
        overlap_isopleths(PTPS): When PTPS is passed, overlap_isopleths method is
            invoked with SpanSelector min-max range for garnet composition.
            Default None
        search_composition(PTPS): When PTPS is passed, search_composition method is
            invoked with SpanSelector mean values for garnet composition.
            Default None
        xticks_rotation (float, optional): Rotation of xtick labels. Default 0
        twin (bool, optional): When ``True``, the plot has two independent y-axes
            for better scaling. Endmembers must be separated into two groups using
            `data1` and `data2` args. When ``False`` both groups are plotted on same
            axes. Default ``True``
        data1 (list, optional): list of endmember names in first group. Default
            `["Prp", "Grs", "Sps"]`
        data2 (list, optional): list of endmember names in first group.
            Default `["Alm"]`
        datalim1 (tuple, optional): y-axis limits for first axis or auto when
            ``None``. Default ``None``
        datalim2 (tuple, optional): y-axis limits for second axis or auto when
            ``None``. Default ``None``
        omit (list, optional): index or list of indexes to be omitted from plot.
            Default None
        percents (bool): When ``True`` y-axes scale is percents, otherwise fraction
        xlabel (str, optional): label of the x-axis. Default ``None``
        filename (str, optional): When not ``None``, the plot is saved to file,
            otherwise the plot is shown.
        maxticks (int): maximum number of ticks on x-axis. Default 20

    """

    def onselect(xmin, xmax):
        sel = em.loc[math.ceil(xmin) : math.floor(xmax)]
        if not sel.empty:
            desc = sel.describe()
            if kwargs.get("overlap_isopleths", False):
                print(
                    f"""pt.overlap_isopleths(
    "g", "xFeX", ({desc["Alm"]["min"]}, {desc["Alm"]["max"]}),
    "g", "xMgX", ({desc["Prp"]["min"]}, {desc["Prp"]["max"]}),
    "g", "xMnX", ({desc["Sps"]["min"]}, {desc["Sps"]["max"]}),
    "g", "xCaX", ({desc["Grs"]["min"]}, {desc["Grs"]["max"]}),
)"""
                )
                plt.ion()
                kwargs.get("overlap_isopleths").overlap_isopleths(
                    "g",
                    "xFeX",
                    (desc["Alm"]["min"], desc["Alm"]["max"]),
                    "g",
                    "xMgX",
                    (desc["Prp"]["min"], desc["Prp"]["max"]),
                    "g",
                    "xMnX",
                    (desc["Sps"]["min"], desc["Sps"]["max"]),
                    "g",
                    "xCaX",
                    (desc["Grs"]["min"], desc["Grs"]["max"]),
                )
                plt.ioff()
            elif kwargs.get("search_composition", False):
                print(
                    f"""pt.search_composition(
    "g", "xFeX", {desc["Alm"]["mean"]},
    "g", "xMgX", {desc["Prp"]["mean"]},
    "g", "xMnX", {desc["Sps"]["mean"]},
    "g", "xCaX", {desc["Grs"]["mean"]},
)"""
                )
                plt.ion()
                kwargs.get("search_composition").search_composition(
                    "g",
                    "xFeX",
                    desc["Alm"]["mean"],
                    "g",
                    "xMgX",
                    desc["Prp"]["mean"],
                    "g",
                    "xMnX",
                    desc["Sps"]["mean"],
                    "g",
                    "xCaX",
                    desc["Grs"]["mean"],
                    isopleths=True,
                )
                plt.ioff()
            else:
                print(desc)

    data1 = kwargs.get("data1", ["Alm"])
    data2 = kwargs.get("data2", ["Prp", "Sps", "Grs"])
    datalim1 = kwargs.get("datalim1", None)
    datalim2 = kwargs.get("datalim2", None)
    filename = kwargs.get("filename", None)
    maxticks = kwargs.get("maxticks", 20)
    colors1 = list(mcolors.TABLEAU_COLORS.keys())[: len(data1)]
    colors2 = list(mcolors.TABLEAU_COLORS.keys())[len(data1) : len(data1) + len(data2)]
    fig, ax1 = plt.subplots()
    if kwargs.get("percents", False):
        multiple = 100
        unit = " [%]"
    else:
        multiple = 1
        unit = " [prop.]"
    if kwargs.get("use_index", False):
        xvals = em.index
        xlabel = kwargs.get("xlabel", "index")
    else:
        xvals = range(len(em))
        xlabel = kwargs.get("xlabel", "position")
    ax1.set_xlabel(xlabel)
    # omit
    if kwargs.get("omit", None) is not None:
        em.loc[kwargs.get("omit")] = np.nan
    if kwargs.get("twin", True):
        ax1.set_ylabel(" ".join(data1) + unit)
        h1 = ax1.plot(xvals, multiple * em[data1], marker="o", ms=4)
        for h, color in zip(h1, colors1):
            h.set_color(color)
        if datalim1 is not None:
            ax1.set_ylim(datalim1[0], datalim1[1])
        ax2 = ax1.twinx()
        ax2.set_ylabel(" ".join(data2) + unit)
        h2 = ax2.plot(xvals, multiple * em[data2], marker="o", ms=4)
        for h, color in zip(h2, colors2):
            h.set_color(color)
        if datalim2 is not None:
            ax2.set_ylim(datalim2[0], datalim2[1])
        plt.legend(
            h1 + h2,
            data1 + data2,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=3,
            ncol=len(data1 + data2),
            mode="expand",
            borderaxespad=0.0,
        )
    else:
        ax1.set_ylabel(" ".join(data1 + data2) + unit)
        h1 = ax1.plot(xvals, multiple * em[data1 + data2], marker="o", ms=4)
        for h, color in zip(h1, colors1 + colors2):
            h.set_color(color)
        if datalim1 is not None:
            ax1.set_ylim(datalim1[0], datalim1[1])
        plt.legend(
            h1,
            data1 + data2,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=3,
            ncol=len(data1 + data2),
            mode="expand",
            borderaxespad=0.0,
        )
    # Find at most maxticks ticks on the x-axis at 'nice' locations
    xloc = MaxNLocator(maxticks - 1, integer=True)
    ax1.xaxis.set_major_locator(xloc)
    ax1.tick_params(axis="x", labelrotation=kwargs.get("xticks_rotation", 0))
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    else:
        # if index used, spanselector created
        if kwargs.get("use_index", False):
            span = SpanSelector(
                ax1,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.5, facecolor="tab:blue"),
                interactive=True,
                drag_from_anywhere=True,
            )
        plt.show()
    plt.close(fig)
