import matplotlib.pyplot as plt


def plot_grt_profile(
    em,
    data1=["Prp", "Grs", "Sps"],
    data2=["Alm"],
    datalim1=None,
    datalim2=None,
    twin=True,
    title=None,
    filename=None,
    percents=False
):
    if percents:
        em = 100*em
        unit = " [%]"
    else:
        unit = " [prop.]"
    fig, ax1 = plt.subplots()
    if twin:
        ax1.set_xlabel(title)
        ax1.set_ylabel(" ".join(data1) + unit)
        ax1.locator_params(nbins=15, axis='x')
        h1 = ax1.plot(em.index, em[data1], marker="o", ms=4)
        if datalim1 is not None:
            ax1.set_ylim(datalim1[0], datalim1[1])
        ax2 = ax1.twinx()
        ax2.set_ylabel(" ".join(data2) + unit)
        h2 = ax2.plot(em.index, em[data2], marker="o", ms=4, color="red")
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
        ax1.set_xlabel(title)
        ax1.set_ylabel(" ".join(data1 + data2) + unit)
        h1 = ax1.plot(em.index, em[data1 + data2], marker="o", ms=4)
        plt.legend(
            h1,
            data1 + data2,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=3,
            ncol=len(data1 + data2),
            mode="expand",
            borderaxespad=0.0,
        )
    # Find at most 26 ticks on the x-axis at 'nice' locations
    xloc = plt.MaxNLocator(25)
    ax1.xaxis.set_major_locator(xloc)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()

