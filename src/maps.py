from pathlib import Path
import importlib.resources
import json
import colorcet as cc  # noqa: F401
import periodictable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.path import Path as mplPath
from matplotlib.widgets import PolygonSelector, RectangleSelector
import matplotlib.patches as mpatches
import pandas as pd

# import seaborn as sns
import h5py
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import FunctionTransformer


class MapStore:
    """
    MapStore class provides file data storage in HDF5 format

    Args:
        h5file (str): string representation of storage filename

    Attributes:
        samples (list): List of sample names stored in MapStore
        h5file (str): string representation of storage filename
    """

    def __init__(self, h5file):
        self.h5file = h5file
        if not Path(self.h5file).exists():
            with h5py.File(self.h5file, "w") as hf:
                print("MapStore created")
        with h5py.File(self.h5file, "r") as hf:
            self.samples = list(hf.keys())

    def __repr__(self):
        return f"MapStore {Path(self.h5file).stem} with {len(self.samples)} samples"

    def info(self):
        """Returns pandas dataframe with storage information"""
        with h5py.File(self.h5file, "r") as hf:
            res = []
            for name in self.samples:
                samplestore = hf.get(name)
                nmaps = len(samplestore.get("maps").keys())
                shape = tuple(samplestore.attrs["shape"])
                nmasks = len(samplestore.get("masks").keys())
                if "kmeans" in samplestore:
                    kmeans = "Yes"
                else:
                    kmeans = "No"
                res.append([nmaps, shape[0], shape[1], nmasks, kmeans])
        return pd.DataFrame(
            res,
            index=self.samples,
            columns=["Maps", "Rows", "Cols", "Masks", "Clustered"],
        )

    def get_sample(self, name):
        """Retrieve sample from MapStore

        Args:
            name (str): Name of the sample. See HStore.samples
                for possible values

        """
        if name in self.samples:
            with h5py.File(self.h5file, "r") as hf:
                maps = {}
                masks = {}
                samplestore = hf.get(name)
                # attributes
                aspect = float(samplestore.attrs["aspect"])
                active_mask = samplestore.attrs["active_mask"]
                if active_mask == "":
                    active_mask = None
                # data
                mapstore = samplestore.get("maps")
                for element in mapstore.keys():
                    vals = np.array(mapstore.get(element))
                    maps[element] = vals
                maskstore = samplestore.get("masks")
                for mask in maskstore.keys():
                    vals = np.array(maskstore.get(mask))
                    masks[mask] = vals
                if "kmeans" in samplestore:
                    kmeanstore = samplestore.get("kmeans")
                    clusters = np.array(kmeanstore.get("clusters"))
                    centers = np.array(kmeanstore.get("centers"))
                    n_clusters = int(kmeanstore.attrs["n_clusters"])
                    legend = MapLegend(n_clusters=n_clusters)
                    store = json.loads(kmeanstore.attrs["store"])
                    for label, v in store.items():
                        legend.add(label, v["color"], v["values"])
                else:
                    clusters = None
                    centers = None
                    legend = None
            return Mapset(
                maps,
                name=name,
                aspect=aspect,
                masks=masks,
                active_mask=active_mask,
                clusters=clusters,
                centers=centers,
                legend=legend,
            )
        else:
            print(f"Mapset {name} is not in MapStore.")

    def save_sample(self, sample, overwrite=False):
        """Store sample in MapStore

        Args:
            sample (edstools.Mapset): Mapset instance to be stored
                in MapStore
        overwrite (bool, optional): Whether to overwrite existing
            sample. Default False

        """
        if sample.name in self.samples:
            if overwrite:
                with h5py.File(self.h5file, "a") as hf:
                    del hf[sample.name]
                    self.samples = list(hf.keys())
            else:
                print(f"Mapset {sample.name} is already in MapStore.")
        if sample.name not in self.samples:
            with h5py.File(self.h5file, "a") as hf:
                samplestore = hf.create_group(sample.name)
                samplestore.attrs["shape"] = sample.shape
                samplestore.attrs["aspect"] = sample.aspect
                if sample.active_mask is None:
                    samplestore.attrs["active_mask"] = ""
                else:
                    samplestore.attrs["active_mask"] = sample.active_mask
                mapstore = samplestore.create_group("maps")
                for element in sample.maps:
                    mapstore.create_dataset(
                        element, data=sample[element], compression="gzip", dtype=int
                    )
                maskstore = samplestore.create_group("masks")
                for mask in sample.masks:
                    maskstore.create_dataset(
                        mask, data=sample.get_mask(mask), compression="gzip", dtype=bool
                    )
                if sample.clusters is not None:
                    kmeanstore = samplestore.create_group("kmeans")
                    kmeanstore.create_dataset(
                        "clusters", data=sample.clusters, compression="gzip", dtype=int
                    )
                    kmeanstore.create_dataset(
                        "centers", data=sample.centers, compression="gzip", dtype=float
                    )
                    kmeanstore.attrs["n_clusters"] = sample.legend.n_clusters
                    kmeanstore.attrs["store"] = json.dumps(sample.legend.store)

                self.samples = list(hf.keys())
                print(f"{sample.name} saved.")

    def delete_sample(self, name):
        """Delete sample from MapStore

        Args:
            name (str): Name of the sample to be deleted

        """
        if name not in self.samples:
            print(f"Mapset {name} is not in MapStore.")
        else:
            with h5py.File(self.h5file, "a") as hf:
                del hf[name]
                self.samples = list(hf.keys())
                print(f"Mapset {name} deleted.")

    def update_legend(self, sample):
        """Update stored legend to the one from the sample.

        Note: Provided smaple must exists in MapStore

        Args:
            sample (edstools.Mapset): Mapset instance to be stored
                in MapStore

        """
        if sample.name in self.samples:
            with h5py.File(self.h5file, "a") as hf:
                samplestore = hf.get(sample.name)
                if "kmeans" in samplestore:
                    kmeanstore = samplestore.get("kmeans")
                    kmeanstore.attrs["n_clusters"] = sample.legend.n_clusters
                    kmeanstore.attrs["store"] = json.dumps(sample.legend.store)
                    print(f"Legend for {sample.name} updated.")
                else:
                    print(f"No clustering stored for sample {sample.name}.")
        else:
            print(f"Mapset {sample.name} not exists in MapStore.")

    @classmethod
    def from_examples(cls, example=None):
        """Get example MapStore

        Args:
            example (str, optional): Name of example. When None, available examples
                are printed. Default is `None`

        Returns:
            Oxides: datatable

        """
        resources = importlib.resources.files("epmatools") / "data"
        datapath = resources / "maps"
        if example is None:
            print(f"Available examples: {[f.stem for f in datapath.glob('*.h5')]}")
        else:
            fname = (datapath / example).with_suffix(".h5")
            assert fname.exists(), "Example {example} do not exists."
            return cls(str(fname))


class Mapset:
    """
    A class to represent a collection of spatially overlapping EDS maps.

    Note: All maps must have same spatial resolution. Most of the arguments
        are accessible as instance properties.

    Maps data could be accessed by different ways. Raw data (ignoring mask)
    could be obtained by name of the map used as index of sample instance::

        >>> s['Si']

    Masked data (i.e. np.nan values represents masked values) could be
    obtained by method `get_map`::

        >>> s.get_map('Si')

    Flattened array of non-masked values could be obtained by `values`
    method::

        >>> s.values('Si')

    Args:
        maps (dict): Collection of maps stored in dictionary.
            The key is the name of the map (e.g. element name)
            and map itself have to be 2D numpy.ndarray.
        name (str, optional): Name of the sample. Default is 'default'.
        aspect (float): Aspect of the axis scaling, i.e. the ratio of y-unit to
                x-unit. Default 1
        masks (dict): Collection of masks stored in dictionary. All masks
            are 2D boolean numpy arrays. Values corresponding to True are masked.
        active_mask (str): Name of active mask or None. Default value is None.
        clusters (numpy.array): Clusters from KMeans clustering. Defaut is None
        centers (numpy.array): Centers from KMeans clustering. Defaut is None
        labels (numpy.array): Labels from Agglomerative clustering. Defaut is None
        img (numpy.array): 2D array of Agglomerative labels. Defaut is None
        legend (edstool.MapLegend): Legend for phase map. Defaut is None
        default_cmap (str): Name of default matplotlib colormap. Default is 'inferno'.
        figsize (tuple): Default figure size. Default is (8, 6).
        transpose (bool): Whether to show map transposed. Default is False.

    Attributes:
        df (pandas.DataFrame): All nonmasked values from all maps arranged in
            Pandas DataFrame.
        element_df (pandas.DataFrame): All nonmasked values from element maps
            arranged in Pandas DataFrame.
        maps (list): List of all available maps in sample.
        masks (list): List of all available masks in sample.
        active_mask (str): Name of active mask or None.
        element_maps (list): List of all elemental maps in sample.
        total_counts (nump.ndarray): 2d array of total elemental count. It is
            sum of all elemental maps.
        label_df (pandas.DataFrame): All values from maps aranged in pandas
            DataFrame with values of classes from Agglomerative clustering.
        shape (tuple): (rows, cols) tuple of maps shape.
        name (str, optional): Name of the sample.
        aspect (float): Aspect of the axis scaling, i.e. the ratio of y-unit to
                x-unit.
        mask (numpy.ndarray): Active data mask as 2D boolean numpy array.
        clusters (numpy.array): Clusters from KMeans clustering.
        centers (numpy.array): Centers from KMeans clustering.
        labels (numpy.array): Labels from Agglomerative clustering.
        img (numpy.array): 2D array of Agglomerative labels.
        legend (edstool.MapLegend): Legend for phase map.
        default_cmap (str): Name of default matplotlib colormap.
        figsize (tuple): Default figure size.
        transpose (bool): Whether to show map transposed.

    """

    def __init__(self, maps, **kwargs):
        assert isinstance(maps, dict), "Argument maps must be dictionary"
        # check shapes
        itmap = iter(maps)
        self.shape = maps[next(itmap)].shape
        for el in itmap:
            assert maps[el].shape == self.shape, "Shape of all maps must be same"
        self.__maps = maps
        # defaults
        self.__masks = kwargs.get("masks", {})
        self.__active_mask = None
        if "active_mask" in kwargs:
            if kwargs["active_mask"] in self.__masks:
                self.__active_mask = kwargs["active_mask"]
        self.name = kwargs.get("name", "default")
        self.aspect = kwargs.get("aspect", 1)
        self.clusters = kwargs.get("clusters", None)
        self.centers = kwargs.get("centers", None)
        self.legend = kwargs.get("legend", MapLegend())
        self.default_cmap = kwargs.get("cmap", "inferno")
        self.figsize = kwargs.get("figsize", (8, 6))
        self.transpose = kwargs.get("transpose", False)
        if self.clusters is not None:
            self.aggclusters()

    def __repr__(self):
        return (
            f"Mapset {self.name} with {len(self.__maps)} {self.shape} maps. "
            f"Active mask {self.active_mask}."
        )

    def __getitem__(self, name):
        if name in self:
            return self.__maps[name].astype("float")
        else:
            print(f"There is no map {name} in mapset")

    def __setitem__(self, name, data):
        assert isinstance(data, np.ndarray), "Argument must be NumPy array"
        assert (
            self.shape == data.shape
        ), f"Data shape do not match mapset shape {self.shape}"
        self.__maps[name] = data

    def __delitem__(self, name):
        if name in self:
            del self.__maps[name]
        else:
            print(f"There is no map {name} in mapset")

    def __contains__(self, name):
        return name in self.__maps

    def __len__(self):
        return len(self.__maps)

    def values(self, expr=None):
        """Returns flattened array of nonmasked values.

        Args:
            expr (str, optional): Name of the map or expression using names
                of the map, e.g. 'Si', 'Fe/(Fe+Mg)', 'Na+Ca+K'. If None, the
                total counts are used.

        """
        dt = self.get_map(expr)
        return dt[np.invert(self.mask)].flatten()

    @property
    def active_mask(self):
        return self.__active_mask

    @active_mask.setter
    def active_mask(self, name):
        if (name in self.__masks) or (name is None):
            self.__active_mask = name
        else:
            print(f"Mask {name} not exists in sample.")

    @property
    def df(self):
        dt = np.array([self.values(el) for el in self.maps]).T
        return pd.DataFrame(dt, columns=self.maps)

    @property
    def element_df(self):
        dt = np.array([self.values(el) for el in self.element_maps]).T
        return pd.DataFrame(dt, columns=self.element_maps)

    @property
    def maps(self):
        return list(self.__maps.keys())

    @property
    def element_maps(self):
        all_elements = set([el.symbol for el in periodictable.elements][1:])
        return list(all_elements.intersection(self.maps))

    @property
    def total_counts(self):
        tc = np.zeros_like(self.mask, dtype=float)
        for el in self.element_maps:
            tc += self[el]
        return tc

    def get_map(self, expr=None):
        """Returns map as 2D numpy.ndarray with np.nan as masked values.

        Args:
            expr (str, optional): Name of the map or expression using names
                of the map, e.g. 'Si', 'Fe/(Fe+Mg)', 'Na+Ca+K'. If None, the
                total counts are used.

        """
        if expr in self:
            dt = self[expr]
            dt[self.mask] = np.nan
        elif expr is None:
            dt = self.total_counts
            dt[self.mask] = np.nan
        else:
            for el in self.maps:
                expr = expr.replace(el, f'self.get_map("{el}")')
            try:
                dt = eval(expr)
            except NameError:
                print(f"Expression {expr} could not be evaluated")
                print(f"Available maps: {self.maps}")
                return
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
            # dt = interpolate_missing_pixels(
            #    dt, np.isnan(dt) | np.isinf(dt), method="nearest"
            # )
            dt[np.isinf(dt)] = np.nan
        return dt

    @property
    def mask(self):
        if self.active_mask is None:
            return np.full(self.shape, False)
        else:
            return self.__masks[self.active_mask]

    @property
    def masks(self):
        return list(self.__masks.keys())

    def clear_mask(self):
        """Remove active mask."""
        self.active_mask = None

    def get_mask(self, name):
        """Get mask by name.

        Args:
            name (str): Name of the mask

        """
        if name in self.__masks:
            return self.__masks[name].copy()
        else:
            print(f"Mask {name} not exists in sample")
            print(f"Available masks: {self.masks}")

    def add_mask(self, name, mask, **kwargs):
        """Add mask to sample masks

        Args:
            name (str): Name of the mask
            mask (numpy.ndarray): 2D boolean numpy array
            activate (bool): Whether the mask should be set as active. Default
                is True

        """
        assert mask.dtype == bool, "Mask must be boolean array"
        assert (
            self.shape == mask.shape
        ), f"Mask shape do not match mapset shape {self.shape}"
        if name in self.__masks:
            if kwargs.get("overwrite", False):
                self.__masks[name] = mask
                if kwargs.get("activate", True):
                    self.active_mask = name
            else:
                print(f"Mask {name} already exists. Use overwrite=True to overwrite")
        else:
            self.__masks[name] = mask
            if kwargs.get("activate", True):
                self.active_mask = name

    def remove_mask(self, name):
        if name in self.__masks:
            del self.__masks[name]
            if self.active_mask == name:
                self.active_mask = None
        else:
            print(f"Mask {name} not found in sample masks")

    def draw_mask(self, name, expr=None, **kwargs):
        """Create mask by drawing polygon

        Args:
            name (str): Name of the mask
            expr (str, optional): Name of the map or expression using names
                of the map, e.g. 'Si', 'Fe/(Fe+Mg)', 'Na+Ca+K'. If None, the
                total counts are used.
            invert (bool): Whether mask inside or outside of the polygon.
                Default is False, so inside is masked.
            activate (bool): Whether the mask should be set as active. Default
                is True

        For additional arguments see `Mapset.show`
        """
        figsize = kwargs.get("figsize", self.figsize)
        invert = kwargs.get("invert", False)
        cmap = plt.get_cmap(kwargs.get("cmap", self.default_cmap))
        cmap.set_under(kwargs.get("under", cmap(0.0)))
        cmap.set_over(kwargs.get("over", cmap(1.0)))
        cmap.set_bad(kwargs.get("masked", "white"))
        cdfclip = kwargs.get("cdfclip", (2, 98))
        title = kwargs.get("title", "Total element counts" if expr is None else expr)
        colorbar = kwargs.get("colorbar", True)
        dt = self.get_map(expr)
        if kwargs.get("zscore", False):
            dt = stats.zscore(dt, nan_policy="omit")
        cmin = np.nanpercentile(dt, cdfclip[0])
        cmax = np.nanpercentile(dt, cdfclip[1])
        vmin = kwargs.get("vmin", cmin)
        vmax = kwargs.get("vmax", cmax)
        if kwargs.get("log", False):
            if vmin == 0:
                vmin = dt[dt > 0].min()
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        f, ax = plt.subplots(figsize=figsize)
        ax.set_aspect(self.aspect)
        dt_masked = np.ma.masked_where(self.mask | np.isnan(dt) | np.isinf(dt), dt)
        img = ax.imshow(dt_masked, cmap=cmap, norm=norm)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if title is not None:
            ax.set_title(title)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            f.colorbar(img, cax=cax, extend="both")
        f.tight_layout()
        selector = PolygonSelector(ax, lambda *args: None)
        plt.show()
        ny, nx = self.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        pp = mplPath(selector.verts)
        mask = pp.contains_points(np.array([x, y]).T).reshape((ny, nx))
        if invert:
            self.__masks[name] = np.invert(mask)
        else:
            self.__masks[name] = mask
        if kwargs.get("activate", True):
            self.active_mask = name

    def clip(self, rmin=0, rmax=None, cmin=0, cmax=None, **kwargs):
        """Clip out rectangular area from sample.

        Note: All existing masks are also clipped

        Args:
            rmin (int): Minimum row to be clipped. Default 0.
            rmax (int): Maximum row to be clipped. Default is max.
            cmin (int): Minimum column to be clipped. Default 0.
            cmax (int): Maximum column to be clipped. Default is max.

        Additional keyword arguments are passed to Mapset.__init__.

        """
        if rmax is None:
            rmax = self.shape[0]
        if cmax is None:
            cmax = self.shape[1]
        maps = {}
        for element in self.maps:
            maps[element] = self[element][rmin:rmax, cmin:cmax]
        masks = {}
        for mask in self.masks:
            masks[mask] = self.get_mask(mask)[rmin:rmax, cmin:cmax]
        kwargs["masks"] = masks
        if "active_mask" not in kwargs:
            kwargs["active_mask"] = self.active_mask
        return Mapset(maps, **kwargs)

    def randomclip(self, rows, cols, **kwargs):
        """Clip randomly placed rectangular area

        Args:
            rows (int): Number of rows to be clipped
            cols (int): Number of columns to be clipped

        Additional keyword arguments are passed to Mapset.__init__.

        """
        rmin = np.random.randint(0, self.shape[0] - rows)
        cmin = np.random.randint(0, self.shape[1] - cols)
        return self.clip(rmin, rmin + rows, cmin, cmin + cols, **kwargs)

    ### Plotting

    def show(self, expr=None, **kwargs):
        """Show map on figure

        Args:
            expr (str, optional): Name of the map or expression using names
                of the map, e.g. 'Si', 'Fe/(Fe+Mg)', 'Na+Ca+K'. If None, the
                total counts are used.
            colorbar (bool): Wheter to show colorbar. Default True.
            vmin: Minimum value mapped to lowest color in colormap. Default
                value is calculated according to cdfclip setting.
            vmax: Maximum value mapped to lowest color in colormap. Default
                value is calculated according to cdfclip setting.
            cdfclip (tuple): Default colormap mapping of values based on cumulative
                distribution range. Default is (2, 98)
            cmap (str): Name of matplotlib colormap. Default is 'inferno'
            under (str): Name of the color used for values below vmin. Default is
                lowest color of active colormap.
            over (str): Name of the color used for values above vmax. Default is
                highest color of active colormap.
            zscore (bool): Whether to transform values to z-score. Default False.
            log (bool): Whether to use logarithmic mapping to color map. Default False.
            title (str): Figure title. Default is the expr.
            figsize (tuple): Figure size. Default is Mapset.figsize
            transpose (bool): Whether to transpose the map. Default is Mapset.transpose
            filename (str): If provided, the figure is saved to file. Default None.
            clip (bool): If True, user can provide by mouse the rectangular area used
                for clipping. Clipped sample is returned.

        """

        def onselect_function(eclick, erelease):
            if (rect_selector.extents[1] - rect_selector.extents[0] >= 1) & (
                rect_selector.extents[3] - rect_selector.extents[2] >= 1
            ):
                plt.close(f)

        figsize = kwargs.get("figsize", self.figsize)
        transpose = kwargs.get("transpose", self.transpose)
        cmap = plt.get_cmap(kwargs.get("cmap", self.default_cmap))
        cmap.set_under(kwargs.get("under", cmap(0.0)))
        cmap.set_over(kwargs.get("over", cmap(1.0)))
        cmap.set_bad(kwargs.get("masked", "white"))
        cdfclip = kwargs.get("cdfclip", (2, 98))
        title = kwargs.get("title", "Total element counts" if expr is None else expr)
        if self.active_mask is not None:
            title += f" [{self.active_mask}]"
        colorbar = kwargs.get("colorbar", True)
        dt = self.get_map(expr)
        if kwargs.get("zscore", False):
            dt = stats.zscore(dt, nan_policy="omit")
        cmin = np.nanpercentile(dt, cdfclip[0])
        cmax = np.nanpercentile(dt, cdfclip[1])
        vmin = kwargs.get("vmin", cmin)
        vmax = kwargs.get("vmax", cmax)
        if kwargs.get("log", False):
            if vmin == 0:
                vmin = dt[dt > 0].min()
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        f, ax = plt.subplots(figsize=figsize)
        ax.set_aspect(self.aspect)
        dt_masked = np.ma.masked_where(self.mask | np.isnan(dt) | np.isinf(dt), dt)
        if transpose:
            img = ax.imshow(dt_masked.T, cmap=cmap, norm=norm)
        else:
            img = ax.imshow(dt_masked, cmap=cmap, norm=norm)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if title is not None:
            ax.set_title(title)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            f.colorbar(img, cax=cax, extend="both")
        filename = kwargs.get("filename", None)
        f.tight_layout()
        if filename is not None:
            plt.axis("off")
            f.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(f)
        else:
            if kwargs.get("clip", False):
                rect_selector = RectangleSelector(
                    ax, onselect_function, button=[1], minspanx=1, minspany=1
                )
            plt.show()
            if kwargs.get("clip", False):
                if (rect_selector.extents[1] - rect_selector.extents[0] >= 1) & (
                    rect_selector.extents[3] - rect_selector.extents[2] >= 1
                ):
                    if transpose:
                        rmin, rmax, cmin, cmax = map(int, rect_selector.extents)
                    else:
                        cmin, cmax, rmin, rmax = map(int, rect_selector.extents)
                    print(f"Using: clip({rmin}, {rmax}, {cmin}, {cmax})")
                    return self.clip(rmin, rmax, cmin, cmax)

    def kde(self, expr=None, **kwargs):
        """Show Gaussian kernel density estimate of map values

        Args:
            expr (str, optional): Name of the map or expression using names
                of the map, e.g. 'Si', 'Fe/(Fe+Mg)', 'Na+Ca+K'. If None, the
                total counts are used.
            vmin: Minimum value used for KDE. Default value is calculated
                according to cdfclip setting.
            vmax: Maximum value used for KDE. Default value is calculated
                according to cdfclip setting.
            cdfclip (tuple): Default range of values used for KDE based on
                cumulative distribution range. Default is (0, 100)
            zscore (bool): Whether to transform values to z-score. Default False.
            log (bool): Whether to use logarithms of values for cdfclip.
                Default False.
            title (str): Figure title. Default is the expr.
            figsize (tuple): Figure size. Default is Mapset.figsize

        """
        figsize = kwargs.get("figsize", (8, 6))
        title = kwargs.get("title", "Total element counts" if expr is None else expr)
        values = self.values(expr)
        if kwargs.get("zscore", False):
            values = stats.zscore(values, nan_policy="omit")
        cdfclip = kwargs.get("cdfclip", (0, 100))
        if kwargs.get("log", False):
            cmin = np.percentile(np.log(values), cdfclip[0])
            cmax = np.percentile(np.log(values), cdfclip[1])
            vmin = kwargs.get("vmin", np.exp(cmin))
            vmax = kwargs.get("vmax", np.exp(cmax))
        else:
            cmin = np.percentile(values, cdfclip[0])
            cmax = np.percentile(values, cdfclip[1])
            vmin = kwargs.get("vmin", cmin)
            vmax = kwargs.get("vmax", cmax)
        values = values[values >= vmin]
        values = values[values <= vmax]
        kernel = stats.gaussian_kde(
            np.random.choice(
                values, size=int(np.exp(np.log(len(values)) / 2)), replace=True
            )
        )
        x_d = np.linspace(values.min(), values.max(), 1000)
        f, ax = plt.subplots(figsize=figsize)
        ax.plot(x_d, kernel(x_d).T)
        if title is not None:
            ax.set_title(f"Gaussian KDE - {title}")
        plt.show()

    ### Class methods for phase mapping

    def create_clusters(self, **kwargs):
        """Calculate KMeans clustering of elemental maps

        To selects initial cluster centroids, k-means++ is used.

        Note: KMeans are used for subsequent Agglomerative clustering
            to create phase map.

        Args:
            n_kmeans (int): Number of clusters to be created. Default 256
            ignore (list): List of elements to be ignored for KMeans
                clustering. Default []
            zscore (bool): Transform values to zscore before clustering.
                Default False
            log1p (bool): Logarithmic transform of values before clustering.
                Default False

        Additional keyword arguments are passed to aggclusters() method.

        """
        n_kmeans = kwargs.get("n_kmeans", 256)
        ignore = kwargs.get("ignore", [])
        use = list(set(self.element_maps) - set(ignore))
        dt = pd.DataFrame(np.array([self.values(el) for el in use]).T, columns=use)
        if kwargs.get("zscore", False):
            dt = dt.apply(stats.zscore)
        if kwargs.get("log1p", False):
            tr = FunctionTransformer(
                np.log1p, validate=True, feature_names_out="one-to-one"
            ).fit(dt)
            dt = pd.DataFrame(tr.transform(dt), columns=tr.get_feature_names_out())
        kmeans = KMeans(n_clusters=n_kmeans, init="k-means++", n_init="auto")
        print("Clustering, please wait...")
        self.clusters = kmeans.fit_predict(dt)
        self.centers = kmeans.cluster_centers_
        self.legend = MapLegend(**kwargs)
        self.aggclusters(**kwargs)

    def aggclusters(self, **kwargs):
        """Agglomerative clustering of KMeans.

        Args:
            n_clusters (int): Number of classes to be created. Default is
                n_clusters defined in Mapset.legend

        Note: When n_clusters is provided and is different from active legend
            n_clusters value, the sample legend is re-initialized.

        Additional keyword arguments are passsed for MapLegend()

        """
        assert (
            self.clusters is not None
        ), "Mapset not yet clustered. Use create_clusters() method."

        n_clusters = kwargs.get("n_clusters", self.legend.n_clusters)
        hierarchical_cluster = AgglomerativeClustering(
            n_clusters=n_clusters, metric="euclidean", linkage="ward"
        )
        self.labels = hierarchical_cluster.fit_predict(self.centers)
        agg = self.clusters.copy()
        for ix in range(len(self.labels)):
            agg[self.clusters == ix] = self.labels[ix]

        self.img = np.full(self.shape, np.nan)
        self.img[np.invert(self.mask)] = agg
        if n_clusters != self.legend.n_clusters:
            self.legend = MapLegend(**kwargs)

    @property
    def label_df(self):
        assert self.labels is not None, "Not aggregated. Use aggclusters() method."
        df = self.element_df.copy()
        df["Label"] = self.labels[self.clusters]
        df["Counts"] = self.values().astype(int)
        return df

    def phase_df(self, phase):
        """Return averaged values for each class of Agglomerative clustering of
        given phase as pandas DataFrame.

        Args:
            phase (str): Name of phase to be retrieved.

        """
        assert self.labels is not None, "Not aggregated. Use aggclusters() method."
        mask = np.invert(self.mask)
        assert phase in self.legend.store, f"Phase {phase} not found in legend."
        pmask = np.zeros_like(self.img, dtype=bool)
        for value in self.legend.store[phase]["values"]:
            pmask = np.logical_or(pmask, self.img == value)
        mask = np.logical_and(mask, pmask)
        elements = self.element_df.columns
        df = pd.DataFrame(
            np.array([self[el][mask].flatten() for el in elements]).T,
            columns=elements,
        )
        df["Class"] = self.img[mask].flatten().astype(int)
        df["Counts"] = self.total_counts[mask].flatten().astype(int)
        return df

    def get_label_mask(self, *args, invert=False):
        """Get mask corresponding to given class(es) from Agglomerative clustering.

        Args:
            value (int): Any number of classes to be used for mask
            invert (bool): Invert mask. Default False

        """
        assert self.labels is not None, "Not aggregated. Use aggclusters() method."
        mask = np.full(self.img.shape, False)
        for v in args:
            mask = np.logical_or(mask, self.img == v)
        if invert:
            mask = np.invert(mask)
        return mask

    def label_info(self, sorted=False):
        """Returns averaged values for each class of Agglomerative clustering
        as Pandas DataFrame.

        Note: When particular class has legend label, the name of the phase is
        added to result.

        Args:
            sorted (bool): Whether to sort according to class proportion
        """
        assert self.labels is not None, "Not aggregated. Use aggclusters() method."
        df = self.label_df
        dfg = df.groupby("Label")
        mn = dfg.mean()
        elements = self.element_df.columns
        res = pd.concat(
            (100 * mn[elements].divide(mn[elements].sum(axis=1), axis=0), mn["Counts"]),
            axis=1,
        )
        res["Prop"] = 100 * dfg.size() / np.count_nonzero(~self.mask)
        for label, leg in self.legend.store.items():
            for v in leg["values"]:
                res.loc[v, "Phase"] = label
        if sorted:
            return res.sort_values("Prop", ascending=False).dropna(
                how="all", subset=elements
            )
        else:
            return res.dropna(how="all", subset=elements)

    def get_phase_mask(self, phase, invert=False):
        """Get mask corresponding to given phase from legend.

        Args:
            phase (str): Name of phase. Must be in legend
            invert (bool): Invert mask. Default False

        """
        assert phase in self.legend.store, f"Phase {phase} not found in legend."
        mask = self.get_label_mask(*self.legend.store[phase]["values"])
        if invert:
            mask = np.invert(mask)
        return mask

    def phase_info(self, sorted=False, **kwargs):
        """Returns averaged values for each phase from legend as Pandas DataFrame.

        Args:
            sorted (bool): Whether to sort according to phase proportion
        """
        assert self.labels is not None, "Not aggregated. Use aggclusters method."
        dfs = []
        if self.legend.store:
            for phase in self.legend.store:
                df = self.phase_df(phase)
                df["Phase"] = phase
                dfs.append(df)
            df = pd.concat(dfs)
            dfg = df.groupby("Phase")
            mn = dfg.mean()
            elements = self.element_df.columns
            res = pd.concat(
                (
                    100 * mn[elements].divide(mn[elements].sum(axis=1), axis=0),
                    mn["Counts"],
                ),
                axis=1,
            )
            res["Prop"] = 100 * dfg.size() / np.count_nonzero(~self.mask)
            if sorted:
                return res.sort_values("Prop", ascending=False)
            else:
                return res
        else:
            print("The legend has no entry. Use MapLegend.add method...")

    def phasemap(self, **kwargs):
        """Show phase map using sample legend.

        Args:
            figsize (tuple): Figure size. Default is Mapset.figsize
            transpose (bool): Whether to transpose the map. Default is Mapset.transpose
            filename (str): If provided, the figure is saved to file. Default None.

        """
        assert self.labels is not None, "Not aggregated. Use aggclusters method."

        def format_coord(x, y):
            col = round(x)
            row = round(y)
            nrows, ncols, _ = self.shape
            if 0 <= col < ncols and 0 <= row < nrows:
                if transpose:
                    if self.mask.T[row, col]:
                        clabel = "M"
                        compo = ""
                    else:
                        clabel = int(self.img.T[row, col])
                        compo = " ".join(
                            [
                                f"{el}: {self[el].T[row, col]}"
                                for el in self.element_df.columns
                            ]
                        )
                else:
                    if self.mask[row, col]:
                        clabel = "M"
                        compo = ""
                    else:
                        clabel = int(self.img[row, col])
                        compo = " ".join(
                            [
                                f"{el}: {self[el][row, col]}"
                                for el in self.element_df.columns
                            ]
                        )
                return f"Class: {clabel} {compo}"

        figsize = kwargs.get("figsize", (8, 6))
        transpose = kwargs.get("transpose", self.transpose)
        f, ax = plt.subplots(figsize=figsize)
        ax.set_aspect(self.aspect)
        dix = np.zeros_like(self.img, dtype=int)
        dix[~np.isnan(self.img)] = self.img[~np.isnan(self.img)].astype(int)
        if transpose:
            RGBA = np.dstack([self.legend.ind2rgb(dix.T), ~self.mask.T])
        else:
            RGBA = np.dstack([self.legend.ind2rgb(dix), ~self.mask])
        ax.imshow(RGBA)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title("Phase map")
        ax.format_coord = format_coord
        # create a patch (proxy artist) for every color
        patches = self.legend.get_patches(self.img, self.mask)
        if patches:
            ax.legend(
                handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
            )
        filename = kwargs.get("filename", None)
        f.tight_layout()
        if filename is not None:
            f.savefig(filename, dpi=300)
        plt.show()


class MapLegend:
    """A class to store phase map legend.

    Note: Legend have to be used for agglomerative clustering with exactly
    same number of classes as defined in legend.

    Args:
        n_clusters (int): Number of classes used with this legend. Default 16
        cmap (str): Color map used for auto color mapping. Default 'nipy_spectral'

    Attributes:
        n_clusters (int): Number of classes used in legend
        norm (matplotlib.colors.Normalize): Matplotlib mapping to colors
        cmap (str): Matplotlib color map used for automatic color mapping
        store (dict): Legend items for labelled classes
        unlabeled (list): List of unlabeled classes
    """

    def __init__(self, **kwargs):
        self.n_clusters = kwargs.get("n_clusters", 16)
        self.norm = colors.Normalize(vmin=0, vmax=self.n_clusters - 1)
        cmap = kwargs.get("cmap", "nipy_spectral")
        if cmap in plt.colormaps():
            self.__cmap = cmap
        else:
            self.__cmap = "nipy_spectral"
        self.store = {}

    def __repr__(self):
        header = f"Legend for {self.n_clusters} clusters.\n"
        if self.store:
            return header + str(pd.DataFrame(self.store).T)
        else:
            return header + "No items"

    @property
    def cmap(self):
        return plt.get_cmap(self.__cmap)

    @cmap.setter
    def cmap(self, name):
        if name in plt.colormaps():
            self.__cmap = name
        else:
            print(
                f"Colormap {name} not available. Available colormaps are:"
                f"{plt.colormaps()}"
            )

    def class_color(self, value):
        """Return color for given class

        Args:
            value (int): class number
        """
        color = colors.to_rgb(self.cmap(self.norm(value)))
        for v in self.store.values():
            if value in v["values"]:
                color = colors.to_rgb(v["color"])
                break
        return color

    @property
    def lut(self):
        """Return RGB color array for all classes"""
        return np.array([self.class_color(v) for v in np.arange(self.n_clusters)])

    def ind2rgb(self, ind):
        return self.lut[ind]

    def add(self, phase, color, values):
        """Add new phase to the legend

        Args:
            phase (str): Name of phase
            color (str): Name of matplotlib color
            values (tuple): tuple of class values corresponding to given phase
        """
        assert isinstance(phase, str), "phase must be string"
        assert colors.is_color_like(
            color
        ), "color must be matplotlib named color or RGB tuple"
        if isinstance(values, int):
            values = (values,)
        if isinstance(values, list):
            values = tuple(values)
        if not isinstance(values, tuple):
            raise ValueError("values must be int or list/tuple of ints")
        self.store[phase] = dict(color=color, values=values)

    @property
    def unlabeled(self):
        others = set(np.arange(self.n_clusters))
        for v in self.store.values():
            others -= set(v["values"])
        return list(others)

    def get_patches(self, img, mask):
        def count_value(value):
            return len(np.where(img[~mask].flatten() == value)[0])

        tot = np.count_nonzero(~mask)
        others = self.unlabeled
        other = sum([count_value(v) for v in others])
        patches = []
        for label, v in self.store.items():
            cnt = sum([count_value(v) for v in v["values"]])
            if cnt > 0:
                info = f" ({100*cnt / tot:.2f}%) {list(v['values'])}"
                patch = mpatches.Patch(color=v["color"], label=label + info)
                patches.append(patch)
        if other > 0:
            opatch = mpatches.Patch(
                color="none", label=f"Others ({100*other / tot:.2f}%)"
            )
            patches.append(opatch)
        return patches
