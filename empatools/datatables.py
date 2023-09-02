import pyparsing
from functools import wraps
import pandas as pd
from pandas.api.types import is_numeric_dtype
from periodictable import formula, oxygen
from periodictable.core import ision

"""
from petrotools import Oxides
import minerals as m

d = Oxides.example_data()
c = d.get_sample('-g-')

garnet = m.Garnet_Fe2()

apfu = c.apfu(garnet)

apfu.endmembers()

#######################################

from petrotools import Oxides
import minerals as m
from plotting import plot_grt_profile

fn = '/home/ondro/Active/Bodonchin/Petro/230804-Mongol/230804-analyses.xlsx'

d = Oxides.from_upsg_empa(fn, sheet_name='GRT')
s = d.get_sample('LX449A')

garnet = m.Garnet_Fe2()

apfu = s.apfu(garnet)
em = apfu.endmembers()
plot_grt_profile(em, percents=True)

# Molecular average
s.molprop().mean.oxwt()

###################

from petrotools import Oxides
import minerals as m

fn = '/home/ondro/Active/Bodonchin/Petro/230804-Mongol/230804-analyses.xlsx'

d = Oxides.from_upsg_empa(fn, skipfooter=0, sheet_name='OTHER')
p = d.get_sample('pl')

plg = m.Feldspar()
p.mineral_endmembers(plg)

############

from petrotools import Oxides
import minerals as m

d = Oxides.from_clipboard()
"""


def oxide2props(f):
    ncat, element = f.structure[0]
    noxy = f.atoms[oxygen]
    charge = 2 * noxy // ncat
    return dict(
        mass=f.mass,
        cation=element.ion[charge],
        noxy=noxy,
        ncat=ncat,
        charge=charge,
        elfrac=f.mass_fraction[element],
    )


def ion2props(ion):
    return dict(mass=ion.mass, element=ion.element, charge=ion.charge)


def compo(**dekwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            df = func(self, *args, **kwargs)
            res = self._data.copy()
            res[self._valid] = df
            # parse kwargs
            dekwargs["units"] = dekwargs.get("units", self.units)
            dekwargs["desc"] = dekwargs.get("desc", self.desc)
            return type(self)(res, name=self.name, **dekwargs)

        return wrapper

    return decorator


class Compo:
    def __init__(self, df, **kwargs):
        # Check argument (convert Series to DataFrame if needed)
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).T
        assert isinstance(
            df, pd.DataFrame
        ), "Argument must be pandas.DataFrame or pd Series"
        # clean column names
        df = df.rename(columns=lambda x: x.strip())
        # parse common kwargs
        self.name = kwargs.get("name", "Compo")
        self.desc = kwargs.get("desc", "Original data")
        # set index
        index_col = kwargs.get("index_col", None)
        if index_col in df:
            df = df.reset_index(drop=True).set_index(index_col)
        self._data = df.copy()

    def __len__(self):
        """Return number of data in datatable"""
        return self._data.shape[0]

    @property
    def df(self):
        return self._data[self._valid].copy()

    @property
    def others(self):
        return self._data[self._others].copy()

    @property
    def names(self):
        return self._valid.copy()

    @property
    def sum(self):
        res = self.df.sum(axis=1)
        res.name = "Total"
        return res

    @property
    def mean(self):
        return type(self)(
            self.df.mean(axis=0), name=self.name, units=self.units, desc="Average"
        )

    def set_index(self, key):
        """Set index of datatable

        Args:
            key (str or list like): Either name of column (see ``Oxides.others``) or
            collection of same length as datatable
        """
        return type(self)(
            self._data.reset_index(
                drop=isinstance(self._data.index, pd.RangeIndex)
            ).set_index(key),
            units=self.units,
            name=self.name,
            desc=self.desc,
        )

    def reset_index(self):
        """Reset index to default pandas.RangeIndex"""
        return type(self)(
            self._data.reset_index(),
            units=self.units,
            name=self.name,
            desc=self.desc,
        )

    def get_sample(self, s):
        """Select subset of data from datatable

        Args:
            s (str or int): When string, returns all data which contain string in
            index. When numeric returns single record.

        Returns:
            Oxides: slected data as datatable
        """
        index = self._data.index
        if is_numeric_dtype(index):
            ix = index == s
        else:
            ix = pd.Series([str(v) for v in index], index=index).str.contains(s)
        return type(self)(
            self._data[ix].copy(),
            units=self.units,
            name=self.name,
            desc=self.desc,
        )

    def to_latex(self, add_total=True, transpose=True, precision=2):
        """Convert datatable to LaTeX representation

        Args:
            add_total (bool, optional): Add column `"Total"` with total sums.
            Default `True`
            transpose (bool, optional): Place samples as columns. Default ``True``
            precision (bool, optional): Nimber of decimal places. Default 2

        """
        return (
            self.table(add_total=add_total, transpose=transpose)
            .fillna("")
            .style.format(precision=precision)
            .to_latex()
        )

    @property
    def props(self):
        return pd.DataFrame(self._props, index=self.names)

    def to_clipboard(self, add_total=True, transpose=True):
        """Copy table to clipboard

        Args:
            add_total (bool, optional): Add column `"Total"` with total sums.
            Default `True`
            transpose (bool, optional): Place samples as columns. Default ``True``
            precision (bool, optional): Nimber of decimal places. Default 2

        """
        df = self.table(add_total=add_total, transpose=transpose)
        df.to_clipboard(excel=True)
        print("Copied to clipboard.")


class Oxides(Compo):
    """A class to store oxides composition.

    There are different way to create `Oxides` object:

    - passing `pandas.DataFrame` with analyses in rows and oxides in columns
    - from clipboard using ``from_clipboard()`` method
    - from Excel using ``from_excel()`` method
    - using ``example_data`` method

    Args:
        df (pandas.DataFrame): plunge direction of linear feature in degrees
        units (str, optional): units of datatable. Default is `"wt%"`
        name (str, optional): name of datatable. Default is `"Compo"`
        desc (str, optional): description of datatable. Default is `"Original data"`
        index_col (str, optional): name of column used as index. Default ``None``

    Attributes:
        df (pandas.DataFrame): subset of datatable with only oxides columns
        units (str): units of data
        elements (pandas.DataFrame): subset of datatable with only elements columns
        others (pandas.DataFrame): subset of datatable with other columns
        props (pandas.DataFrame): chemical properties of present oxides
        names (list): list of present oxides
        sum (pandas.Series): total sum of oxides in current units
        mean (pandas.Series): mean of oxides in current units
        name (str): name of datatable.
        desc (str): description of datatable.
        cat_number (Oxides): Cations number i.e. # moles of cation in mineral
        oxy_number (Oxides): Oxygen number i.e. # moles of oxygen in mineral

    Example:
        >>> d = Oxides(df)
        >>> d = Oxides.from_clipboard()
        >>> d = Oxides.from_excel('analyses.xlsx', sheet_name='Minerals', skiprows=3)
        >>> d = Oxides.example_data()

    """
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.parse_columns()
        self.units = kwargs.get("units", "wt%")

    def parse_columns(self):
        # parse columns
        self._props = []
        self._valid = []
        self._elements = []
        self._others = []
        for col in self._data.columns:
            try:
                f = formula(col)
                if (len(f.atoms) == 2) and (oxygen in f.atoms):
                    self._props.append(oxide2props(f))
                    self._valid.append(col)
                elif (len(f.atoms) == 1) and (is_numeric_dtype(self._data[col].dtype)):
                    self._elements.append(col)
                else:
                    self._others.append(col)
            except (ValueError, pyparsing.exceptions.ParseException):
                self._others.append(col)

    def __repr__(self):
        return "\n".join(
            [
                f"Oxides: {self.name} [{self.units}] - {self.desc}",
                f"{self.df}",
            ]
        )

    def _repr_html_(self):
        return (
            self.df.style.set_caption(
                f"Oxides: {self.name} [{self.units}] - {self.desc}"
            )
            .format(precision=4)
            .to_html()
        )

    @compo(desc="Normalized")
    def normalize(self, to=100):
        """Normalize the values

        Args:
            to (it or float): desired sum. Default 100

        Returns:
            Oxides: normalized datatable

        """
        return to * self.df.div(self.sum, axis=0)

    @property
    def elements(self):
        return self._data[self._elements].copy()

    @compo(units="mol%")
    def molprop(self):
        """Convert oxides weight percents to molar proportions

        Returns:
            Oxides: molar proportions datatable

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        return self.df.div(self.props["mass"])

    @compo(units="wt%")
    def oxwt(self):
        """Convert oxides molar proportions to weight percents

        Returns:
            Oxides: weight percents datatable

        """
        assert self.units == "mol%", "Oxides must be molar percents"
        return self.df.mul(self.props["mass"])

    @compo(desc="Elemental weight")
    def elwt(self):
        """Convert oxides weight percents to elements weight percents

        Returns:
            Oxides: elements weight percents datatable

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        res = self.df.mul(self.props["elfrac"])
        res.columns = [str(cat) for cat in self.props["cation"]]
        return res

    @compo(desc="Elemental weight")
    def elwt_oxy(self):
        """Convert oxides weight percents to elements weight percents incl. oxygen

        Returns:
            Oxides: elements weight percents datatable

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        res = self.df.mul(self.props["elfrac"])
        res.columns = [str(cat) for cat in self.props["cation"]]
        res["O{2-}"] = self.sum - res.sum(axis=1)
        return res

    @property
    @compo(units="moles", desc="Cations number")
    def cat_number(self):
        return self.props["ncat"] * self.molprop().df

    @property
    @compo(units="moles", desc="Oxygens number")
    def oxy_number(self):
        return self.props["noxy"] * self.molprop().df

    def onf(self, noxy):
        """Oxygen normalisation factor - ideal oxygens / sum of oxygens

        Args:
            noxy (int): ideal oxygens

        Returns:
            pandas.Series: oxygen normalisation factors

        """
        return noxy / self.oxy_number.sum

    def cnf(self, ncat):
        """Cation normalisation factor - ideal cations / sum of cations

        Args:
            ncat (int): ideal cations

        Returns:
            pandas.Series: cation normalisation factors

        """
        return ncat / self.cat_number.sum

    @compo(units="charge")
    def charges(self, ncat):
        """Calculates charges based on number of cations

        Args:
            ncat (int): number of cations

        Returns:
            Oxides: charges datatable

        """
        return (
            self.cat_number.df.multiply(self.cnf(ncat), axis=0) * self.props["charge"]
        )

    def chargedef(self, noxy, ncat):
        """Calculates charge deficiency based on number of cations and oxygens

        Args:
            noxy (int): ideal number of oxygens
            ncat (int): ideal number of cations

        Returns:
            pandas.Series: charge deficiency values

        """
        return 2 * noxy - self.charges(ncat).sum

    def cations(self, noxy=1, ncat=1, tocat=False):
        """Cations calculated on the basis of oxygens or cations

        Args:
            noxy (int, optional): ideal number of oxygens. Default 1
            ncat (int, optional): ideal number of cations. Default 1
            tocat (bool, optional): when ``True`` normalized to ``ncat``,
            otherwise to ``noxy``. Default ``False``

        Returns:
            Ions: cations datatable

        """
        if tocat:
            df = self.cat_number.df.multiply(self.cnf(ncat), axis=0)
            df.columns = [str(cat) for cat in self.props["cation"]]
            return Ions(
                df,
                desc=f"Cations p.f.u based on {ncat} cations",
            )
        else:
            df = self.cat_number.df.multiply(self.onf(noxy), axis=0)
            df.columns = [str(cat) for cat in self.props["cation"]]
            return Ions(
                df,
                desc=f"Cations p.f.u based on {noxy} oxygens",
            )

    def apfu(self, mineral, tocat=False):
        """Cations calculated on the basis of mineral formula

        Args:
            mineral (Mineral): instance of mineral
            tocat (bool, optional): when ``True`` normalized to ``mineral.ncat``,
            otherwise to ``mineral.noxy``. Default ``False``

        Returns:
            APFU: a.p.f.u datatable

        """
        noxy, ncat = mineral.noxy, mineral.ncat
        if mineral.needsFe3:
            dt = self.recalculate_Fe(noxy, ncat)
        else:
            dt = self
        if tocat:
            df = dt.cat_number.df.multiply(dt.cnf(ncat), axis=0)
            df.columns = [str(cat) for cat in dt.props["cation"]]
            return APFU(
                df,
                mineral=mineral,
                name=self.name,
                desc=f"Cations p.f.u based on {ncat} cations",
            )
        else:
            df = dt.cat_number.df.multiply(dt.onf(noxy), axis=0)
            df.columns = [str(cat) for cat in dt.props["cation"]]
            return APFU(
                df,
                mineral=mineral,
                name=self.name,
                desc=f"Cations p.f.u based on {noxy} oxygens",
            )

    def mineral_endmembers(self, mineral, tocat=False, force=False):
        """Calculate mineral end-members

        Args:
            mineral (Mineral): instance of mineral.
            tocat (bool, optional): when True normalized to ncat, otherwise to noxy.
            Default False
            force (bool, optional): when True, remaining cations are added to last site

        Returns:
            pandas.DataFrame: calculated endmembers

        """
        apfu = self.apfu(mineral, tocat=tocat)
        return apfu.endmembers(force=force)

    def convert_Fe(self):
        """Recalculate FeO to Fe2O3 or vice-versa

        Note: When only FeO exists, all is recalculated to Fe2O3. When only Fe2O3
        exists, all is recalculated to FeO. Otherwise datatable is not changed.

        Returns:
            Oxides: datatable with converted Fe

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        if ("Fe2O3" not in self.names) and ("FeO" in self.names):
            Fe2to3 = formula("Fe2O3").mass / formula("FeO").mass / 2
            res = self._data.copy()
            res["Fe2O3"] = Fe2to3 * res["FeO"]
            res = res.drop(columns="FeO")
            return Oxides(res, name=self.name, desc="Fe converted")
        elif ("Fe2O3" in self.names) and ("FeO" not in self.names):
            Fe3to2 = 2 * formula("FeO").mass / formula("Fe2O3").mass
            res = self._data.copy()
            res["FeO"] = Fe3to2 * res["Fe2O3"]
            res = res.drop(columns="Fe2O3")
            return Oxides(res, name=self.name, desc="Fe converted")
        else:
            print("No Fe in data. Nothing changed")
            return self

    def recalculate_Fe(self, noxy, ncat):
        """Recalculate Fe based on charge balance

        Args:
            noxy (int): ideal number of oxygens. Default 1
            ncat (int): ideal number of cations. Default 1

        Note: Either both FeO and Fe2O3 are present or any of then, the composition
        is modified to fullfil charge balance for given cations and oxygens.

        Returns:
            Oxides: datatable with recalculated Fe

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        charge = self.cat_number.df.multiply(self.cnf(ncat), axis=0)
        if ("Fe2O3" in self.names) & ("FeO" not in self.names):
            charge["Fe2O3"].loc[pd.isna(self.df["Fe2O3"])] = 0
            toconv = self.chargedef(noxy, ncat)
            charge["Fe2O3"] += toconv
            charge["FeO"] = -toconv
            ncats = self.props["ncat"]
            ncats["FeO"] = 1
            mws = self.props["mass"]
            mws["FeO"] = formula("FeO").mass
        elif "Fe2O3" in self.names:
            charge["Fe2O3"].loc[pd.isna(self.df["Fe2O3"])] = 0
            toconv = self.chargedef(noxy, ncat).clip(lower=0, upper=charge["FeO"])
            charge["Fe2O3"] += toconv
            charge["FeO"] = charge["FeO"] - toconv
            ncats = self.props["ncat"]
            mws = self.props["mass"]
        elif "FeO" in self.names:
            charge["Fe2O3"] = self.chargedef(noxy, ncat).clip(
                lower=0, upper=charge["FeO"]
            )
            charge["FeO"] = charge["FeO"] - charge["Fe2O3"]
            ncats = self.props["ncat"].copy()
            ncats["Fe2O3"] = 2
            mws = self.props["mass"].copy()
            mws["Fe2O3"] = formula("Fe2O3").mass
        else:
            print("No Fe in data. Nothing changed")
            return self
        res = self._data.copy()
        ncharge = charge / ncat
        df = ncharge.mul(mws).mul(self.cat_number.sum, axis="rows").div(ncats)
        res[df.columns] = df
        return Oxides(res, name=self.name, desc="Fe corrected")

    def apatite_correction(self):
        """Apatite correction

        Note: All P2O5 is assumed to be apatite based and is removed from composition
        CaO mol% = CaO mol% - 3.33 * P2O5 mol%

        Returns:
            Oxides: apatite corrected datatable

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        if ("P2O5" in self.names) and ("CaO" in self.names):
            df = self.molprop().normalize(to=1).df
            df["CaO"] = (df["CaO"] - 3.33 * df["P2O5"]).clip(lower=0)
            df = df.drop(columns="P2O5")
            mws = self.props["mass"].drop(labels="P2O5")
            df = df.mul(mws, axis=1)
            df = df.div(df.sum(axis=1), axis=0).mul(self.sum, axis=0)
            res = self._data.copy().drop(columns="P2O5")
            res[df.columns] = df
            return Oxides(res, name=self.name, desc="ap corrected")
        else:
            print("Not Ca and P in data. Nothing changed")
            return self

    def check_cations(self, noxy, ncat, confidence=None):
        """Return normalized error of calculated cations

        err = [ncat - #cat(noxy)]/ncat

        Args:
            noxy (int): number of oxygens
            ncat (int): number of cations
            confidence (float, optional): if not ``None``, returns booleans with
            ``True``, where error is within confidence

        Returns:
            pandas.Series: boolean or normalized errors

        """
        err = abs(ncat - self.cations(noxy=noxy).sum) / ncat
        if confidence is not None:
            return err <= confidence
        else:
            return err

    def table(self, add_total=True, transpose=True):
        # helper for exports
        df = self.df
        if add_total:
            df["Total"] = self.sum
        if transpose:
            df = df.T
        return df

    @classmethod
    def from_clipboard(cls, index_col=None, vertical=False):
        """Parse datatable from clipboard.

        Note: By default, oxides should be arranged in columns with one line header
        containing case-sensitive oxides formula (e.g. Al2O3). All other columns are
        available as ``Oxides.others`` and are not used for calculations.

        Args:
            index_col (str or None, optional): name of the columns used for index.
            Default None.
            vertical (bool, optional): Set ``True`` when oxides are aranged in rows.
            Default ``False``

        Returns:
            Oxides: datatable

        """
        df = pd.read_clipboard(index_col=False)
        if vertical:
            df = df.set_index(df.columns[0]).T
            df.columns.name = None
        return cls(df, index_col=index_col)

    @classmethod
    def from_excel(cls, filename, **kwargs):
        """Read datatable from Excel file.

        Note: Oxides must be arranged in columns with one line header
        containing case-sensitive oxides formula (e.g. Al2O3). All other columns are
        available as ``Oxides.others`` and are not used for calculations.

        Args:
            filename (str): string path to file. For other possibility see
            ``pandas.read_excel``
            **kwargs: all keyword arguments are passed to ``pandas.read_excel``

        Returns:
            Oxides: datatable

        """
        index_col = kwargs.pop("index_col", None)
        df = pd.read_excel(filename, **kwargs)
        return cls(df, index_col=index_col)

    @classmethod
    def from_upsg_empa(cls, filename, index_col="Comment", **kwargs):
        if "skiprows" not in kwargs:
            kwargs["skiprows"] = 3
        if "skipfooter" not in kwargs:
            kwargs["skipfooter"] = 6
        kwargs["index_col"] = False
        df = pd.read_excel(filename, **kwargs)
        return cls(df, index_col=index_col)

    @classmethod
    def example_data(cls, sample="default"):
        """Get exemple datatable

        Args:
            sample (str, optional): Name of sample, on of the `"default"`,
            `"pyroxenes"`, `"avgpelite"` or `"grt_profile"`. Default is `"default"`

        Returns:
            Oxides: datatable

        """
        # fmt: off
        examples = dict(
            default = {
                "SiO2":{"0":37.218,"1":37.363,"2":23.748,"3":46.986,"4":48.389,"5":48.87,"6":61.839,"7":37.816,"8":37.584,"9":60.657,"10":61.338,"11":28.057,"12":27.565,"13":27.091,"14":46.078,"15":45.487},
                "Al2O3":{"0":20.349,"1":20.037,"2":22.152,"3":39.183,"4":32.414,"5":32.696,"6":24.661,"7":21.184,"8":21.311,"9":25.087,"10":25.025,"11":53.95,"12":53.885,"13":53.841,"14":35.793,"15":35.669},
                "MgO":{"0":13.784,"1":14.106,"2":9.611,"3":0.151,"4":8.227,"5":8.39,"6":0.0,"7":4.189,"8":4.651,"9":0.001,"10":0.015,"11":2.04,"12":1.972,"13":2.099,"14":0.652,"15":0.803},
                "FeO":{"0":13.14,"1":12.539,"2":31.549,"3":1.264,"4":7.713,"5":7.482,"6":0.484,"7":35.118,"8":33.895,"9":0.0,"10":0.03,"11":11.888,"12":11.828,"13":11.875,"14":0.784,"15":0.895},
                "MnO":{"0":0.0,"1":0.0,"2":0.066,"3":0.033,"4":0.06,"5":0.068,"6":0.015,"7":1.001,"8":0.999,"9":0.0,"10":0.003,"11":0.186,"12":0.175,"13":0.18,"14":0.0,"15":0.015},
                "CaO":{"0":0.013,"1":0.013,"2":0.028,"3":0.41,"4":0.027,"5":0.049,"6":5.738,"7":1.435,"8":1.454,"9":5.849,"10":5.801,"11":0.036,"12":0.0,"13":0.031,"14":0.002,"15":0.0},
                "K2O":{"0":8.998,"1":8.946,"2":0.023,"3":3.409,"4":0.0,"5":0.0,"6":0.032,"7":0.0,"8":0.0,"9":0.106,"10":0.079,"11":0.0,"12":0.0,"13":0.0,"14":9.333,"15":9.394},
                "Na2O":{"0":0.404,"1":0.461,"2":0.035,"3":4.883,"4":0.924,"5":0.845,"6":7.917,"7":0.0,"8":0.035,"9":7.94,"10":7.997,"11":0.0,"12":0.0,"13":0.0,"14":1.182,"15":1.171},
                "TiO2":{"0":1.01,"1":1.206,"2":0.047,"3":0.083,"4":0.005,"5":0.0,"6":0.0,"7":0.0,"8":0.013,"9":0.0,"10":0.047,"11":0.585,"12":0.607,"13":0.554,"14":0.697,"15":0.643},
                "Cr2O3":{"0":0.022,"1":0.02,"2":0.0,"3":0.0,"4":0.009,"5":0.0,"6":0.01,"7":0.0,"8":0.034,"9":0.0,"10":0.0,"11":0.063,"12":0.003,"13":0.046,"14":0.081,"15":0.033},
                "ZnO":{"0":0.02,"1":0.042,"2":0.017,"3":0.0,"4":0.0,"5":0.0,"6":0.019,"7":0.0,"8":0.0,"9":0.003,"10":0.057,"11":1.024,"12":1.035,"13":1.116,"14":0.0,"15":0.015},
                "P2O5":{"0":0.005,"1":0.0,"2":0.037,"3":0.006,"4":0.005,"5":0.0,"6":0.025,"7":0.006,"8":0.03,"9":0.113,"10":0.064,"11":0.0,"12":0.0,"13":0.0,"14":0.0,"15":0.0},
                "Y2O3":{"0":0.007,"1":0.0,"2":0.0,"3":0.0,"4":0.0,"5":0.0,"6":0.0,"7":0.0,"8":0.107,"9":0.0,"10":0.0,"11":0.0,"12":0.0,"13":0.0,"14":0.0,"15":0.034},
                "F":{"0":0.367,"1":0.313,"2":0.017,"3":0.0,"4":0.0,"5":0.0,"6":0.0,"7":0.0,"8":0.0,"9":0.0,"10":0.0,"11":0.0,"12":0.0,"13":0.0,"14":0.0,"15":0.0},
                "Cl":{"0":0.368,"1":0.335,"2":0.196,"3":0.065,"4":0.014,"5":0.023,"6":0.0,"7":0.004,"8":0.014,"9":0.0,"10":0.0,"11":0.007,"12":0.0,"13":0.014,"14":0.013,"15":0.0},
                "Total":{"0":95.467,"1":95.173,"2":87.475,"3":96.458,"4":97.784,"5":98.418,"6":100.74,"7":100.752,"8":100.124,"9":99.756,"10":100.456,"11":97.834,"12":97.07,"13":96.844,"14":94.612,"15":94.159},
                "Comment":{"0":"bt-01","1":"bt-02","2":"chl-04","3":"pa-05","4":"cd-06","5":"cd-07","6":"pl-08","7":"g-09","8":"g-10","9":"pl-22","10":"pl-23","11":"st-33","12":"st-34","13":"st-35","14":"ms-49","15":"ms-50"}  # noqa: E501
            },
            pyroxenes={
                "SiO2": {"0": 45.33, "1": 45.64, "2": 45.43, "3": 45.42, "4": 45.0},
                "TiO2": {"0": 0.69, "1": 0.71, "2": 0.69, "3": 0.68, "4": 0.67},
                "Cr2O3": {"0": 0.15, "1": 0.16, "2": 0.14, "3": 0.16, "4": 0.18},
                "Al2O3": {"0": 12.54, "1": 12.58, "2": 12.56, "3": 12.59, "4": 12.43},
                "FeO": {"0": 9.64, "1": 9.83, "2": 9.82, "3": 9.59, "4": 9.79},
                "MnO": {"0": 0.15, "1": 0.16, "2": 0.14, "3": 0.16, "4": 0.12},
                "NiO": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0},
                "MgO": {"0": 16.98, "1": 17.07, "2": 16.84, "3": 16.97, "4": 16.85},
                "CaO": {"0": 9.68, "1": 9.73, "2": 9.66, "3": 9.62, "4": 9.55},
                "Na2O": {"0": 1.16, "1": 1.1, "2": 1.11, "3": 1.09, "4": 1.07},
                "K2O": {"0": 0.39, "1": 0.38, "2": 0.37, "3": 0.38, "4": 0.38},
                "H2O": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0},
                "F": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0},
                "Cl": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0},
                "Total": {"0": 96.71, "1": 97.36, "2": 96.76, "3": 96.66, "4": 96.04},
            },
            # Caddick, 2008
            avgpelite={
                "SiO2": {"0": 59.8},
                "Al2O3": {"0": 16.57},
                "FeO": {"0": 5.81},
                "MgO": {"0": 2.62},
                "CaO": {"0": 1.09},
                "Na2O": {"0": 1.73},
                "K2O": {"0": 3.53},
                "TiO2": {"0": 0.75},
                "MnO": {"0": 0.1},
            },
            grt_profile={"Na2O":{"0":0.008,"1":0.003,"2":0.007,"3":0.009,"4":0.008,"5":0.01,"6":0.017,"7":0.007,"8":0.031,"9":0.012,"10":0.129,"11":0.001,"12":0.007,"13":0.012,"14":0.0,"15":0.014,"16":0.017,"17":0.026,"18":0.024,"19":0.009,"20":0.032,"21":0.0,"22":0.006,"23":0.018,"24":0.017,"25":0.018,"26":0.014,"27":0.009,"28":0.019,"29":0.016,"30":0.015,"31":0.015,"32":0.014,"33":0.02,"34":0.004,"35":0.017,"36":0.02,"37":0.016,"38":0.028,"39":0.021,"40":0.009,"41":0.013,"42":0.016,"43":0.018,"44":0.013,"45":0.0,"46":0.01,"47":0.008,"48":0.015,"49":0.027,"50":0.008,"51":0.016,"52":0.006,"53":0.014,"54":0.023,"55":0.006,"56":0.031,"57":0.01,"58":0.008,"59":0.004,"60":0.012,"61":0.011,"62":0.015,"63":0.013,"64":0.016,"65":0.007,"66":0.033,"67":0.021,"68":0.0,"69":0.007,"70":0.031,"71":0.006,"72":0.022,"73":0.014,"74":0.019,"75":0.02,"76":0.015,"77":0.023,"78":0.006,"79":0.021,"80":0.04,"81":0.012,"82":0.023,"83":0.013,"84":0.018,"85":0.01,"86":0.022,"87":0.016,"88":0.003,"89":0.011,"90":0.002,"91":0.0,"92":0.011,"93":0.001,"94":0.012,"95":0.017,"96":0.015,"97":0.02,"98":0.008,"99":0.011},
                "Al2O3":{"0":21.19,"1":21.291,"2":21.211,"3":20.998,"4":21.325,"5":21.14,"6":21.103,"7":21.077,"8":21.087,"9":21.106,"10":21.052,"11":21.078,"12":21.083,"13":21.233,"14":21.268,"15":21.122,"16":21.146,"17":20.88,"18":20.882,"19":20.814,"20":20.924,"21":20.89,"22":20.825,"23":20.912,"24":20.857,"25":20.676,"26":20.752,"27":20.797,"28":20.85,"29":20.849,"30":20.726,"31":20.83,"32":20.701,"33":20.884,"34":20.621,"35":20.636,"36":20.645,"37":20.64,"38":20.697,"39":21.06,"40":20.78,"41":20.724,"42":20.686,"43":20.674,"44":20.556,"45":20.619,"46":20.491,"47":20.483,"48":20.656,"49":20.608,"50":20.574,"51":20.71,"52":20.585,"53":20.594,"54":20.692,"55":20.642,"56":20.543,"57":20.59,"58":20.594,"59":20.572,"60":20.566,"61":20.555,"62":20.539,"63":20.726,"64":20.579,"65":20.535,"66":20.725,"67":20.513,"68":20.625,"69":20.582,"70":20.628,"71":20.568,"72":20.501,"73":20.546,"74":20.527,"75":20.551,"76":20.605,"77":20.741,"78":20.595,"79":20.774,"80":20.617,"81":20.731,"82":20.557,"83":20.681,"84":20.637,"85":20.808,"86":20.638,"87":20.794,"88":20.8,"89":20.736,"90":20.51,"91":20.736,"92":20.696,"93":20.728,"94":20.76,"95":20.655,"96":20.703,"97":20.695,"98":20.781,"99":20.665},
                "P2O5":{"0":0.018,"1":0.04,"2":0.034,"3":0.027,"4":0.013,"5":0.029,"6":0.025,"7":0.041,"8":0.02,"9":0.018,"10":0.016,"11":0.022,"12":0.003,"13":0.038,"14":0.012,"15":0.0,"16":0.018,"17":0.021,"18":0.018,"19":0.037,"20":0.015,"21":0.007,"22":0.024,"23":0.024,"24":0.007,"25":0.003,"26":0.014,"27":0.035,"28":0.037,"29":0.016,"30":0.025,"31":0.037,"32":0.047,"33":0.042,"34":0.031,"35":0.04,"36":0.033,"37":0.044,"38":0.029,"39":0.008,"40":0.015,"41":0.001,"42":0.027,"43":0.026,"44":0.001,"45":0.033,"46":0.022,"47":0.019,"48":0.029,"49":0.039,"50":0.036,"51":0.015,"52":0.006,"53":0.041,"54":0.04,"55":0.027,"56":0.025,"57":0.001,"58":0.022,"59":0.033,"60":0.022,"61":0.033,"62":0.006,"63":0.029,"64":0.058,"65":0.02,"66":0.038,"67":0.008,"68":0.036,"69":0.009,"70":0.0,"71":0.025,"72":0.039,"73":0.033,"74":0.03,"75":0.036,"76":0.047,"77":0.027,"78":0.031,"79":0.051,"80":0.026,"81":0.0,"82":0.01,"83":0.013,"84":0.024,"85":0.019,"86":0.0,"87":0.023,"88":0.007,"89":0.018,"90":0.009,"91":0.019,"92":0.013,"93":0.026,"94":0.026,"95":0.012,"96":0.008,"97":0.028,"98":0.02,"99":0.039},
                "Y2O3":{"0":0.002,"1":0.0,"2":0.008,"3":0.0,"4":0.005,"5":0.001,"6":0.0,"7":0.017,"8":0.015,"9":0.035,"10":0.018,"11":0.064,"12":0.031,"13":0.044,"14":0.076,"15":0.04,"16":0.175,"17":0.122,"18":0.095,"19":0.053,"20":0.093,"21":0.0,"22":0.092,"23":0.078,"24":0.092,"25":0.075,"26":0.0,"27":0.086,"28":0.099,"29":0.022,"30":0.096,"31":0.115,"32":0.099,"33":0.07,"34":0.024,"35":0.028,"36":0.042,"37":0.052,"38":0.026,"39":0.008,"40":0.003,"41":0.019,"42":0.0,"43":0.0,"44":0.0,"45":0.0,"46":0.0,"47":0.015,"48":0.013,"49":0.0,"50":0.008,"51":0.009,"52":0.0,"53":0.0,"54":0.025,"55":0.0,"56":0.0,"57":0.025,"58":0.039,"59":0.039,"60":0.017,"61":0.038,"62":0.016,"63":0.03,"64":0.003,"65":0.039,"66":0.013,"67":0.045,"68":0.0,"69":0.0,"70":0.048,"71":0.064,"72":0.052,"73":0.115,"74":0.143,"75":0.111,"76":0.132,"77":0.113,"78":0.133,"79":0.123,"80":0.16,"81":0.094,"82":0.029,"83":0.072,"84":0.032,"85":0.042,"86":0.007,"87":0.025,"88":0.036,"89":0.011,"90":0.015,"91":0.004,"92":0.007,"93":0.0,"94":0.006,"95":0.0,"96":0.0,"97":0.018,"98":0.0,"99":0.0},
                "CaO":{"0":1.969,"1":1.929,"2":1.992,"3":1.908,"4":1.987,"5":2.075,"6":1.96,"7":1.92,"8":2.023,"9":1.929,"10":2.042,"11":1.947,"12":1.959,"13":1.901,"14":1.882,"15":1.854,"16":2.103,"17":2.273,"18":2.437,"19":2.596,"20":2.767,"21":3.098,"22":3.0,"23":3.168,"24":3.189,"25":3.085,"26":3.452,"27":3.427,"28":3.73,"29":3.694,"30":3.736,"31":3.906,"32":3.986,"33":4.082,"34":4.14,"35":4.289,"36":4.541,"37":4.419,"38":4.553,"39":4.585,"40":4.914,"41":4.918,"42":5.155,"43":5.375,"44":5.167,"45":4.848,"46":5.241,"47":5.238,"48":4.955,"49":5.035,"50":4.849,"51":4.768,"52":4.717,"53":4.631,"54":4.365,"55":4.394,"56":4.359,"57":4.119,"58":4.199,"59":4.134,"60":4.036,"61":4.0,"62":4.081,"63":3.999,"64":3.964,"65":4.083,"66":3.9,"67":3.768,"68":3.686,"69":3.53,"70":3.365,"71":3.216,"72":3.096,"73":2.994,"74":2.781,"75":2.671,"76":2.6,"77":2.469,"78":2.352,"79":2.297,"80":2.229,"81":2.027,"82":1.812,"83":1.838,"84":1.956,"85":1.944,"86":1.909,"87":2.033,"88":2.002,"89":1.972,"90":2.02,"91":1.996,"92":1.942,"93":1.991,"94":1.921,"95":1.946,"96":1.952,"97":1.943,"98":1.913,"99":2.157},
                "K2O":{"0":0.011,"1":0.0,"2":0.006,"3":0.0,"4":0.0,"5":0.008,"6":0.001,"7":0.0,"8":0.001,"9":0.0,"10":0.031,"11":0.001,"12":0.0,"13":0.002,"14":0.007,"15":0.0,"16":0.002,"17":0.01,"18":0.0,"19":0.014,"20":0.002,"21":0.001,"22":0.005,"23":0.0,"24":0.003,"25":0.002,"26":0.002,"27":0.0,"28":0.007,"29":0.0,"30":0.0,"31":0.0,"32":0.0,"33":0.0,"34":0.001,"35":0.004,"36":0.003,"37":0.001,"38":0.0,"39":0.0,"40":0.0,"41":0.002,"42":0.005,"43":0.009,"44":0.0,"45":0.0,"46":0.001,"47":0.001,"48":0.0,"49":0.002,"50":0.0,"51":0.002,"52":0.0,"53":0.0,"54":0.003,"55":0.0,"56":0.0,"57":0.0,"58":0.0,"59":0.006,"60":0.0,"61":0.0,"62":0.0,"63":0.001,"64":0.0,"65":0.0,"66":0.0,"67":0.0,"68":0.0,"69":0.0,"70":0.0,"71":0.004,"72":0.0,"73":0.0,"74":0.004,"75":0.0,"76":0.005,"77":0.003,"78":0.0,"79":0.0,"80":0.0,"81":0.002,"82":0.0,"83":0.017,"84":0.0,"85":0.002,"86":0.004,"87":0.0,"88":0.0,"89":0.0,"90":0.0,"91":0.004,"92":0.002,"93":0.005,"94":0.0,"95":0.003,"96":0.0,"97":0.005,"98":0.006,"99":0.005},
                "TiO2":{"0":0.0,"1":0.0,"2":0.015,"3":0.029,"4":0.009,"5":0.046,"6":0.011,"7":0.042,"8":0.072,"9":0.005,"10":0.022,"11":0.05,"12":0.021,"13":0.002,"14":0.004,"15":0.0,"16":0.0,"17":0.022,"18":0.007,"19":0.025,"20":0.03,"21":0.0,"22":0.051,"23":0.018,"24":0.045,"25":0.077,"26":0.039,"27":0.039,"28":0.068,"29":0.037,"30":0.077,"31":0.042,"32":0.104,"33":0.086,"34":0.023,"35":0.069,"36":0.069,"37":0.08,"38":0.051,"39":0.342,"40":0.047,"41":0.09,"42":0.074,"43":0.06,"44":0.058,"45":0.044,"46":0.075,"47":0.09,"48":0.103,"49":0.085,"50":0.055,"51":0.089,"52":0.006,"53":0.094,"54":0.081,"55":0.06,"56":0.065,"57":0.077,"58":0.07,"59":0.061,"60":0.06,"61":0.09,"62":0.085,"63":0.017,"64":0.085,"65":0.15,"66":0.063,"67":0.105,"68":0.044,"69":0.031,"70":0.027,"71":0.058,"72":0.035,"73":0.021,"74":0.039,"75":0.005,"76":0.098,"77":0.03,"78":0.0,"79":0.016,"80":0.016,"81":0.0,"82":0.02,"83":0.0,"84":0.034,"85":0.01,"86":0.046,"87":0.022,"88":0.005,"89":0.014,"90":0.015,"91":0.03,"92":0.026,"93":0.02,"94":0.019,"95":0.026,"96":0.0,"97":0.008,"98":0.0,"99":0.0},
                "Cr2O3":{"0":0.0,"1":0.036,"2":0.0,"3":0.026,"4":0.033,"5":0.028,"6":0.031,"7":0.046,"8":0.057,"9":0.04,"10":0.032,"11":0.019,"12":0.029,"13":0.01,"14":0.0,"15":0.023,"16":0.019,"17":0.024,"18":0.028,"19":0.003,"20":0.026,"21":0.004,"22":0.026,"23":0.01,"24":0.013,"25":0.0,"26":0.0,"27":0.015,"28":0.015,"29":0.0,"30":0.047,"31":0.019,"32":0.017,"33":0.038,"34":0.063,"35":0.001,"36":0.024,"37":0.019,"38":0.032,"39":0.029,"40":0.0,"41":0.048,"42":0.006,"43":0.003,"44":0.02,"45":0.024,"46":0.008,"47":0.031,"48":0.03,"49":0.029,"50":0.054,"51":0.03,"52":0.017,"53":0.014,"54":0.018,"55":0.003,"56":0.009,"57":0.015,"58":0.029,"59":0.018,"60":0.034,"61":0.034,"62":0.046,"63":0.039,"64":0.004,"65":0.0,"66":0.014,"67":0.051,"68":0.0,"69":0.007,"70":0.0,"71":0.038,"72":0.038,"73":0.0,"74":0.02,"75":0.004,"76":0.042,"77":0.006,"78":0.001,"79":0.0,"80":0.007,"81":0.01,"82":0.0,"83":0.005,"84":0.006,"85":0.045,"86":0.072,"87":0.033,"88":0.022,"89":0.026,"90":0.046,"91":0.0,"92":0.057,"93":0.054,"94":0.052,"95":0.032,"96":0.033,"97":0.055,"98":0.044,"99":0.079},
                "FeO":{"0":33.202,"1":32.513,"2":32.515,"3":32.275,"4":31.936,"5":31.832,"6":31.754,"7":31.77,"8":31.457,"9":31.44,"10":30.908,"11":31.016,"12":30.371,"13":30.554,"14":30.078,"15":30.063,"16":29.655,"17":29.133,"18":28.823,"19":28.2,"20":28.153,"21":27.781,"22":27.44,"23":26.98,"24":26.624,"25":26.744,"26":26.316,"27":26.039,"28":25.47,"29":25.395,"30":25.292,"31":25.065,"32":24.695,"33":24.845,"34":24.375,"35":24.382,"36":23.945,"37":24.124,"38":23.682,"39":23.692,"40":23.717,"41":23.206,"42":23.243,"43":23.033,"44":23.157,"45":23.276,"46":22.976,"47":23.037,"48":23.25,"49":23.17,"50":23.209,"51":23.73,"52":23.628,"53":23.969,"54":23.87,"55":24.008,"56":24.042,"57":24.194,"58":24.016,"59":24.249,"60":24.573,"61":24.439,"62":24.775,"63":24.818,"64":24.738,"65":24.973,"66":24.89,"67":25.165,"68":25.515,"69":25.952,"70":25.891,"71":26.288,"72":26.607,"73":26.969,"74":27.239,"75":27.568,"76":27.867,"77":28.292,"78":28.416,"79":28.762,"80":28.715,"81":29.157,"82":29.559,"83":30.21,"84":30.291,"85":30.933,"86":32.105,"87":31.812,"88":31.627,"89":31.531,"90":31.462,"91":31.499,"92":31.825,"93":31.862,"94":32.173,"95":32.284,"96":32.44,"97":32.667,"98":32.8,"99":32.509},
                "MnO":{"0":2.021,"1":1.981,"2":2.129,"3":2.345,"4":2.636,"5":2.878,"6":3.2,"7":3.425,"8":3.797,"9":3.903,"10":3.983,"11":4.381,"12":4.998,"13":5.367,"14":5.897,"15":6.486,"16":6.671,"17":6.958,"18":7.361,"19":7.622,"20":8.013,"21":8.352,"22":8.687,"23":9.082,"24":9.338,"25":9.726,"26":9.894,"27":10.066,"28":10.461,"29":10.563,"30":10.789,"31":10.898,"32":10.903,"33":11.069,"34":11.344,"35":11.487,"36":11.565,"37":11.708,"38":11.715,"39":11.72,"40":11.757,"41":11.758,"42":11.822,"43":11.954,"44":11.917,"45":12.047,"46":11.749,"47":11.862,"48":11.896,"49":11.958,"50":11.786,"51":11.78,"52":11.805,"53":11.661,"54":11.759,"55":11.715,"56":11.657,"57":11.544,"58":11.485,"59":11.415,"60":11.44,"61":11.473,"62":11.282,"63":11.263,"64":10.993,"65":10.841,"66":10.647,"67":10.552,"68":10.292,"69":10.274,"70":10.007,"71":9.634,"72":9.629,"73":9.151,"74":8.882,"75":8.585,"76":8.296,"77":8.073,"78":7.668,"79":7.308,"80":7.078,"81":6.685,"82":6.298,"83":5.727,"84":5.003,"85":4.322,"86":3.081,"87":2.954,"88":3.113,"89":3.218,"90":3.062,"91":2.827,"92":2.679,"93":2.444,"94":2.246,"95":2.098,"96":1.935,"97":1.804,"98":1.737,"99":2.71},
                "ZnO":{"0":0.018,"1":0.0,"2":0.017,"3":0.0,"4":0.0,"5":0.041,"6":0.0,"7":0.027,"8":0.0,"9":0.021,"10":0.0,"11":0.012,"12":0.018,"13":0.015,"14":0.004,"15":0.023,"16":0.0,"17":0.0,"18":0.0,"19":0.0,"20":0.0,"21":0.027,"22":0.0,"23":0.009,"24":0.0,"25":0.0,"26":0.003,"27":0.028,"28":0.01,"29":0.012,"30":0.001,"31":0.008,"32":0.0,"33":0.008,"34":0.0,"35":0.0,"36":0.029,"37":0.042,"38":0.014,"39":0.0,"40":0.0,"41":0.019,"42":0.0,"43":0.038,"44":0.0,"45":0.02,"46":0.025,"47":0.002,"48":0.0,"49":0.0,"50":0.0,"51":0.0,"52":0.0,"53":0.028,"54":0.0,"55":0.006,"56":0.03,"57":0.0,"58":0.0,"59":0.013,"60":0.014,"61":0.0,"62":0.0,"63":0.01,"64":0.018,"65":0.006,"66":0.015,"67":0.0,"68":0.009,"69":0.036,"70":0.015,"71":0.0,"72":0.035,"73":0.008,"74":0.0,"75":0.026,"76":0.0,"77":0.003,"78":0.003,"79":0.0,"80":0.008,"81":0.01,"82":0.021,"83":0.0,"84":0.0,"85":0.015,"86":0.0,"87":0.0,"88":0.0,"89":0.0,"90":0.0,"91":0.007,"92":0.0,"93":0.01,"94":0.0,"95":0.022,"96":0.007,"97":0.0,"98":0.011,"99":0.002},
                "MgO":{"0":4.087,"1":4.628,"2":4.639,"3":4.549,"4":4.501,"5":4.43,"6":4.339,"7":4.227,"8":4.076,"9":4.113,"10":4.131,"11":3.97,"12":3.791,"13":3.707,"14":3.618,"15":3.5,"16":3.354,"17":3.252,"18":3.125,"19":3.039,"20":2.932,"21":2.734,"22":2.711,"23":2.621,"24":2.534,"25":2.481,"26":2.378,"27":2.394,"28":2.253,"29":2.182,"30":2.163,"31":2.121,"32":2.087,"33":2.075,"34":1.977,"35":1.919,"36":1.862,"37":1.877,"38":1.826,"39":1.771,"40":1.798,"41":1.731,"42":1.73,"43":1.647,"44":1.689,"45":1.715,"46":1.724,"47":1.691,"48":1.72,"49":1.724,"50":1.765,"51":1.756,"52":1.737,"53":1.804,"54":1.856,"55":1.881,"56":1.878,"57":1.916,"58":1.909,"59":1.963,"60":1.967,"61":2.06,"62":2.016,"63":2.052,"64":2.115,"65":2.088,"66":2.184,"67":2.213,"68":2.277,"69":2.277,"70":2.383,"71":2.471,"72":2.52,"73":2.578,"74":2.705,"75":2.741,"76":2.854,"77":2.943,"78":2.976,"79":3.187,"80":3.182,"81":3.306,"82":3.439,"83":3.707,"84":3.791,"85":4.033,"86":4.143,"87":4.271,"88":4.331,"89":4.324,"90":4.357,"91":4.453,"92":4.389,"93":4.484,"94":4.556,"95":4.625,"96":4.597,"97":4.57,"98":4.327,"99":3.347},
                "SiO2":{"0":37.244,"1":37.574,"2":37.646,"3":37.568,"4":37.69,"5":37.591,"6":37.498,"7":37.623,"8":37.408,"9":37.515,"10":37.55,"11":37.454,"12":37.511,"13":37.126,"14":37.395,"15":37.418,"16":37.348,"17":37.252,"18":37.18,"19":37.319,"20":37.289,"21":37.133,"22":37.197,"23":37.067,"24":37.216,"25":37.153,"26":37.261,"27":37.271,"28":37.226,"29":37.223,"30":37.009,"31":37.017,"32":36.776,"33":37.169,"34":36.905,"35":37.157,"36":37.152,"37":37.025,"38":37.062,"39":36.921,"40":37.209,"41":37.093,"42":37.141,"43":36.912,"44":36.765,"45":36.864,"46":36.993,"47":37.04,"48":36.952,"49":37.04,"50":37.018,"51":37.048,"52":36.86,"53":36.819,"54":36.886,"55":36.794,"56":36.833,"57":36.844,"58":36.688,"59":36.868,"60":36.744,"61":36.87,"62":36.848,"63":36.928,"64":36.858,"65":36.885,"66":36.864,"67":36.912,"68":36.732,"69":36.871,"70":36.73,"71":36.718,"72":36.566,"73":36.618,"74":36.685,"75":36.678,"76":36.696,"77":36.682,"78":36.632,"79":36.7,"80":36.744,"81":36.782,"82":36.825,"83":36.752,"84":36.945,"85":37.16,"86":37.025,"87":36.986,"88":37.139,"89":36.982,"90":37.063,"91":37.017,"92":37.063,"93":36.817,"94":36.719,"95":36.689,"96":37.027,"97":37.014,"98":36.674,"99":36.608}}
        )
        # fmt: on
        return cls(pd.DataFrame(examples[sample]))


class Ions(Compo):
    """A class to store cations composition.

    There are different way to create ``Ions`` object:

    - passing ``pandas.DataFrame`` with analyses in rows and cations in columns. Note,
      that cations needs to by defined with charge e.g. `"Si{4+}"`, `"Na{+}"`
      or `"O{2-}"`
    - calculated from ``Oxides`` datatable

    Args:
        df (pandas.DataFrame): plunge direction of linear feature in degrees
        units (str, optional): units of datatable. Default is `"wt%"`
        name (str, optional): name of datatable. Default is `"Compo"`
        desc (str, optional): description of datatable. Default is `"Original data"`
        index_col (str, optional): name of column used as index. Default ``None``

    Attributes:
        df (pandas.DataFrame): subset of datatable with only ions columns
        units (str): units of data
        others (pandas.DataFrame): subset of datatable with other columns
        props (pandas.DataFrame): chemical properties of present ions
        names (list): list of present ions
        sum (pandas.Series): total sum of oxides in current units
        mean (pandas.Series): mean of oxides in current units
        name (str): name of datatable.
        desc (str): description of datatable.

    Example:
        >>> d = Ions(df)
        >>> d = Oxides.cations(noxy=12, ncat=8)

    """
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.parse_columns()
        self.units = kwargs.get("units", "atoms")

    def parse_columns(self):
        self._props = []
        self._valid = []
        self._cations = []
        self._anions = []
        self._others = []
        for col in self._data.columns:
            try:
                f = formula(col)
                if len(f.atoms) == 1:
                    atom = f.structure[0][1]
                    if ision(atom):
                        props = ion2props(atom)
                        self._props.append(props)
                        self._valid.append(col)
                        if props["charge"] > 0:
                            self._cations.append(col)
                        else:
                            self._anions.append(col)
                    else:
                        self._others.append(col)
                else:
                    self._others.append(col)
            except (ValueError, pyparsing.exceptions.ParseException):
                self._others.append(col)

    def __repr__(self):
        return "\n".join(
            [
                f"Ions: {self.name} [{self.units}] - {self.desc}",
                f"{self.df}",
            ]
        )

    def _repr_html_(self):
        return (
            self.df.style.set_caption(f"Ions: {self.name} [{self.units}] - {self.desc}")
            .format(precision=4)
            .to_html()
        )

    def table(self, add_total=True, transpose=True):
        df = self.df
        if add_total:
            df["Total"] = self.sum
        if transpose:
            df = df.T
        return df


class APFU(Ions):
    """A class to store cations p.f.u for mineral.

    There are different way to create ``APFU`` object:

    - passing ``pandas.DataFrame`` with analyses in rows and cations in columns. Note,
      that cations needs to by defined with charge e.g. `"Si{4+}"`, `"Na{+}"`
      or `"O{2-}"`
    - calculated from ``Oxides`` datatable

    Args:
        df (pandas.DataFrame): plunge direction of linear feature in degrees
        mineral (Mineral): mineral used for formula calculations. See `empatools.mindb`
        units (str, optional): units of datatable. Default is `"wt%"`
        name (str, optional): name of datatable. Default is `"Compo"`
        desc (str, optional): description of datatable. Default is `"Original data"`
        index_col (str, optional): name of column used as index. Default ``None``

    Attributes:
        df (pandas.DataFrame): subset of datatable with only ions columns
        mineral (Mineral): mineral used for formula calculations
        units (str): units of data
        others (pandas.DataFrame): subset of datatable with other columns
        props (pandas.DataFrame): chemical properties of present ions
        names (list): list of present ions
        sum (pandas.Series): total sum of oxides in current units
        mean (pandas.Series): mean of oxides in current units
        name (str): name of datatable.
        desc (str): description of datatable.
        reminder (APFU): Returns remaining atoms after mineral formula occupation

    Example:
        >>> d = APFU(df)
        >>> d = Oxides.apfu(mindb.Garnet_Fe2())

    """
    def __init__(self, df, mineral, **kwargs):
        super().__init__(df, **kwargs)
        self.parse_columns()
        self.mineral = mineral

    def __repr__(self):
        return "\n".join(
            [
                f"APFU[{self.mineral}]: {self.name} [{self.units}] - {self.desc}",
                f"{self.df}",
            ]
        )

    def _repr_html_(self):
        return (
            self.df.style.set_caption(
                f"APFU[{self.mineral}]: {self.name} [{self.units}] - {self.desc}"
            )
            .format(precision=4)
            .to_html()
        )

    def get_sample(self, s):
        res = super().get_sample(s)
        res.mineral = self.mineral

    def endmembers(self, force=False):
        """Calculate endmembers proportions

        Args:
            force (bool, optional): when True, remaining cations are added to last site

        """
        if self.mineral.has_endmembers:
            res = []
            for ix, row in self.df.iterrows():
                res.append(self.mineral.endmembers(row, force=force))
            return pd.DataFrame(res, index=self.df.index)
        else:
            raise TypeError(f"{self.mineral} has no endmembers")

    def mineral_apfu(self, force=False):
        """Calculate apfu from structural formula

        Note: Ions instance must be based on mineral

        Args:
            force (bool, optional): when True, remaining cations are added to last site

        """
        if self.mineral.has_structure:
            res = []
            for ix, row in self.df.iterrows():
                res.append(self.mineral.apfu(row, force=force))
            return Ions(
                pd.DataFrame(res, index=self._data.index),
                mineral=self.mineral,
                name=self.name,
                desc=self.desc,
            )
        else:
            raise TypeError(f"{self.mineral} has no structure")

    @property
    def reminder(self):
        return self.df - self.mineral_apfu().df

    def check_formula(self, confidence=None):
        """Return normalized error of calculated cations

        Args:
            confidence (float, optional): if not None, returns boolean vector with True,
            where error is within confidence

        """
        err = abs(self.mineral.ncat - self.sum) / self.mineral.ncat
        if confidence is not None:
            return err <= confidence
        else:
            return err

    def table(self, add_total=True, transpose=True):
        df = self.df
        if add_total:
            df["Total"] = self.sum
        ox = pd.Series(len(self) * [self.mineral.noxy], index=df.index, name="Oxygen")
        df = pd.concat([ox, df], axis=1)
        if transpose:
            df = df.T
        return df


####################################################


def read_actlabs(filename=""):
    if filename != "":
        df = pd.read_excel(filename, header=2)
        units = df.iloc[0]
        limits = df.iloc[1]
        method = df.iloc[2]
        df = df.rename(columns={"Analyte Symbol": "Sample"})[3:].set_index("Sample").T
        # replace detection limits
        for col in df:
            df[col][df[col].str.contains("< ") is True] = 0
        df = df.astype(float)
        return df, units, limits, method


def read_bureau_veritas(filename=""):
    if filename != "":
        df = pd.read_excel(filename, skiprows=9)
        method = df.iloc[0][2:]
        cols = df.iloc[1]
        cols[:2] = ["Sample", "Type"]
        units = df.iloc[2][2:]
        limits = df.iloc[3][2:]
        selection = df.iloc[:, 1] == "Rock Pulp"

        dt = df.iloc[:, 2:][selection]
        # replace detection limits
        for col in dt:
            dt[col][dt[col].astype(str).str.startswith("<") is True] = 0

        dt = dt.astype(float).copy()

        res = pd.concat([df[selection].iloc[:, :2], dt], axis=1)
        res.columns = cols.str.strip()
        return res, units, limits, method
