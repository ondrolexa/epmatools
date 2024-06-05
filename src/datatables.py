import pyparsing
import importlib.resources
import pandas as pd
from pandas.api.types import is_numeric_dtype
from periodictable import formula, oxygen
from periodictable.core import ision


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


class Compo:
    def __init__(self, df, **kwargs):
        # Check argument (convert Series to DataFrame if needed)
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).T
        assert isinstance(
            df, pd.DataFrame
        ), "Argument must be pandas.DataFrame or pd Series"
        # clean column names
        if kwargs.get("strip_column_names", True):
            df = df.rename(columns=lambda x: x.strip())
        # parse common kwargs
        self.name = kwargs.get("name", "Compo")
        self.desc = kwargs.get("desc", "Original data")
        # set index
        index_col = kwargs.get("index_col", None)
        if index_col in df:
            df = df.reset_index(drop=True).set_index(index_col)
        if kwargs.get("strip_index", True):
            if pd.api.types.is_string_dtype(df.index.dtype):
                df.index = [v.strip() for v in df.index]
        self._data = df.copy()

    def __len__(self):
        """Return number of data in datatable"""
        return self._data.shape[0]

    def __getitem__(self, index):
        if isinstance(index, str):
            if index in self._valid:
                return self.df[index].copy()
            else:
                raise ValueError(f"Index must be on of {self._valid}")
        if isinstance(index, slice):
            return self.finalize(self._data.loc[index].copy())
        else:
            raise TypeError("Only string could be used as index.")

    def finalize(self, vals, **kwargs):
        return type(self)(
            vals,
            units=kwargs.get("units", self.units),
            name=kwargs.get("name", self.name),
            desc=kwargs.get("desc", self.desc),
        )

    def reversed(self):
        """Return in reversed order"""
        res = self._data.reindex(index=self._data.index[::-1])
        return self.finalize(res)

    def iterrows(self, what=None):
        """Return row iterator yielding tuples (label, row)

        Args:
            what (str): Which columns are included. None for all, 'valid' for valid,
                'elements' for elements and 'others' for others. Default None

        """
        if what is None:
            return self._data.iterrows()
        else:
            return getattr(self, what).iterrows()

    @property
    def df(self):
        return self._data[self._valid].copy()

    valid = df

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
        res = self.df.mean(axis=0)
        return self.finalize(res, desc="Mean")

    def drop(self, labels):
        """Drop rows based on index

        Args:
            labels: single or list of indexes to be dropped
        """
        return self.finalize(self._data.drop(labels))

    def set_index(self, key):
        """Set index of datatable

        Args:
            key (str or list like): Either name of column (see ``Oxides.others``) or
                collection of same length as datatable
        """
        assert key in self._others, f"Column name must be one of {self._others}"
        return self.finalize(
            self._data.reset_index(drop=True).set_index(key, drop=False)
        )

    def reset_index(self):
        """Reset index to default pandas.RangeIndex"""
        return self.finalize(self._data.reset_index(drop=True))

    def head(self, n=5):
        """Return first n rows

        Args:
            n (int): Number of rows. Default 5

        """
        return self.finalize(self._data.iloc[:n])

    def tail(self, n=5):
        """Return last n rows

        Args:
            n (int): Number of rows. Default 5

        """
        return self.finalize(self._data.iloc[-n:])

    def row(self, label, what=None):
        """Return row as pandas Series

        Args:
            label (label): label from index
            what (str): Which columns are included. None for all, 'valid' for valid,
                'elements' for elements and 'others' for others. Default None

        """
        if what is None:
            return self._data.loc[label].copy()
        else:
            return getattr(self, what).loc[label].copy()

    def search(self, s, on=None):
        """Search subset of data from datatable containing string s in index or column

        Note: Works only with non-numeric index or column

        Args:
            s (str): Returns all data which contain string s in index.

        Returns:
            Selected data as datatable
        """
        assert isinstance(s, str), "Argument must be string"
        if on is None:
            col = self._data.index
        else:
            assert on in self._data, f"Column {on} not found"
            col = self._data[on]
        if not is_numeric_dtype(col):
            ix = pd.Series([str(v) for v in col], index=self._data.index).str.contains(
                s
            )
            return self.finalize(self._data.loc[ix].copy())
        else:
            if on is None:
                print("Index is numeric. Try to use .row method")
            else:
                print("Selected column is numeric. Try to use .row method")

    def select(self, loc):
        """Select rows by label(s) or a boolean array

        Args:
            loc: Single or list of labels, slice or boolean array.

        Returns:
            Selected data as datatable
        """
        return self.finalize(self._data.loc[loc].copy())

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
        >>> d = Oxides.from_examples('minerals')

    """

    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.parse_columns()
        # drop rows with no data
        self._data = self._data.dropna(how="all", subset=self._valid)
        # finish
        self.units = kwargs.get("units", "wt%")
        self.decimals = kwargs.get("decimals", 3)

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
                f"{self.df.round(decimals=self.decimals)}",
            ]
        )

    def _repr_html_(self):
        return (
            self.df.style.set_caption(
                f"Oxides: {self.name} [{self.units}] - {self.desc}"
            )
            .format(precision=self.decimals)
            .to_html()
        )

    def normalize(self, to=100):
        """Normalize the values

        Args:
            to (int or float): desired sum. Default 100

        Returns:
            Oxides: normalized datatable

        """
        return self.finalize(to * self.df.div(self.sum, axis=0), desc="Normalized")

    @property
    def elements(self):
        return self._data[self._elements].copy()

    def molprop(self):
        """Convert oxides weight percents to molar proportions

        Returns:
            Oxides: molar proportions datatable

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        return self.finalize(self.df.div(self.props["mass"]), units="mol%")

    def oxwt(self):
        """Convert oxides molar proportions to weight percents

        Returns:
            Oxides: weight percents datatable

        """
        assert self.units == "mol%", "Oxides must be molar percents"
        return self.finalize(self.df.mul(self.props["mass"]), units="wt%")

    def elwt(self):
        """Convert oxides weight percents to elements weight percents

        Returns:
            Oxides: elements weight percents datatable

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        res = self.df.mul(self.props["elfrac"])
        res.columns = [str(cat) for cat in self.props["cation"]]
        return Ions(res, units="wt%", desc="Elemental weight")

    def elwt_oxy(self):
        """Convert oxides weight percents to elements weight percents incl. oxygen

        Returns:
            Oxides: elements weight percents datatable

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        res = self.df.mul(self.props["elfrac"])
        res.columns = [str(cat) for cat in self.props["cation"]]
        res["O{2-}"] = self.sum - res.sum(axis=1)
        return Ions(res, units="wt%", desc="Elemental weight")

    @property
    def cat_number(self):
        return self.finalize(
            self.props["ncat"] * self.molprop().df, units="moles", desc="Cations number"
        )

    @property
    def oxy_number(self):
        return self.finalize(
            self.props["noxy"] * self.molprop().df, units="moles", desc="Oxygens number"
        )

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

    def charges(self, ncat):
        """Calculates charges based on number of cations

        Args:
            ncat (int): number of cations

        Returns:
            Oxides: charges datatable

        """
        return self.finalize(
            self.cat_number.df.multiply(self.cnf(ncat), axis=0) * self.props["charge"],
            units="charge",
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
        if mineral.needsFe == "Fe2":
            dt = self.convert_Fe()
        elif mineral.needsFe == "Fe3":
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

    def convert_Fe(self, to="FeO"):
        """Recalculate FeO to Fe2O3 or vice-versa

        Note:
            When only FeO exists, all is recalculated to Fe2O3. When only Fe2O3
            exists, all is recalculated to FeO. When both exists, Fe2O3 is
            recalculated and added to FeO. Otherwise datatable is not changed.

        Args:
            to (str): to what iron oxide Fe should be converted. Default `"FeO"`

        Returns:
            Oxides: datatable with converted Fe oxide

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        if (to == "FeO") and ("Fe2O3" in self.names):
            Fe3to2 = 2 * formula("FeO").mass / formula("Fe2O3").mass
            res = self._data.copy()
            if "FeO" in self.names:
                res["FeO"] += Fe3to2 * res["Fe2O3"]
            else:
                res["FeO"] = Fe3to2 * res["Fe2O3"]
            res = res.drop(columns="Fe2O3")
            return Oxides(res, name=self.name, desc="Fe converted")
        elif (to == "Fe2O3") and ("FeO" in self.names):
            Fe2to3 = formula("Fe2O3").mass / formula("FeO").mass / 2
            res = self._data.copy()
            if "Fe2O3" in self.names:
                res["Fe2O3"] += Fe2to3 * res["FeO"]
            else:
                res["Fe2O3"] = Fe2to3 * res["FeO"]
            res = res.drop(columns="FeO")
            return Oxides(res, name=self.name, desc="Fe converted")
        else:
            return self

    def recalculate_Fe(self, noxy, ncat):
        """Recalculate Fe based on charge balance

        Args:
            noxy (int): ideal number of oxygens. Default 1
            ncat (int): ideal number of cations. Default 1

        Note:
            Either both FeO and Fe2O3 are present or any of then, the composition
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

        Note:
            All P2O5 is assumed to be apatite based and is removed from composition

                `CaO mol% = CaO mol% - (10 / 3) * P2O5 mol%`

        Returns:
            Oxides: apatite corrected datatable

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        if ("P2O5" in self.names) and ("CaO" in self.names):
            df = self.molprop().normalize(to=1).df
            df["CaO"] = (df["CaO"] - (10 / 3) * df["P2O5"]).clip(lower=0)
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

    def TCbulk(self, H2O=-1, oxygen=0.01, system="MnNCKFMASHTO"):
        """Print oxides formatted as THERMOCALC bulk script

        Note:
            The CaO is recalculate using apatite correction based on P205 if available.

        Args:
            H2O (float): wt% of water. When -1 the amount is calculated as 100 - Total
                Default -1.
            oxygen (float): value to calculate moles of ferric iron.
                Moles FeO = FeOtot - 2O and moles Fe2O3 = O. Default 0.01
            system (str): axfile to be used. One of 'MnNCKFMASHTO', 'NCKFMASHTO',
                'KFMASH', 'NCKFMASHTOCr', 'NCKFMASTOCr'. Default 'MnNCKFMASHTO'

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        if system == "MnNCKFMASHTO":
            bulk = [
                "H2O",
                "SiO2",
                "Al2O3",
                "CaO",
                "MgO",
                "FeO",
                "K2O",
                "Na2O",
                "TiO2",
                "MnO",
                "O",
            ]
        elif system == "NCKFMASHTO":
            bulk = [
                "H2O",
                "SiO2",
                "Al2O3",
                "CaO",
                "MgO",
                "FeO",
                "K2O",
                "Na2O",
                "TiO2",
                "O",
            ]
        elif system == "KFMASH":
            bulk = ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O"]
        elif system == "NCKFMASHTOCr":
            bulk = [
                "H2O",
                "SiO2",
                "Al2O3",
                "MgO",
                "FeO",
                "K2O",
                "Na2O",
                "TiO2",
                "O",
                "Cr2O3",
            ]
        elif system == "NCKFMASTOCr":
            bulk = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "TiO2", "O", "Cr2O3"]
        else:
            raise TypeError(f"{system} not implemented")

        df = self.convert_Fe().apatite_correction().df
        # Water
        if "H2O" in bulk:
            if H2O == -1:
                H2O = 100 - df.sum(axis=1)
                H2O[H2O < 0] = 0
            else:
                H2O = H2O * df.sum(axis=1) / (100 - H2O)

            df["H2O"] = H2O
        if "O" in bulk:
            df["O"] = oxygen
        df = Oxides(df[bulk]).molprop().normalize()

        print("bulk" + "".join([f"{lbl:>7}" for lbl in bulk]))
        for ix, row in df._data.iterrows():
            print("bulk" + "".join([f" {v:6.3f}" for v in row.values]) + f"  % {ix}")

    def Perplexbulk(self, H2O=-1, oxygen=0.01, system="MnNCKFMASHTO"):
        """Print oxides formatted as PerpleX thermodynamic component list

        Note:
            The CaO is recalculate using apatite correction based on P205 if available.

        Args:
            H2O (float): wt% of water. When -1 the amount is calculated as 100 - Total
                Default -1.
            oxygen (float): value to calculate moles of ferric iron.
                Moles FeO = FeOtot - O and moles Fe2O3 = O. Default 0.01
            system (str): axfile to be used. One of 'MnNCKFMASHTO', 'NCKFMASHTO',
                'KFMASH', 'NCKFMASHTOCr', 'NCKFMASTOCr'. Default 'MnNCKFMASHTO'

        """
        assert self.units == "wt%", "Oxides must be weight percents"
        if system == "MnNCKFMASHTO":
            bulk = [
                "H2O",
                "SiO2",
                "Al2O3",
                "CaO",
                "MgO",
                "FeO",
                "K2O",
                "Na2O",
                "TiO2",
                "MnO",
                "O2",
            ]
        elif system == "NCKFMASHTO":
            bulk = [
                "H2O",
                "SiO2",
                "Al2O3",
                "CaO",
                "MgO",
                "FeO",
                "K2O",
                "Na2O",
                "TiO2",
                "O2",
            ]
        elif system == "KFMASH":
            bulk = ["H2O", "SiO2", "Al2O3", "MgO", "FeO", "K2O"]
        elif system == "NCKFMASHTOCr":
            bulk = [
                "H2O",
                "SiO2",
                "Al2O3",
                "MgO",
                "FeO",
                "K2O",
                "Na2O",
                "TiO2",
                "O2",
                "Cr2O3",
            ]
        elif system == "NCKFMASTOCr":
            bulk = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "TiO2", "O2", "Cr2O3"]
        else:
            raise TypeError(f"{system} not implemented")

        df = self.convert_Fe().apatite_correction().df
        # Water
        if "H2O" in bulk:
            if H2O == -1:
                H2O = 100 - df.sum(axis=1)
                H2O[H2O < 0] = 0
            else:
                H2O = H2O * df.sum(axis=1) / (100 - H2O)

            df["H2O"] = H2O
        if "O2" in bulk:
            df["O2"] = 2 * oxygen
        df = Oxides(df[bulk]).molprop().normalize()

        print("begin thermodynamic component list")
        for ox, val in df._data.iloc[0].items():
            print(f"{ox:6s}1 {val:8.5f}      0.00000      0.00000     molar amount")
        print("end thermodynamic component list")

    @classmethod
    def from_clipboard(cls, index_col=None, vertical=False):
        """Parse datatable from clipboard.

        Note:
            By default, oxides should be arranged in columns with one line header
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

        Note:
            Oxides must be arranged in columns with one line header
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
    def from_examples(cls, example=None):
        """Get example datatable

        Args:
            example (str, optional): Name of example. When None, available examples
                are printed. Default is `None`

        Returns:
            Oxides: datatable

        """
        resources = importlib.resources.files("epmatools") / "data"
        datapath = resources / "oxides"
        if example is None:
            print(f"Available examples: {[f.stem for f in datapath.glob('*.csv')]}")
        else:
            fname = (datapath / example).with_suffix(".csv")
            assert fname.exists(), "Example {example} do not exists."
            return cls(pd.read_csv(fname))


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
        self.decimals = kwargs.get("decimals", 4)

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
                f"{self.df.round(decimals=self.decimals)}",
            ]
        )

    def _repr_html_(self):
        return (
            self.df.style.set_caption(f"Ions: {self.name} [{self.units}] - {self.desc}")
            .format(precision=self.decimals)
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
        mineral (Mineral): mineral used for formula calculations. See `epmatools.mindb`
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
        self.decimals = kwargs.get("decimals", 4)
        self.mineral = mineral

    def __repr__(self):
        return "\n".join(
            [
                f"APFU[{self.mineral}]: {self.name} [{self.units}] - {self.desc}",
                f"{self.df.round(decimals=self.decimals)}",
            ]
        )

    def _repr_html_(self):
        return (
            self.df.style.set_caption(
                f"APFU[{self.mineral}]: {self.name} [{self.units}] - {self.desc}"
            )
            .format(precision=self.decimals)
            .to_html()
        )

    def finalize(self, vals, **kwargs):
        return type(self)(
            vals,
            mineral=self.mineral,
            units=kwargs.get("units", self.units),
            name=kwargs.get("name", self.name),
            desc=kwargs.get("desc", self.desc),
        )

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
            raise TypeError(f"{self.mineral} has no endmembers method defined")

    def mineral_apfu(self, force=False):
        """Calculate apfu from structural formula

        Args:
            force (bool, optional): when True, remaining cations are added to last site

        """
        if self.mineral.has_structure:
            res = []
            for ix, row in self.df.iterrows():
                res.append(self.mineral.apfu(row, force=force))
            return APFU(
                pd.DataFrame(res, index=self._data.index),
                mineral=self.mineral,
                name=self.name,
                desc=self.desc,
            )
        else:
            raise TypeError(f"{self.mineral} has no structure defined")

    @property
    def reminder(self):
        """Returns reminding cations"""
        return self.df - self.mineral_apfu().df

    @property
    def error(self):
        """Returns percentage error of calculated cations"""
        return 100 * abs(self.mineral.ncat - self.sum) / self.sum

    def table(self, add_total=True, transpose=True):
        df = self.df
        if add_total:
            df["Total"] = self.sum
        ox = pd.Series(len(self) * [self.mineral.noxy], index=df.index, name="Oxygen")
        df = pd.concat([ox, df], axis=1)
        if transpose:
            df = df.T
        return df
