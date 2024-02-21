from pathlib import Path
import re
import numpy as np
import pandas as pd


def read_actlabs(src. **kwargs):
    if 'header' not in kwargs:
        kwargs['header'] = 2
    df = pd.read_excel(src, **kwargs)
    units = df.iloc[0]
    limits = df.iloc[1]
    method = df.iloc[2]
    df = df.rename(columns={"Analyte Symbol": "Sample"})[3:].set_index("Sample")
    # replace detection limits
    for col in df:
        ix = df[col].astype(str).str.startswith("< ")
        if any(ix):
            df.loc[ix, col] = np.nan

    df = df.astype(float)
    return df, units, limits, method


def read_bureau_veritas(src=""):
    df = pd.read_excel(src, skiprows=9)
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


def from_upsg_eds(src, merged=False, astype=int, name=False):
    """Reads map data from EDS export.

    Args:
        src (str): path to directory with exported data
        merged (bool): Must be True for merged export. Default False
        astype (dtype): dtype of imported maps. Default int is suitable
            for maps with counts, otherwise should be float
        name (bool): When True, function returns also name of the mapset
            derived from directory stem

    Returns:
        maps (dict): Dictionary of maps (suitable for Mapset)

    """
    src = Path(src)
    assert src.is_dir(), "Source path must be dir"
    if merged:
        csvs = [p for p in src.iterdir() if p.suffix == ".csv"]
        assert len(csvs) == 1, "There must be only one csv file in stacked export"
        df = pd.read_csv(csvs[0])
        df.columns = df.columns.str.strip()
    cnds = [p for p in src.iterdir() if p.suffix == ".cnd"]
    maps = {}
    shape = None
    for ix, cnd in enumerate(cnds):
        meta = {}
        with open(cnd, "r") as f:
            for ln in f.readlines():
                if ln.startswith("$"):
                    tags = ln.strip().split()
                    key = tags[0].split("$")[1].split("%")[0]
                    if len(tags) > 1:
                        meta[key] = tags[1:]
                    else:
                        meta[key] = None
        if "XM_AP_SA_PIXELS" in meta:
            cols, rows = map(int, meta["XM_AP_SA_PIXELS"])
        else:
            raise ValueError(f"No $XM_AP_SA_PIXELS in {cnd}")
        if "XM_ELEM_NAME" in meta:
            if meta["XM_ELEM_NAME"] is not None:
                element = meta["XM_ELEM_NAME"][0]
            else:
                if "XM_ELEM_IMS_SIGNAL_TYPE" in meta:
                    element = meta["XM_ELEM_IMS_SIGNAL_TYPE"][0]
                else:
                    raise ValueError(f"No $XM_ELEM_IMS_SIGNAL_TYPE in {cnd}")
        else:
            raise ValueError(f"No $XM_ELEM_NAME in {cnd}")
        if shape is None:
            shape = (rows, cols)
        else:
            if shape != (rows, cols):
                raise ValueError("All linedata must have same shape")
        if merged:
            maps[element] = df[element].values.reshape((rows, cols)).astype(astype)
        else:
            csv = cnd.with_suffix(".csv")
            if not csv.exists():
                raise ValueError(f"No {csv.stem}.csv file found for {cnd.stem}.cnd")
            values = np.loadtxt(csv, delimiter=",")
            if values.ndim == 1:
                values = values.reshape((rows, cols)).astype(astype)
            else:
                if shape == values.shape:
                    values = values.astype(astype)
                else:
                    raise ValueError("Matrix shape do not correspond metafile info")
            maps[element] = values
        print(f"{ix + 1}/{len(cnds)} {element} parsed...")
    if name:
        return maps, src.stem
    else:
        return maps


def from_line_data_separated(src):
    src = Path(src)
    assert src.is_dir(), "Source path must be dir"
    linedirs = [p for p in src.iterdir() if p.is_dir()]
    # identify prefix and suffix
    stems = [ld.stem for ld in linedirs]
    n = 0
    while all([ld.startswith(stems[0][:n]) for ld in stems]):
        n += 1
    prefix = stems[0][: n - 1]
    n = 0
    while all([ld.endswith(stems[0][::-1][:n][::-1]) for ld in stems]):
        n += 1
    suffix = stems[0][::-1][: n - 1][::-1]
    #
    maps = {}
    shape = None
    for ix, linedir in enumerate(linedirs):
        cnd = next(linedir.rglob("*.cnd"))
        ok = False
        with open(cnd, "r") as f:
            for ln in f.readlines():
                if ln.startswith("$XM_AP_SA_PIXELS%"):
                    _, cols, rows = map(int, ln[17:].split())
                    ok = True
        assert ok, "No $XM_AP_SA_PIXELS% keyword in metafile"
        if shape is None:
            shape = (rows, cols)
        else:
            assert shape == (rows, cols), "All linedata must have same shape"
        csv = next(linedir.rglob("*.csv"))
        els = re.findall(f"{prefix}(.*?){suffix}", linedir.stem)
        assert len(els) == 1, "Element name parsing error"
        maps[els[0]] = np.loadtxt(csv).reshape((rows, cols)).astype(int)
        print(f"{ix+1}/{len(linedirs)} {els[0]} parsed...")
    return maps, src.stem
