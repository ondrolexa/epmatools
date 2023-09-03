import numpy as np
import pandas as pd


class Site:
    def __init__(self, name, ncat, candidates):
        self.name = name
        self.ncat = ncat
        self.candidates = candidates
        self.atoms = {}

    def __repr__(self):
        return f"Site {self.name}[{self.ncat}][{sum(self.atoms.values())}]"

    def add(self, atom, amount, force=False):
        if force:
            if atom in self.atoms:
                self.atoms[atom] += amount
            else:
                self.atoms[atom] = amount
        else:
            free = self.free
            if free > 0:
                if free > amount:
                    self.atoms[atom] = amount
                else:
                    self.atoms[atom] = free

    def get(self, atom):
        return self.atoms.get(atom, 0)

    @property
    def free(self):
        return self.ncat - sum(self.atoms.values())


class StrucForm:
    def __init__(self, mineral):
        self.mineral = mineral
        self.sites = [Site(*s) for s in self.mineral.structure]
        self.reminder = None

    def get(self, atom):
        return sum([s.get(atom) for s in self.sites])

    @property
    def apfu(self):
        if self.reminder is not None:
            apfu = pd.Series(
                [self.get(e) for e in self.reminder.index], index=self.reminder.index
            )
            # Add atoms not in analysis
            for site in self.sites:
                for atom in site.candidates:
                    if atom not in apfu:
                        apfu[atom] = 0.0
            return apfu
        else:
            raise ValueError("Not yet calculated")

    def info(self):
        if self.reminder is not None:
            res = ""
            for site in sorted(self.sites, key=lambda site: site.name):
                s = ""
                for atom, val in site.atoms.items():
                    s += f"{atom}({val:.3f})"
                res += f"[{s}]"
            return res
        else:
            raise ValueError("Not yet calculated")

    def check_stechiometry(self):
        if self.reminder is not None:
            return np.mean([abs(s.free) / s.ncat for s in self.sites])
        else:
            raise ValueError("Not yet calculated")


class Mineral:
    def __init__(self):
        if self.has_structure:
            self.ncat = sum([s[1] for s in self.structure])

    def __repr__(self):
        return type(self).__name__

    @property
    def has_endmembers(self):
        return hasattr(self, "endmembers")

    @property
    def has_structure(self):
        return hasattr(self, "structure")

    def calculate(self, cations, force=False):
        sf = StrucForm(self)
        lastsite = {}
        for site in sf.sites:
            for atom in site.candidates:
                available = cations.get(atom, 0.0) - sf.get(atom)
                if available > 0:
                    site.add(atom, available)
                    lastsite[atom] = site

        # Force site occupancy
        if force:
            occ = pd.Series([sf.get(e) for e in cations.index], index=cations.index)
            reminder = cations - occ
            for atom, r in reminder.items():
                if (r > 0) & (atom in lastsite):
                    lastsite[atom].add(atom, r, force=True)
        # reminder
        occ = pd.Series([sf.get(e) for e in cations.index], index=cations.index)
        sf.reminder = cations - occ
        return sf

    def apfu(self, cations, force=False):
        sf = self.calculate(cations, force=force)
        return sf.apfu


class Garnet_Fe2(Mineral):
    """Garnet using total Fe with 4 endmembers"""

    def __init__(self):
        self.noxy = 12
        self.needsFe3 = False
        # fmt: off
        self.structure = (
            ("Z", 3, ["Si{4+}", "Al{3+}"]),
            ("Y", 2, ["Si{4+}", "Al{3+}", "Ti{4+}", "Cr{3+}", "Mg{2+}", "Fe{2+}", "Mn{2+}"]),  # noqa: E501
            ("X", 3, ["Y{3+}", "Mg{2+}", "Fe{2+}", "Mn{2+}", "Ca{2+}", "Na{+}"]),
        )
        # fmt: on
        super().__init__()

    def endmembers(self, cations, force=False):
        apfu = self.apfu(cations, force=force)
        esum = apfu["Fe{2+}"] + apfu["Mn{2+}"] + apfu["Mg{2+}"] + apfu["Ca{2+}"]
        em = dict(
            Alm=apfu["Fe{2+}"] / esum,
            Prp=apfu["Mg{2+}"] / esum,
            Sps=apfu["Mn{2+}"] / esum,
            Grs=apfu["Ca{2+}"] / esum,
        )
        return pd.Series(em)


class Garnet_Fe3(Mineral):
    """Garnet using Fe2 and Fe3 with 6 endmembers"""

    def __init__(self):
        self.noxy = 12
        self.needsFe3 = True
        # fmt: off
        self.structure = (
            ("Z", 3, ["Si{4+}", "Al{3+}", "Fe{3+}"]),
            ("Y", 2, ["Si{4+}", "Al{3+}", "Ti{4+}", "Cr{3+}", "Fe{3+}", "Mg{2+}", "Fe{2+}", "Mn{2+}"]),  # noqa: E501
            ("X", 3, ["Y{3+}", "Mg{2+}", "Fe{2+}", "Mn{2+}", "Ca{2+}", "Na{+}"]),
        )
        # fmt: on
        super().__init__()

    def endmembers(self, cations, force=False):
        apfu = self.apfu(cations, force=force)
        s8t = apfu["Fe{2+}"] + apfu["Mn{2+}"] + apfu["Mg{2+}"] + apfu["Ca{2+}"]
        s6t = apfu["Ti{4+}"] + apfu["Al{3+}"] + apfu["Cr{3+}"] + apfu["Fe{3+}"]
        # end members in mol%
        em = dict(
            Alm=apfu["Fe{2+}"] / s8t,
            Prp=apfu["Mg{2+}"] / s8t,
            Sps=apfu["Mn{2+}"] / s8t,
            Grs=(apfu["Al{3+}"] / s6t) * (apfu["Ca{2+}"] / s8t),
            Adr=(apfu["Fe{3+}"] / s6t) * (apfu["Ca{2+}"] / s8t),
            Uv=(apfu["Cr{3+}"] / s6t) * (apfu["Ca{2+}"] / s8t),
            CaTi=(apfu["Ti{4+}"] / s6t) * (apfu["Ca{2+}"] / s8t),
        )
        return pd.Series(em)


class Feldspar(Mineral):
    """Feldspar with 3 endmembers"""

    def __init__(self):
        self.noxy = 8
        self.needsFe3 = False
        self.structure = (
            ("T", 4, ["Si{4+}", "Al{3+}"]),
            ("A", 1, ["K{+}", "Na{+}", "Ca{2+}"]),
        )
        super().__init__()

    def endmembers(self, cations, force=False):
        apfu = self.apfu(cations, force=force)
        alk = apfu["Ca{2+}"] + apfu["Na{+}"] + apfu["K{+}"]
        em = dict(
            An=apfu["Ca{2+}"] / alk,
            Ab=apfu["Na{+}"] / alk,
            Or=apfu["K{+}"] / alk,
        )
        return pd.Series(em)


class Pyroxene_Fe2(Mineral):
    """Pyroxen using total Fe with 3 endmembers"""

    def __init__(self):
        self.noxy = 6
        self.needsFe3 = False
        self.structure = (
            ("T", 2, ["Si{4+}", "Al{3+}"]),
            ("M1", 1, ["Al{3+}", "Ti{4+}", "Cr{3+}", "Mn{2+}", "Mg{2+}", "Fe{2+}"]),
            ("M2", 1, ["Mg{2+}", "Fe{2+}", "Ca{2+}", "Na{+}", "K{+}"]),
        )
        super().__init__()

    def endmembers(self, cations, force=False):
        apfu = self.apfu(cations, force=force)
        esum = apfu["Fe{2+}"] + apfu["Mn{2+}"] + apfu["Mg{2+}"] + apfu["Ca{2+}"]
        em = dict(
            En=apfu["Mg{2+}"] / esum,
            Wo=apfu["Ca{2+}"] / esum,
            Fs=(apfu["Fe{2+}"] + apfu["Mn{2+}"]) / esum,
        )
        return pd.Series(em)


class Pyroxene_Fe3(Mineral):
    """Pyroxene with Na-Cr with 6 endmembers"""

    def __init__(self):
        self.noxy = 6
        self.needsFe3 = True
        # fmt: off
        self.structure = (
            ("T", 2, ["Si{4+}", "Al{3+}", "Fe{3+}"]),
            ("M1", 1, ["Al{3+}", "Ti{4+}", "Fe{3+}", "Cr{3+}", "Mn{2+}", "Mg{2+}", "Fe{2+}"]),  # noqa: E501
            ("M2", 1, ["Mg{2+}", "Fe{2+}", "Ca{2+}", "Na{+}", "K{+}"]),
        )
        # fmt: on
        super().__init__()

    def endmembers(self, cations, force=False):
        apfu = self.apfu(cations, force=force)
        A = np.array(
            [
                apfu["Ca{2+}"],
                apfu["Al{3+}"],
                apfu["Fe{3+}"],
                apfu["Mg{2+}"],
                apfu["Fe{2+}"],
                apfu["Cr{3+}"],
            ]
        )
        M = np.array(
            [
                [0, 0, 0, 2, 0, 0],
                [2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        em = A @ np.linalg.inv(M)
        em = em / em.sum()
        return pd.Series(em, index=["En", "Wo", "Fs", "Jd", "Aeg", "Kos"])
