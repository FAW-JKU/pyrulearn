"""Generated datasets: data defined by a rule rather than downloaded.

`parity` -- the parity of a few relevant Boolean features among
irrelevant ones: a flat DNF needs 2^(k-1) rules of k conditions, a deep
network builds it from small parts (XOR and XNOR of pairs, then of those).
`Parity` is the same as a dataset for `run_cv`.

`krk_illegal` -- the KRK illegality task of the ILP literature (Muggleton,
Bain, Hayes-Michie & Michie, "An experimental comparison of human and
machine learning formalisms", ML 1989): a chess position of white king,
white rook and black king, white to move; is it illegal? The six
coordinates are nominal attributes, so a learner sees only "this piece is
on that file", never "same file" or "adjacent" -- the relations the ILP
systems got as background knowledge, which a deep network would have to
build in its hidden nodes. With ``relations=True`` the ILP systems'
background knowledge comes along as relational features (`eq`, `adj`,
`lt` between the coordinates of two pieces), so a propositional learner
has what the relational ones had. `KRKIllegal` is the same as a dataset
for `pyrulearn.experiments.runner.run_cv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

#: the attributes, in the order of the ILP literature: file and rank of the
#: white king, the white rook, the black king
KRK_ATTRIBUTES = ("wkf", "wkr", "wrf", "wrr", "bkf", "bkr")
FILES = "abcdefgh"
RANKS = "12345678"
#: the pairs of pieces the relations compare
KRK_PAIRS = (("wk", "wr"), ("wk", "bk"), ("wr", "bk"))


def krk_is_illegal(wkf, wkr, wrf, wrr, bkf, bkr) -> np.ndarray:
    """Whether positions (coordinates 0-7, array-like) are illegal with
    white to move: two pieces on one square, the kings next to each other,
    or the black king in check -- the rook on its file or rank with the
    white king not between them."""
    wkf, wkr, wrf, wrr, bkf, bkr = (np.asarray(v) for v in (wkf, wkr, wrf, wrr, bkf, bkr))
    same_square = (((wkf == wrf) & (wkr == wrr)) | ((wkf == bkf) & (wkr == bkr))
                   | ((wrf == bkf) & (wrr == bkr)))
    kings_adjacent = (np.abs(wkf - bkf) <= 1) & (np.abs(wkr - bkr) <= 1)

    def between(a, x, b):
        return ((a < x) & (x < b)) | ((b < x) & (x < a))

    file_check = (wrf == bkf) & ~((wkf == wrf) & between(wrr, wkr, bkr))
    rank_check = (wrr == bkr) & ~((wkr == wrr) & between(wrf, wkf, bkf))
    return same_square | kings_adjacent | file_check | rank_check


def krk_illegal(n: Optional[int] = 1000, random_state: Optional[int] = 0, relations: bool = False,
                coordinates: bool = True):
    """A `pandas.DataFrame` of `n` distinct KRK positions drawn uniformly
    from all 8^6 = 262,144 (all of them, in order, for ``n=None``): the six
    coordinates as nominal values (files ``"a"``-``"h"``, ranks ``"1"``-
    ``"8"``) and ``class`` -- ``"illegal"`` or ``"legal"``. About a third
    are illegal.

    `relations` adds the background knowledge of the ILP literature as 24
    Boolean relational columns: for each pair of pieces (`KRK_PAIRS`) and
    each of file and rank, ``eq(x,y)`` (the same), ``adj(x,y)`` (one
    apart) and ``lt(x,y)``, ``lt(y,x)`` (smaller), e.g. ``eq(wrf,bkf)``:
    rook and black king on one file. They are declared in
    ``frame.attrs["relational"]`` (sources and expression), so
    `pyrulearn.data.io.build_dataspec` makes them RELATIONAL attributes.
    With them the task has a short exact theory (same square, adjacent
    kings, a check not blocked), which bounds what a learner can reach
    without them. ``coordinates=False`` leaves out the six coordinates
    (only with `relations`): the background knowledge alone."""
    import pandas as pd

    if not coordinates and not relations:
        raise ValueError("without coordinates, krk_illegal needs relations=True")
    total = 8 ** 6
    if n is None:
        codes = np.arange(total)
    else:
        if not 0 < n <= total:
            raise ValueError(f"n must be between 1 and {total}, got {n}")
        codes = np.sort(np.random.default_rng(random_state).choice(total, size=n, replace=False))
    coords = [(codes // 8 ** (5 - i)) % 8 for i in range(6)]
    illegal = krk_is_illegal(*coords)
    frame = {}
    for name, values in zip(KRK_ATTRIBUTES, coords if coordinates else ()):
        letters = np.array(list(FILES if name.endswith("f") else RANKS))
        frame[name] = pd.Categorical(letters[values], categories=list(letters))
    relational = {}
    if relations:
        index = dict(zip(KRK_ATTRIBUTES, coords))
        for p, q in KRK_PAIRS:
            for d in "fr":
                x, y = p + d, q + d
                a, b = index[x], index[y]
                for name, values, expression in (
                        (f"eq({x},{y})", a == b, f"{x} = {y}"),
                        (f"adj({x},{y})", np.abs(a - b) == 1, f"|{x} - {y}| = 1"),
                        (f"lt({x},{y})", a < b, f"{x} < {y}"),
                        (f"lt({y},{x})", b < a, f"{y} < {x}")):
                    frame[name] = values
                    relational[name] = {"sources": [x, y], "expression": expression}
    frame["class"] = np.where(illegal, "illegal", "legal")
    df = pd.DataFrame(frame)
    if relational:
        df.attrs["relational"] = relational
    return df


@dataclass(frozen=True)
class KRKIllegal:
    """`krk_illegal(n, random_state, relations, coordinates)` as a dataset
    for `run_cv` (its `name` -- which keys the cache -- says how it was
    drawn)."""

    n: Optional[int] = 1000
    random_state: Optional[int] = 0
    relations: bool = False
    coordinates: bool = True
    openml_id = None
    openml_version = None
    target = "class"

    @property
    def name(self) -> str:
        return "krk-illegal" + ("" if self.n is None else f"-{self.n}") + (
            "" if self.random_state in (None, 0) else f"-seed{self.random_state}") + (
            "-relational" if self.relations else "") + ("" if self.coordinates else "-only")

    def load(self, data_home: Optional[str] = None, drop_degenerate: bool = True) -> Tuple[object, str]:
        return krk_illegal(self.n, self.random_state, self.relations, self.coordinates), self.target


def parity(n_relevant: int = 4, n_irrelevant: int = 6, n: Optional[int] = None, noise: float = 0.0,
           random_state: Optional[int] = 0):
    """A `pandas.DataFrame` of Boolean features ``x1``, ``x2``, ... (dtype
    bool) and ``class``: ``"odd"`` if an odd number of the first
    `n_relevant` features is true, else ``"even"``; the other
    `n_irrelevant` features don't matter. All 2^(n_relevant +
    n_irrelevant) rows, in order (``n=None``), or `n` distinct ones drawn
    uniformly. `noise` flips each label with that probability."""
    import pandas as pd

    d = n_relevant + n_irrelevant
    if n_relevant < 1 or n_irrelevant < 0:
        raise ValueError("parity needs at least one relevant feature and no negative number of irrelevant ones")
    if not 0 <= noise <= 1:
        raise ValueError(f"noise must be in [0, 1], got {noise!r}")
    rng = np.random.default_rng(random_state)
    total = 2 ** d
    if n is None:
        codes = np.arange(total)
    else:
        if not 0 < n <= total:
            raise ValueError(f"n must be between 1 and {total}, got {n}")
        codes = np.sort(rng.choice(total, size=n, replace=False))
    bits = ((codes[:, None] >> np.arange(d - 1, -1, -1)) & 1).astype(bool)
    odd = bits[:, :n_relevant].sum(axis=1) % 2 == 1
    if noise:
        odd = odd ^ (rng.random(len(odd)) < noise)
    frame = {f"x{i + 1}": bits[:, i] for i in range(d)}
    frame["class"] = np.where(odd, "odd", "even")
    return pd.DataFrame(frame)


@dataclass(frozen=True)
class Parity:
    """`parity(...)` as a dataset for `run_cv` (its `name` -- which keys
    the cache -- says how it was made)."""

    n_relevant: int = 4
    n_irrelevant: int = 6
    n: Optional[int] = None
    noise: float = 0.0
    random_state: Optional[int] = 0
    openml_id = None
    openml_version = None
    target = "class"

    @property
    def name(self) -> str:
        out = f"parity-{self.n_relevant}-of-{self.n_relevant + self.n_irrelevant}"
        if self.n is not None:
            out += f"-n{self.n}"
        if self.noise:
            out += f"-noise{self.noise:g}"
        if self.random_state not in (None, 0) and (self.n is not None or self.noise):
            out += f"-seed{self.random_state}"
        return out

    def load(self, data_home: Optional[str] = None, drop_degenerate: bool = True) -> Tuple[object, str]:
        return parity(self.n_relevant, self.n_irrelevant, self.n, self.noise, self.random_state), self.target
