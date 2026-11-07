"""
Maintain pyrulearn/experiments/catalog.json, the dataset catalog behind
`pyrulearn.experiments.catalog`.

The catalog's curated fields (name, openml_id, target, tags, uci/kaggle
links, nominal, missing_markers, notes) are edited by hand; this script
fills in everything else from the sources themselves, so no statistic is
ever typed in:

- from OpenML's metadata (no download): name, version, license, original
  source URL, attribute counts by declared type (with the entry's
  `nominal` overrides applied);
- from the data itself (downloaded through the same cache
  `CatalogEntry.load` uses; skipped above --max-rows): rows, classes,
  majority-class rate and missing-value rate *after* the entry's own
  loading fixes -- which also proves every entry loads;
- for UCI-only entries (no OpenML id): rows/attributes from UCI's API;
- for CSV entries (`csv_url`): everything from the file itself, a
  boolean or text column counting as nominal.

It also keeps the ``cc18`` tag in sync with the OpenML-CC18 suite (study
99): members get the tag, and ``--add-cc18`` adds members not yet in the
catalog, except the image/signal/text datasets with more than
CC18_MAX_FEATURES attributes, which make little sense for rule learning.

Finally it prints a review list: numeric columns that look like integer
category codes (candidates for an entry's `nominal`) and anything that
failed.

Usage (from the repository root, needs network access):
    python tools/build_catalog.py [--add-cc18] [--no-data] [--max-rows N] [--only name,...]
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyrulearn.experiments.catalog import Catalog, CatalogEntry, fetch_csv_frame, fetch_openml_frame  # noqa: E402

CATALOG = os.path.join(os.path.dirname(__file__), "..", "pyrulearn", "experiments", "catalog.json")
OPENML = "https://www.openml.org/api/v1/json"
UCI = "https://archive.ics.uci.edu/api/dataset"
CC18_STUDY = 99
CC18_MAX_FEATURES = 200
#: a numeric column with this many or fewer distinct integer values (and
#: at least 3) is reported as a possible category code
CODE_MAX_DISTINCT = 10


def get_json(url, tries=4):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def openml_metadata(entry: CatalogEntry) -> dict:
    desc = get_json(f"{OPENML}/data/{entry.openml_id}")["data_set_description"]
    feats = get_json(f"{OPENML}/data/features/{entry.openml_id}")["data_features"]["feature"]
    unknown = set(entry.rename) - {f["name"] for f in feats}
    if unknown:
        raise ValueError(f"`rename` names columns that don't exist: {sorted(unknown)}")
    for f in feats:  # every curated field uses the renamed names
        f["name"] = entry.rename.get(f["name"], f["name"])
    default = desc.get("default_target_attribute")
    target = entry.target or entry.rename.get(default, default)
    if not target:
        raise ValueError("OpenML sets no default target -- give the entry a `target`")
    names = {f["name"] for f in feats}
    for col in [target] + list(entry.drop) + list(entry.nominal) + list(entry.missing_markers):
        if col not in names:
            raise ValueError(f"curated column {col!r} doesn't exist")
    attrs = [f for f in feats
             if f["name"] != target and f["name"] not in entry.drop
             and f["is_ignore"] != "true" and f["is_row_identifier"] != "true"]
    numeric = [f["name"] for f in attrs if f["data_type"] == "numeric" and f["name"] not in entry.nominal]
    strings = [f["name"] for f in attrs if f["data_type"] == "string"]
    target_feat = next(f for f in feats if f["name"] == target)
    return {
        "openml_name": desc["name"],
        "openml_version": int(desc["version"]),
        "original_url": desc.get("original_data_url") or None,
        "license": desc.get("licence") or None,
        "n_features": len(attrs),
        "n_numeric": len(numeric),
        "n_nominal": len(attrs) - len(numeric),
        "_target": target,
        "_classes": target_feat.get("nominal_value"),
        "_review": [f"string (free-text?) columns kept: {strings}"] if strings else [],
    }


def openml_qualities(entry: CatalogEntry) -> dict:
    qs = get_json(f"{OPENML}/data/qualities/{entry.openml_id}")["data_qualities"]["quality"]
    # a few qualities come back as lists (or empty) -- only scalars are needed
    return {q["name"]: float(q["value"]) for q in qs
            if isinstance(q.get("value"), (str, int, float)) and q["value"] != ""}


def from_data(entry: CatalogEntry, target: str) -> tuple:
    """(stats, review notes) measured on the loaded data."""
    import pandas as pd

    frame, _default = fetch_openml_frame(entry.openml_id)
    df = entry.prepare(frame, target, drop_degenerate=False)
    y = df[target]
    attrs = df.drop(columns=[target])
    stats = {
        "n_instances": len(df),
        "n_classes": int(y.nunique()),
        "majority_rate": round(float(y.value_counts(normalize=True).iloc[0]), 4),
        "missing_rate": round(float(attrs.isna().any(axis=1).mean()), 4),
    }
    review = []
    for col in attrs.columns:
        s = attrs[col].dropna()
        if col in entry.nominal or not pd.api.types.is_numeric_dtype(s) or s.empty:
            continue
        distinct = s.unique()
        if 3 <= len(distinct) <= CODE_MAX_DISTINCT and all(float(v).is_integer() for v in distinct):
            review.append(f"{col} ({len(distinct)} integer values: {sorted(int(v) for v in distinct)[:10]})")
    return stats, review


def fill_openml(entry: CatalogEntry, with_data: bool, max_rows: int) -> list:
    meta = openml_metadata(entry)
    target, classes, review_meta = meta.pop("_target"), meta.pop("_classes"), meta.pop("_review")
    for k, v in meta.items():
        setattr(entry, k, v)
    q = openml_qualities(entry)
    rows = int(q.get("NumberOfInstances", 0))
    if with_data and rows <= max_rows:
        stats, review = from_data(entry, target)
    else:
        # metadata only: OpenML's class statistics refer to its default target
        default = entry.target is None
        stats = {
            "n_instances": rows,
            "n_classes": int(q["NumberOfClasses"]) if default and "NumberOfClasses" in q
            else (len(classes) if classes else None),
            "majority_rate": round(q["MajorityClassPercentage"] / 100, 4)
            if default and "MajorityClassPercentage" in q else None,
            "missing_rate": round(q["NumberOfInstancesWithMissingValues"] / rows, 4)
            if rows and "NumberOfInstancesWithMissingValues" in q and not entry.missing_markers else None,
        }
        review = [f"statistics from OpenML metadata only ({rows} rows)"] if with_data else []
    for k, v in stats.items():
        setattr(entry, k, v)
    return review_meta + review


def fill_csv(entry: CatalogEntry) -> list:
    import pandas as pd

    df = entry.prepare(fetch_csv_frame(entry.csv_url), entry.target, drop_degenerate=False)
    y, attrs = df[entry.target], df.drop(columns=[entry.target])
    numeric = [c for c in attrs.columns
               if pd.api.types.is_numeric_dtype(attrs[c]) and not pd.api.types.is_bool_dtype(attrs[c])
               and c not in entry.nominal]
    entry.n_instances = len(df)
    entry.n_features = attrs.shape[1]
    entry.n_numeric = len(numeric)
    entry.n_nominal = attrs.shape[1] - len(numeric)
    entry.n_classes = int(y.nunique())
    entry.majority_rate = round(float(y.value_counts(normalize=True).iloc[0]), 4)
    entry.missing_rate = round(float(attrs.isna().any(axis=1).mean()), 4)
    return []


def fill_uci(entry: CatalogEntry) -> list:
    uci_id = int(entry.uci.rstrip("/").split("/dataset/")[1].split("/")[0])
    d = get_json(f"{UCI}?id={uci_id}")["data"]
    entry.n_instances = d.get("num_instances")
    entry.n_features = d.get("num_features") or None
    entry.license = (d.get("license") or None)
    return []


def sync_cc18(cat: Catalog, add: bool) -> list:
    study = get_json(f"{OPENML}/study/{CC18_STUDY}")["study"]
    ids = [int(i) for i in study["data"]["data_id"]]
    listing = get_json(f"{OPENML}/data/list/data_id/{','.join(map(str, ids))}")["data"]["dataset"]
    notes = []
    by_id = {e.openml_id: e for e in cat.entries if e.openml_id is not None}
    for d in listing:
        did = int(d["did"])
        q = {x["name"]: float(x["value"]) for x in d.get("quality", []) if x.get("value") not in (None, "")}
        n_attrs = int(q.get("NumberOfFeatures", 0)) - 1
        if did in by_id:
            if "cc18" not in by_id[did].tags:
                by_id[did].tags.append("cc18")
        elif n_attrs > CC18_MAX_FEATURES:
            notes.append(f"cc18 member {d['name']} ({did}) left out: {n_attrs} attributes")
        elif add:
            if d["name"] in cat:
                raise ValueError(f"cc18 member {d['name']} ({did}) clashes with an existing entry's name")
            entry = CatalogEntry(name=d["name"], openml_id=did, tags=["cc18"])
            cat.entries.append(entry)
            cat._by_name[entry.name] = entry
            notes.append(f"added cc18 member {d['name']} ({did})")
        else:
            notes.append(f"cc18 member {d['name']} ({did}) not in the catalog (use --add-cc18)")
    return notes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--add-cc18", action="store_true", help="add CC18 members not yet in the catalog")
    ap.add_argument("--no-data", action="store_true", help="metadata only, never download data")
    ap.add_argument("--max-rows", type=int, default=1_000_000,
                    help="download data only for datasets with at most this many rows")
    ap.add_argument("--only", default=None, help="comma-separated entry names to refresh")
    args = ap.parse_args()

    cat = Catalog.from_file(CATALOG)
    report = sync_cc18(cat, args.add_cc18)
    only = set(args.only.split(",")) if args.only else None
    for entry in cat.entries:
        if only and entry.name not in only:
            continue
        print(f"{entry.name} ...", flush=True)
        try:
            notes = (fill_openml(entry, not args.no_data, args.max_rows) if entry.openml_id is not None
                     else fill_csv(entry) if entry.csv_url and not args.no_data
                     else fill_uci(entry) if entry.uci else ["no source to fill statistics from"])
        except Exception as e:  # noqa: BLE001 -- report, keep the rest going
            notes = [f"FAILED: {type(e).__name__}: {e}"]
        report += [f"{entry.name}: {n}" for n in notes]

    cat.entries.sort(key=lambda e: e.name.lower())
    with open(CATALOG, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cat.to_json(), f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"\nwrote {len(cat)} entries to {os.path.normpath(CATALOG)}")
    if report:
        print("\nreview:")
        for line in report:
            print("  " + line)


if __name__ == "__main__":
    main()
