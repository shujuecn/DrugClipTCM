#!/usr/bin/env python3
"""
Build retrieval assets from tabular TCM libraries.

Supported libraries:
- HIT
- HERB 2.0 mapping CSV
- BatmanTCM2.0 (requires CID -> SMILES resolution)

Outputs per library:
- standardized compound table CSV
- retrieval LMDB compatible with unimol/retrieval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Iterable
from urllib import error, parse, request

import numpy as np
import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DICT_PATH = REPO_ROOT / "data_dict" / "dict_mol.txt"
DEFAULT_DATA_DIR = REPO_ROOT / "data"

RDLogger.DisableLog("rdApp.*")

LIBRARY_CONFIGS = {
    "hit": {
        "default_input": DEFAULT_DATA_DIR / "HIT数据库全部.xlsx",
        "source_db": "HIT",
    },
    "herb2": {
        "default_input": DEFAULT_DATA_DIR / "herb2.0所有中药-成分-靶点映射关系.csv",
        "source_db": "HERB2",
    },
    "batman": {
        "default_input": DEFAULT_DATA_DIR / "BatmanTCM2.0全部数据.xlsx",
        "source_db": "BatmanTCM2.0",
    },
}

TABLE_FIELDS = [
    "library",
    "source_db",
    "compound_key",
    "source_compound_id",
    "pubchem_cid",
    "chembl_id",
    "compound_names",
    "canonical_smiles",
    "inchikey",
    "inchi",
    "formula",
    "mol_weight",
    "exact_weight",
    "atom_count",
    "heavy_atom_count",
    "formal_charge",
    "unknown_atom_count",
    "unknown_atoms",
    "herb_count",
    "herb_ids",
    "herb_names",
    "target_count",
    "target_ids",
    "target_symbols",
    "target_names",
    "evidence",
    "row_count",
    "parse_status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retrieval assets from HIT/HERB2/Batman tabular libraries.",
    )
    parser.add_argument(
        "--library",
        required=True,
        choices=sorted(LIBRARY_CONFIGS),
        help="Library to build.",
    )
    parser.add_argument(
        "--input-path",
        default="",
        help="Override the default library file path.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for generated assets. Defaults to data/<library>_assets.",
    )
    parser.add_argument(
        "--dict-path",
        default=str(DEFAULT_DICT_PATH),
        help="DrugCLIP molecule dictionary path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit unique compounds for smoke testing. 0 means all.",
    )
    parser.add_argument(
        "--skip-lmdb",
        action="store_true",
        help="Only export the standardized table.",
    )
    parser.add_argument(
        "--skip-unknown-atoms",
        action="store_true",
        help="Skip molecules containing atoms outside dict_mol.txt.",
    )
    parser.add_argument(
        "--resolve-cids",
        action="store_true",
        help="Resolve PubChem CIDs to SMILES for Batman or other CID-only sources.",
    )
    parser.add_argument(
        "--cid-cache-csv",
        default="",
        help="CSV cache for CID -> SMILES mappings. Reused and updated when --resolve-cids is set.",
    )
    parser.add_argument(
        "--cid-batch-size",
        type=int,
        default=100,
        help="Batch size for CID -> SMILES PubChem requests.",
    )
    parser.add_argument(
        "--cid-timeout",
        type=int,
        default=30,
        help="HTTP timeout seconds for CID resolution.",
    )
    parser.add_argument(
        "--cid-retries",
        type=int,
        default=2,
        help="Retry count per CID batch.",
    )
    return parser.parse_args()


def read_dictionary(path: Path) -> set[str]:
    allowed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token and not token.startswith("["):
                allowed.add(token)
    return allowed


def ensure_list(values: Iterable[str | None]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "null":
            out.append(text)
    return sorted(set(out))


def first_non_empty(values: Iterable[str | None]) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "null":
            return text
    return ""


def load_library_table(library: str, input_path: Path) -> pl.DataFrame:
    if library == "hit":
        df = pl.read_excel(input_path).select(
            [
                pl.lit("HIT").alias("source_db"),
                pl.col("Compound Id").cast(pl.Utf8).alias("source_compound_id"),
                pl.col("Compound Name").cast(pl.Utf8).alias("compound_name"),
                pl.col("Pubchem ID").cast(pl.Utf8).alias("pubchem_cid"),
                pl.col("Chembl ID").cast(pl.Utf8).alias("chembl_id"),
                pl.col("Smiles").cast(pl.Utf8).alias("raw_smiles"),
                pl.col("Formula").cast(pl.Utf8).alias("formula_input"),
                pl.col("Herb ID").cast(pl.Utf8).alias("herb_id"),
                pl.col("Chinese Character").cast(pl.Utf8).alias("herb_name"),
                pl.col("Target ID").cast(pl.Utf8).alias("target_id"),
                pl.col("Symbol").cast(pl.Utf8).alias("target_symbol"),
                pl.col("Protein Name").cast(pl.Utf8).alias("target_name"),
                pl.col("Uniprot").cast(pl.Utf8).alias("target_uniprot"),
            ]
        )
    elif library == "herb2":
        df = pl.read_csv(input_path, infer_schema_length=2000).select(
            [
                pl.lit("HERB2").alias("source_db"),
                pl.col("Ingredient id").cast(pl.Utf8).alias("source_compound_id"),
                pl.col("Ingredient name").cast(pl.Utf8).alias("compound_name"),
                pl.lit(None, dtype=pl.Utf8).alias("pubchem_cid"),
                pl.lit(None, dtype=pl.Utf8).alias("chembl_id"),
                pl.col("Canonical smiles").cast(pl.Utf8).alias("raw_smiles"),
                pl.col("Molecular formula").cast(pl.Utf8).alias("formula_input"),
                pl.col("Herb id").cast(pl.Utf8).alias("herb_id"),
                pl.col("中药名").cast(pl.Utf8).alias("herb_name"),
                pl.col("Target id").cast(pl.Utf8).alias("target_id"),
                pl.col("Gene symbol").cast(pl.Utf8).alias("target_symbol"),
                pl.col("Protein name").cast(pl.Utf8).alias("target_name"),
                pl.col("Source").cast(pl.Utf8).alias("evidence"),
            ]
        )
    elif library == "batman":
        df = pl.read_excel(input_path).select(
            [
                pl.lit("BatmanTCM2.0").alias("source_db"),
                pl.col("CID").cast(pl.Utf8).alias("source_compound_id"),
                pl.col("Compound Name").cast(pl.Utf8).alias("compound_name"),
                pl.col("CID").cast(pl.Utf8).alias("pubchem_cid"),
                pl.lit(None, dtype=pl.Utf8).alias("chembl_id"),
                pl.lit(None, dtype=pl.Utf8).alias("raw_smiles"),
                pl.lit(None, dtype=pl.Utf8).alias("formula_input"),
                pl.lit(None, dtype=pl.Utf8).alias("herb_id"),
                pl.col("中文名").cast(pl.Utf8).alias("herb_name"),
                pl.col("Gene ID").cast(pl.Utf8).alias("target_id"),
                pl.col("Gene Name").cast(pl.Utf8).alias("target_symbol"),
                pl.col("Gene Name").cast(pl.Utf8).alias("target_name"),
                pl.col("Score").cast(pl.Utf8).alias("evidence"),
            ]
        )
    else:
        raise ValueError(f"Unsupported library: {library}")
    return df


def load_cid_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = pl.read_csv(path)
    if not {"cid", "canonical_smiles"}.issubset(set(rows.columns)):
        return {}
    return {
        str(cid): str(smiles)
        for cid, smiles in rows.select("cid", "canonical_smiles").iter_rows()
        if cid is not None and smiles is not None and str(smiles).strip()
    }


def save_cid_cache(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"cid": cid, "canonical_smiles": smiles} for cid, smiles in sorted(mapping.items())]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cid", "canonical_smiles"])
        writer.writeheader()
        writer.writerows(rows)


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def resolve_pubchem_cids(
    cids: list[str],
    batch_size: int,
    timeout: int,
    retries: int,
    cache: dict[str, str],
) -> dict[str, str]:
    unresolved = [cid for cid in cids if cid not in cache]
    if not unresolved:
        return cache

    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/CanonicalSMILES/JSON"
    for batch in chunked(unresolved, batch_size):
        joined = ",".join(batch)
        url = base.format(parse.quote(joined, safe=","))
        last_error = ""
        for attempt in range(retries + 1):
            try:
                req = request.Request(url, headers={"User-Agent": "DrugClipTCM CID resolver"})
                with request.urlopen(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                props = payload.get("PropertyTable", {}).get("Properties", [])
                for item in props:
                    cid = str(item.get("CID"))
                    smiles = item.get("CanonicalSMILES")
                    if cid and smiles:
                        cache[cid] = smiles
                break
            except Exception as exc:  # pylint: disable=broad-except
                last_error = f"{type(exc).__name__}:{exc}"
                if attempt == retries:
                    missing = ", ".join(batch[:5])
                    raise SystemExit(
                        f"Failed to resolve CID batch starting with [{missing}] via PubChem: {last_error}"
                    ) from exc
                time.sleep(1.5 * (attempt + 1))
    return cache


def attach_batman_smiles(
    df: pl.DataFrame,
    args: argparse.Namespace,
) -> pl.DataFrame:
    cache_path = (
        Path(args.cid_cache_csv)
        if args.cid_cache_csv
        else REPO_ROOT / "data" / "batman_cid_smiles_cache.csv"
    )
    cache = load_cid_cache(cache_path)
    cids = ensure_list(df.get_column("pubchem_cid").to_list())
    if args.resolve_cids:
        cache = resolve_pubchem_cids(
            cids,
            batch_size=args.cid_batch_size,
            timeout=args.cid_timeout,
            retries=args.cid_retries,
            cache=cache,
        )
        save_cid_cache(cache_path, cache)
    if not cache:
        raise SystemExit(
            "BatmanTCM2.0 needs CID -> SMILES mappings. "
            "Provide --resolve-cids (network required) or a populated --cid-cache-csv."
        )
    cache_df = pl.DataFrame(
        {"pubchem_cid": list(cache.keys()), "raw_smiles": list(cache.values())}
    )
    out = df.drop("raw_smiles").join(cache_df, on="pubchem_cid", how="left")
    resolved = int(out.select(pl.col("raw_smiles").is_not_null().sum()).item())
    if resolved == 0:
        raise SystemExit("No Batman CIDs could be resolved to SMILES.")
    return out


def aggregate_compounds(df: pl.DataFrame, library: str) -> list[dict[str, object]]:
    rows = [row for row in df.iter_rows(named=True)]
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        compound_id = first_non_empty([row.get("source_compound_id"), row.get("pubchem_cid")])
        smiles = first_non_empty([row.get("raw_smiles")])
        key = compound_id or smiles
        if not key:
            continue
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, object]] = []
    for key, items in groups.items():
        aggregated.append(
            {
                "library": library,
                "source_db": first_non_empty([item.get("source_db") for item in items]),
                "compound_key": key,
                "source_compound_id": first_non_empty([item.get("source_compound_id") for item in items]),
                "pubchem_cid": first_non_empty([item.get("pubchem_cid") for item in items]),
                "chembl_id": first_non_empty([item.get("chembl_id") for item in items]),
                "compound_names": ensure_list(item.get("compound_name") for item in items),
                "raw_smiles": first_non_empty([item.get("raw_smiles") for item in items]),
                "formula_input": first_non_empty([item.get("formula_input") for item in items]),
                "herb_ids": ensure_list(item.get("herb_id") for item in items),
                "herb_names": ensure_list(item.get("herb_name") for item in items),
                "target_ids": ensure_list(item.get("target_id") for item in items),
                "target_symbols": ensure_list(item.get("target_symbol") for item in items),
                "target_names": ensure_list(item.get("target_name") for item in items),
                "evidence": ensure_list(item.get("evidence") for item in items if "evidence" in item),
                "row_count": len(items),
            }
        )
    return aggregated


def embed_smiles(smiles: str) -> tuple[Chem.Mol | None, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "MolFromSmiles returned None"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return None, f"EmbedMolecule failed with code {status}"
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass
    mol = Chem.RemoveHs(mol)
    return mol, ""


def build_outputs(
    aggregated: list[dict[str, object]],
    allowed_atoms: set[str],
    skip_unknown_atoms: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    lmdb_records: list[dict[str, object]] = []

    for item in aggregated:
        row = {
            "library": item["library"],
            "source_db": item["source_db"],
            "compound_key": item["compound_key"],
            "source_compound_id": item["source_compound_id"],
            "pubchem_cid": item["pubchem_cid"],
            "chembl_id": item["chembl_id"],
            "compound_names": ";".join(item["compound_names"]),
            "canonical_smiles": "",
            "inchikey": "",
            "inchi": "",
            "formula": "",
            "mol_weight": "",
            "exact_weight": "",
            "atom_count": 0,
            "heavy_atom_count": 0,
            "formal_charge": 0,
            "unknown_atom_count": 0,
            "unknown_atoms": "",
            "herb_count": len(item["herb_names"]),
            "herb_ids": ";".join(item["herb_ids"]),
            "herb_names": ";".join(item["herb_names"]),
            "target_count": len(item["target_symbols"]),
            "target_ids": ";".join(item["target_ids"]),
            "target_symbols": ";".join(item["target_symbols"]),
            "target_names": ";".join(item["target_names"]),
            "evidence": ";".join(item["evidence"]),
            "row_count": item["row_count"],
            "parse_status": "failed",
            "error": "",
        }

        smiles = first_non_empty([item["raw_smiles"]])
        if not smiles:
            row["error"] = "Missing SMILES"
            rows.append(row)
            continue

        mol, err = embed_smiles(smiles)
        if mol is None:
            row["error"] = err
            rows.append(row)
            continue

        try:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            inchi = Chem.MolToInchi(mol)
            inchikey = Chem.MolToInchiKey(mol)
        except Exception as exc:  # pylint: disable=broad-except
            row["error"] = f"identifier_error:{type(exc).__name__}:{exc}"
            rows.append(row)
            continue

        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
        unknown_atoms = sorted({atom for atom in atoms if atom not in allowed_atoms})

        row.update(
            {
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "inchi": inchi,
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "mol_weight": f"{Descriptors.MolWt(mol):.4f}",
                "exact_weight": f"{Descriptors.ExactMolWt(mol):.4f}",
                "atom_count": mol.GetNumAtoms(),
                "heavy_atom_count": mol.GetNumHeavyAtoms(),
                "formal_charge": Chem.GetFormalCharge(mol),
                "unknown_atom_count": len(unknown_atoms),
                "unknown_atoms": ";".join(unknown_atoms),
                "parse_status": "ok",
            }
        )
        rows.append(row)

        if skip_unknown_atoms and unknown_atoms:
            continue

        lmdb_records.append(
            {
                "atoms": atoms,
                "coordinates": [coords],
                "smi": canonical_smiles,
                "library": item["library"],
                "source_db": item["source_db"],
                "compound_key": item["compound_key"],
                "source_compound_id": item["source_compound_id"],
                "pubchem_cid": item["pubchem_cid"],
                "chembl_id": item["chembl_id"],
                "inchikey": inchikey,
                "formula": row["formula"],
                "compound_names": item["compound_names"],
                "herb_ids": item["herb_ids"],
                "herb_names": item["herb_names"],
                "target_ids": item["target_ids"],
                "target_symbols": item["target_symbols"],
                "target_names": item["target_names"],
                "evidence": item["evidence"],
                "row_count": item["row_count"],
            }
        )
    return rows, lmdb_records


def write_table(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_lmdb(records: list[dict[str, object]], path: Path) -> None:
    import lmdb

    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(path),
        subdir=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
        map_size=1099511627776,
    )
    with env.begin(write=True) as txn:
        for idx, record in enumerate(records):
            txn.put(str(idx).encode("ascii"), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
    env.sync()
    env.close()


def main() -> None:
    args = parse_args()
    config = LIBRARY_CONFIGS[args.library]
    input_path = Path(args.input_path).resolve() if args.input_path else config["default_input"].resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (REPO_ROOT / "data" / f"{args.library}_assets").resolve()
    )
    dict_path = Path(args.dict_path).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if not dict_path.exists():
        raise SystemExit(f"Dictionary file not found: {dict_path}")

    allowed_atoms = read_dictionary(dict_path)
    df = load_library_table(args.library, input_path)
    if args.library == "batman":
        df = attach_batman_smiles(df, args)

    aggregated = aggregate_compounds(df, args.library)
    if args.limit > 0:
        aggregated = aggregated[: args.limit]

    rows, lmdb_records = build_outputs(
        aggregated,
        allowed_atoms=allowed_atoms,
        skip_unknown_atoms=args.skip_unknown_atoms,
    )

    table_path = output_dir / f"{args.library}_standardized_compounds.csv"
    lmdb_path = output_dir / f"{args.library}_retrieval.lmdb"
    write_table(rows, table_path)
    if not args.skip_lmdb:
        write_lmdb(lmdb_records, lmdb_path)

    ok_rows = sum(1 for row in rows if row["parse_status"] == "ok")
    print(f"Library: {args.library}")
    print(f"Unique compounds processed: {len(rows)}")
    print(f"Rows with valid structures: {ok_rows}")
    print(f"LMDB records: {len(lmdb_records)}")
    print(f"Standard table: {table_path}")
    if args.skip_lmdb:
        print("LMDB step skipped.")
    else:
        print(f"LMDB: {lmdb_path}")
    print("Reverse lookup keys preserved: compound_key, inchikey, herb_ids, herb_names")


if __name__ == "__main__":
    main()
