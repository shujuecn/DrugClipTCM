#!/usr/bin/env python3
"""
Build DrugCLIP retrieval assets from multiple TCM libraries.

Supported libraries:
- TCMSP (local structure files + metadata tables)
- HIT (tabular library with SMILES)
- HERB 2.0 (tabular library with SMILES)
- BatmanTCM2.0 (optional CID -> SMILES resolution)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from itertools import repeat
from pathlib import Path
from typing import Iterable
from urllib import parse, request

import numpy as np
import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DICT_PATH = REPO_ROOT / "data_dict" / "dict_mol.txt"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"

RDLogger.DisableLog("rdApp.*")

LIBRARY_CONFIGS = {
    "tcmsp": {
        "source_db": "TCMSP",
        "mode": "structure",
        "default_input_dir": DEFAULT_DATA_DIR / "TCMSP",
        "default_output_dir": DEFAULT_DATA_DIR / "tcmsp_assets",
    },
    "hit": {
        "source_db": "HIT",
        "mode": "smiles",
        "default_input_path": DEFAULT_DATA_DIR / "HIT数据库全部.xlsx",
        "default_output_dir": DEFAULT_DATA_DIR / "hit_assets",
    },
    "herb2": {
        "source_db": "HERB2",
        "mode": "smiles",
        "default_input_path": DEFAULT_DATA_DIR / "herb2.0所有中药-成分-靶点映射关系.csv",
        "default_output_dir": DEFAULT_DATA_DIR / "herb2_assets",
    },
    "batman": {
        "source_db": "BatmanTCM2.0",
        "mode": "smiles",
        "default_input_path": DEFAULT_DATA_DIR / "BatmanTCM2.0全部数据.xlsx",
        "default_output_dir": DEFAULT_DATA_DIR / "batman_assets",
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
    "herb_pinyins",
    "target_count",
    "target_ids",
    "target_symbols",
    "target_names",
    "target_uniprots",
    "disease_count",
    "diseases",
    "evidence",
    "row_count",
    "structure_path",
    "parse_status",
    "error",
]

LIST_COLUMNS = [
    "compound_names",
    "herb_ids",
    "herb_names",
    "herb_pinyins",
    "target_ids",
    "target_symbols",
    "target_names",
    "target_uniprots",
    "diseases",
    "evidence",
]


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build standardized CSV tables and retrieval LMDBs for multiple libraries.",
    )
    parser.add_argument(
        "--libraries",
        nargs="+",
        default=["tcmsp", "hit", "herb2"],
        choices=sorted(LIBRARY_CONFIGS),
        help="Libraries to build in one run.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Optional root output directory. Each library is written to <output-root>/<library>/.",
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
        help="Limit compounds per library for smoke tests. 0 means all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes for RDKit standardization. 0 means auto.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=64,
        help="ProcessPoolExecutor chunksize.",
    )
    parser.add_argument(
        "--skip-lmdb",
        action="store_true",
        help="Only export the standardized CSV table.",
    )
    parser.add_argument(
        "--skip-unknown-atoms",
        action="store_true",
        help="Skip LMDB records containing atoms outside dict_mol.txt.",
    )
    parser.add_argument(
        "--log-path",
        default="",
        help="Optional log file path. Defaults to logs/build_retrieval_assets_<timestamp>.log.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--resolve-cids",
        action="store_true",
        help="Resolve PubChem CIDs to SMILES for Batman or other CID-only sources.",
    )
    parser.add_argument(
        "--cid-cache-csv",
        default="",
        help="CSV cache for CID -> SMILES mappings.",
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
    parser.add_argument(
        "--lmdb-batch-size",
        type=int,
        default=5000,
        help="Number of records per LMDB write transaction.",
    )
    return parser.parse_args()


def configure_logging(args: argparse.Namespace) -> tuple[logging.Logger, Path]:
    if args.log_path:
        log_path = Path(args.log_path).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = (DEFAULT_LOG_DIR / f"build_retrieval_assets_{stamp}.log").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("build_retrieval_assets")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger, log_path


def read_dictionary(path: Path) -> set[str]:
    allowed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token and not token.startswith("["):
                allowed.add(token)
    return allowed


def clean_text_expr(column: str) -> pl.Expr:
    base = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()
    return (
        pl.when(base.is_null() | (base == "") | (base.str.to_lowercase() == "null"))
        .then(None)
        .otherwise(base)
        .alias(column)
    )


def normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    if not df.columns:
        return df
    return df.with_columns(clean_text_expr(column) for column in df.columns)


def list_unique_sorted(column: str) -> pl.Expr:
    return pl.col(column).drop_nulls().unique().sort().alias(column)


def ensure_list(values: Iterable[str | None]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "null":
            out.append(text)
    return sorted(set(out))


def to_output_dir(library: str, output_root: str) -> Path:
    if output_root:
        return (Path(output_root).resolve() / library)
    return Path(LIBRARY_CONFIGS[library]["default_output_dir"]).resolve()


def load_structure_files(input_dir: Path, limit: int) -> list[Path]:
    files = sorted(list(input_dir.rglob("*.mol2")) + list(input_dir.rglob("*.sdf")))
    if limit > 0:
        return files[:limit]
    return files


def load_mol(path: Path) -> tuple[Chem.Mol | None, str]:
    if path.suffix.lower() == ".mol2":
        mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
        if mol is not None:
            return mol, ""
        mol = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
        if mol is None:
            return None, "MolFromMol2File returned None"
    elif path.suffix.lower() == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        mol = supplier[0] if len(supplier) else None
        if mol is not None:
            return mol, ""
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        mol = supplier[0] if len(supplier) else None
        if mol is None:
            return None, "SDMolSupplier returned None"
    else:
        return None, f"Unsupported format: {path.suffix}"

    try:
        Chem.SanitizeMol(mol)
        return mol, "sanitized_from_raw_structure"
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"{type(exc).__name__}:{exc}"


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


def get_coordinates(mol: Chem.Mol) -> np.ndarray | None:
    if mol.GetNumConformers() == 0:
        return None
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    return coords


def join_list(values: list[str] | None) -> str:
    return ";".join(values or [])


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
    for start in range(0, len(values), size):
        yield values[start : start + size]


def resolve_pubchem_cids(
    cids: list[str],
    batch_size: int,
    timeout: int,
    retries: int,
    cache: dict[str, str],
    logger: logging.Logger,
) -> dict[str, str]:
    unresolved = [cid for cid in cids if cid not in cache]
    if not unresolved:
        return cache

    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/CanonicalSMILES/JSON"
    for batch in tqdm(chunked(unresolved, batch_size), total=(len(unresolved) + batch_size - 1) // batch_size, desc="Resolve CID"):
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
                logger.warning("Retrying CID batch after error: %s", last_error)
                time.sleep(1.5 * (attempt + 1))
    return cache


def load_tcmsp_records(input_dir: Path, limit: int, logger: logging.Logger) -> list[dict[str, object]]:
    structure_files = load_structure_files(input_dir, limit=limit)
    logger.info("TCMSP structure files discovered: %s", len(structure_files))
    structures_df = pl.DataFrame(
        {
            "source_compound_id": [path.stem for path in structure_files],
            "structure_path": [str(path) for path in structure_files],
        }
    )

    compound_table = normalize_frame(
        pl.read_excel(input_dir / "2. 13729个化合物筛选数据.xlsx").select(
            [
                pl.col("MOL_ID").cast(pl.Utf8).alias("source_compound_id"),
                pl.col("molecule_name").cast(pl.Utf8).alias("compound_name"),
                pl.col("molecule_ID").cast(pl.Utf8).alias("pubchem_cid"),
            ]
        )
    ).unique(subset=["source_compound_id"], keep="first")

    relation_table = normalize_frame(
        pl.read_excel(input_dir / "3. 中药-化合物-靶点-疾病.xlsx").select(
            [
                pl.col("MOL_ID").cast(pl.Utf8).alias("source_compound_id"),
                pl.col("Chinese name").cast(pl.Utf8).alias("herb_name"),
                pl.col("herb_pinyin").cast(pl.Utf8).alias("herb_pinyin"),
                pl.col("molecule_name").cast(pl.Utf8).alias("compound_name"),
                pl.col("target_ID").cast(pl.Utf8).alias("target_id"),
                pl.col("target_name").cast(pl.Utf8).alias("target_name"),
                pl.col("drugbank_ID").cast(pl.Utf8).alias("target_uniprot"),
                pl.col("validated").cast(pl.Utf8).alias("evidence"),
                pl.col("disease").cast(pl.Utf8).alias("disease"),
            ]
        )
    )

    relation_agg = relation_table.group_by("source_compound_id").agg(
        [
            list_unique_sorted("compound_name").alias("relation_compound_names"),
            list_unique_sorted("herb_name").alias("herb_names"),
            list_unique_sorted("herb_pinyin").alias("herb_pinyins"),
            list_unique_sorted("target_id").alias("target_ids"),
            list_unique_sorted("target_name").alias("target_names"),
            list_unique_sorted("target_uniprot").alias("target_uniprots"),
            list_unique_sorted("evidence").alias("evidence"),
            list_unique_sorted("disease").alias("diseases"),
            pl.len().alias("row_count"),
        ]
    )

    merged = structures_df.join(compound_table, on="source_compound_id", how="left").join(
        relation_agg,
        on="source_compound_id",
        how="left",
    )

    records: list[dict[str, object]] = []
    for row in merged.iter_rows(named=True):
        records.append(
            {
                "library": "tcmsp",
                "source_db": "TCMSP",
                "compound_key": row["source_compound_id"],
                "source_compound_id": row["source_compound_id"],
                "pubchem_cid": row.get("pubchem_cid"),
                "chembl_id": None,
                "compound_names": ensure_list(
                    ((row.get("compound_name") and [row["compound_name"]]) or [])
                    + (row.get("relation_compound_names") or [])
                ),
                "raw_smiles": None,
                "formula_input": None,
                "herb_ids": [],
                "herb_names": ensure_list(row.get("herb_names") or []),
                "herb_pinyins": ensure_list(row.get("herb_pinyins") or []),
                "target_ids": ensure_list(row.get("target_ids") or []),
                "target_symbols": [],
                "target_names": ensure_list(row.get("target_names") or []),
                "target_uniprots": ensure_list(row.get("target_uniprots") or []),
                "diseases": ensure_list(row.get("diseases") or []),
                "evidence": ensure_list(row.get("evidence") or []),
                "row_count": row.get("row_count") or 1,
                "structure_path": row["structure_path"],
            }
        )
    return records


def load_library_table(library: str, logger: logging.Logger) -> pl.DataFrame:
    input_path = Path(LIBRARY_CONFIGS[library]["default_input_path"]).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    logger.info("Loading %s from %s", library, input_path)

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
                pl.lit(None, dtype=pl.Utf8).alias("herb_pinyin"),
                pl.col("Target ID").cast(pl.Utf8).alias("target_id"),
                pl.col("Symbol").cast(pl.Utf8).alias("target_symbol"),
                pl.col("Protein Name").cast(pl.Utf8).alias("target_name"),
                pl.col("Uniprot").cast(pl.Utf8).alias("target_uniprot"),
                pl.lit(None, dtype=pl.Utf8).alias("disease"),
                pl.lit(None, dtype=pl.Utf8).alias("evidence"),
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
                pl.lit(None, dtype=pl.Utf8).alias("herb_pinyin"),
                pl.col("Target id").cast(pl.Utf8).alias("target_id"),
                pl.col("Gene symbol").cast(pl.Utf8).alias("target_symbol"),
                pl.col("Protein name").cast(pl.Utf8).alias("target_name"),
                pl.lit(None, dtype=pl.Utf8).alias("target_uniprot"),
                pl.lit(None, dtype=pl.Utf8).alias("disease"),
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
                pl.lit(None, dtype=pl.Utf8).alias("herb_pinyin"),
                pl.col("Gene ID").cast(pl.Utf8).alias("target_id"),
                pl.col("Gene Name").cast(pl.Utf8).alias("target_symbol"),
                pl.col("Gene Name").cast(pl.Utf8).alias("target_name"),
                pl.lit(None, dtype=pl.Utf8).alias("target_uniprot"),
                pl.lit(None, dtype=pl.Utf8).alias("disease"),
                pl.col("Score").cast(pl.Utf8).alias("evidence"),
            ]
        )
    else:
        raise ValueError(f"Unsupported library: {library}")
    return normalize_frame(df)


def attach_batman_smiles(
    df: pl.DataFrame,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> pl.DataFrame:
    cache_path = (
        Path(args.cid_cache_csv).resolve()
        if args.cid_cache_csv
        else (REPO_ROOT / "data" / "batman_cid_smiles_cache.csv").resolve()
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
            logger=logger,
        )
        save_cid_cache(cache_path, cache)
    if not cache:
        raise SystemExit(
            "BatmanTCM2.0 needs CID -> SMILES mappings. "
            "Provide --resolve-cids (network required) or a populated --cid-cache-csv."
        )
    cache_df = pl.DataFrame({"pubchem_cid": list(cache.keys()), "raw_smiles": list(cache.values())})
    resolved = df.drop("raw_smiles").join(cache_df, on="pubchem_cid", how="left")
    if int(resolved.select(pl.col("raw_smiles").is_not_null().sum()).item()) == 0:
        raise SystemExit("No Batman CIDs could be resolved to SMILES.")
    return normalize_frame(resolved)


def aggregate_tabular_library(df: pl.DataFrame, library: str, limit: int, logger: logging.Logger) -> list[dict[str, object]]:
    grouped = (
        df.with_columns(
            pl.coalesce(
                [
                    pl.col("source_compound_id"),
                    pl.col("pubchem_cid"),
                    pl.col("raw_smiles"),
                ]
            ).alias("compound_key")
        )
        .filter(pl.col("compound_key").is_not_null())
        .group_by("compound_key")
        .agg(
            [
                pl.first("source_db").alias("source_db"),
                pl.first("source_compound_id").alias("source_compound_id"),
                pl.first("pubchem_cid").alias("pubchem_cid"),
                pl.first("chembl_id").alias("chembl_id"),
                pl.first("raw_smiles").alias("raw_smiles"),
                pl.first("formula_input").alias("formula_input"),
                list_unique_sorted("compound_name"),
                list_unique_sorted("herb_id"),
                list_unique_sorted("herb_name"),
                list_unique_sorted("herb_pinyin"),
                list_unique_sorted("target_id"),
                list_unique_sorted("target_symbol"),
                list_unique_sorted("target_name"),
                list_unique_sorted("target_uniprot"),
                list_unique_sorted("disease"),
                list_unique_sorted("evidence"),
                pl.len().alias("row_count"),
            ]
        )
        .sort("compound_key")
    )

    if limit > 0:
        grouped = grouped.head(limit)
    logger.info("%s unique compounds after aggregation: %s", library, grouped.height)

    records: list[dict[str, object]] = []
    for row in grouped.iter_rows(named=True):
        records.append(
            {
                "library": library,
                "source_db": row["source_db"],
                "compound_key": row["compound_key"],
                "source_compound_id": row.get("source_compound_id"),
                "pubchem_cid": row.get("pubchem_cid"),
                "chembl_id": row.get("chembl_id"),
                "compound_names": ensure_list(row.get("compound_name") or []),
                "raw_smiles": row.get("raw_smiles"),
                "formula_input": row.get("formula_input"),
                "herb_ids": ensure_list(row.get("herb_id") or []),
                "herb_names": ensure_list(row.get("herb_name") or []),
                "herb_pinyins": ensure_list(row.get("herb_pinyin") or []),
                "target_ids": ensure_list(row.get("target_id") or []),
                "target_symbols": ensure_list(row.get("target_symbol") or []),
                "target_names": ensure_list(row.get("target_name") or []),
                "target_uniprots": ensure_list(row.get("target_uniprot") or []),
                "diseases": ensure_list(row.get("disease") or []),
                "evidence": ensure_list(row.get("evidence") or []),
                "row_count": row["row_count"],
                "structure_path": None,
            }
        )
    return records


def build_base_row(record: dict[str, object]) -> dict[str, object]:
    return {
        "library": record["library"],
        "source_db": record["source_db"],
        "compound_key": record["compound_key"],
        "source_compound_id": record.get("source_compound_id") or "",
        "pubchem_cid": record.get("pubchem_cid") or "",
        "chembl_id": record.get("chembl_id") or "",
        "compound_names": join_list(record.get("compound_names")),
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
        "herb_count": len(record.get("herb_names") or []),
        "herb_ids": join_list(record.get("herb_ids")),
        "herb_names": join_list(record.get("herb_names")),
        "herb_pinyins": join_list(record.get("herb_pinyins")),
        "target_count": len(record.get("target_names") or []),
        "target_ids": join_list(record.get("target_ids")),
        "target_symbols": join_list(record.get("target_symbols")),
        "target_names": join_list(record.get("target_names")),
        "target_uniprots": join_list(record.get("target_uniprots")),
        "disease_count": len(record.get("diseases") or []),
        "diseases": join_list(record.get("diseases")),
        "evidence": join_list(record.get("evidence")),
        "row_count": record.get("row_count") or 0,
        "structure_path": record.get("structure_path") or "",
        "parse_status": "failed",
        "error": "",
    }


def standardize_record(record: dict[str, object], allowed_atoms: set[str], skip_unknown_atoms: bool) -> tuple[dict[str, object], dict[str, object] | None]:
    row = build_base_row(record)

    structure_path = record.get("structure_path")
    if structure_path:
        mol, load_note = load_mol(Path(str(structure_path)))
        if mol is None:
            row["error"] = load_note
            return row, None
        coords = get_coordinates(mol)
        if coords is None:
            row["parse_status"] = "failed:no_3d"
            row["error"] = "No conformer found in structure file"
            return row, None
        parse_note = load_note
    else:
        smiles = str(record.get("raw_smiles") or "").strip()
        if not smiles:
            row["error"] = "Missing SMILES"
            return row, None
        mol, err = embed_smiles(smiles)
        if mol is None:
            row["error"] = err
            return row, None
        coords = get_coordinates(mol)
        parse_note = ""

    try:
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.MolToInchiKey(mol)
    except Exception as exc:  # pylint: disable=broad-except
        row["error"] = f"identifier_error:{type(exc).__name__}:{exc}"
        return row, None

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
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
            "parse_status": "ok" if not parse_note else f"ok:{parse_note}",
        }
    )

    if skip_unknown_atoms and unknown_atoms:
        return row, None

    lmdb_record = {
        "atoms": atoms,
        "coordinates": [coords],
        "smi": canonical_smiles,
        "library": record["library"],
        "source_db": record["source_db"],
        "compound_key": record["compound_key"],
        "source_compound_id": record.get("source_compound_id"),
        "pubchem_cid": record.get("pubchem_cid"),
        "chembl_id": record.get("chembl_id"),
        "inchikey": inchikey,
        "formula": row["formula"],
        "compound_names": record.get("compound_names") or [],
        "herb_ids": record.get("herb_ids") or [],
        "herb_names": record.get("herb_names") or [],
        "herb_pinyins": record.get("herb_pinyins") or [],
        "target_ids": record.get("target_ids") or [],
        "target_symbols": record.get("target_symbols") or [],
        "target_names": record.get("target_names") or [],
        "target_uniprots": record.get("target_uniprots") or [],
        "diseases": record.get("diseases") or [],
        "evidence": record.get("evidence") or [],
        "row_count": record.get("row_count") or 0,
        "structure_path": record.get("structure_path") or "",
    }
    return row, lmdb_record


def process_records(
    library: str,
    records: list[dict[str, object]],
    allowed_atoms: set[str],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter]:
    rows: list[dict[str, object]] = []
    lmdb_records: list[dict[str, object]] = []
    status_counter: Counter = Counter()

    workers = args.workers if args.workers > 0 else max(1, min(8, (os.cpu_count() or 1) - 1))
    logger.info("%s standardization workers: %s", library, workers)

    if workers == 1:
        iterator = (
            standardize_record(record, allowed_atoms=allowed_atoms, skip_unknown_atoms=args.skip_unknown_atoms)
            for record in records
        )
    else:
        iterator = None
        executor_class = ProcessPoolExecutor
        try:
            executor = executor_class(max_workers=workers)
        except (OSError, PermissionError) as exc:
            logger.warning(
                "%s process pool unavailable (%s). Falling back to thread pool.",
                library,
                exc,
            )
            executor_class = ThreadPoolExecutor
            executor = executor_class(max_workers=workers)

        with executor:
            map_kwargs = {"chunksize": args.chunksize} if executor_class is ProcessPoolExecutor else {}
            iterator = executor.map(
                standardize_record,
                records,
                repeat(allowed_atoms),
                repeat(args.skip_unknown_atoms),
                **map_kwargs,
            )
            for row, lmdb_record in tqdm(iterator, total=len(records), desc=f"{library} standardize"):
                rows.append(row)
                status_counter[row["parse_status"]] += 1
                if lmdb_record is not None:
                    lmdb_records.append(lmdb_record)
            return rows, lmdb_records, status_counter

    for row, lmdb_record in tqdm(iterator, total=len(records), desc=f"{library} standardize"):
        rows.append(row)
        status_counter[row["parse_status"]] += 1
        if lmdb_record is not None:
            lmdb_records.append(lmdb_record)
    return rows, lmdb_records, status_counter


def write_table(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_lmdb(records: list[dict[str, object]], path: Path, batch_size: int, desc: str) -> None:
    import lmdb

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    env = lmdb.open(
        str(path),
        subdir=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
        map_size=1099511627776,
    )
    txn = env.begin(write=True)
    try:
        for idx, record in enumerate(tqdm(records, desc=desc), start=1):
            txn.put(str(idx - 1).encode("ascii"), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
            if idx % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        env.sync()
    finally:
        env.close()


def build_library_assets(
    library: str,
    allowed_atoms: set[str],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    output_dir = to_output_dir(library, args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("========== Build %s ==========", library)
    t0 = time.perf_counter()

    if library == "tcmsp":
        input_dir = Path(LIBRARY_CONFIGS[library]["default_input_dir"]).resolve()
        if not input_dir.exists():
            raise SystemExit(f"Input directory not found: {input_dir}")
        records = load_tcmsp_records(input_dir, limit=args.limit, logger=logger)
    else:
        df = load_library_table(library, logger)
        if library == "batman":
            df = attach_batman_smiles(df, args, logger)
        records = aggregate_tabular_library(df, library=library, limit=args.limit, logger=logger)

    rows, lmdb_records, status_counter = process_records(
        library=library,
        records=records,
        allowed_atoms=allowed_atoms,
        args=args,
        logger=logger,
    )

    table_path = output_dir / f"{library}_standardized_compounds.csv"
    lmdb_path = output_dir / f"{library}_retrieval.lmdb"
    write_table(rows, table_path)
    logger.info("%s standardized table written: %s", library, table_path)
    if not args.skip_lmdb:
        write_lmdb(
            lmdb_records,
            lmdb_path,
            batch_size=args.lmdb_batch_size,
            desc=f"{library} lmdb",
        )
        logger.info("%s lmdb written: %s", library, lmdb_path)

    ok_rows = sum(1 for row in rows if str(row["parse_status"]).startswith("ok"))
    unknown_atom_rows = sum(1 for row in rows if int(row["unknown_atom_count"]) > 0)
    logger.info(
        "%s finished in %.2fs | processed=%s ok=%s lmdb=%s unknown_atom_rows=%s statuses=%s",
        library,
        time.perf_counter() - t0,
        len(rows),
        ok_rows,
        len(lmdb_records),
        unknown_atom_rows,
        dict(status_counter),
    )
    logger.info(
        "%s reverse lookup keys: compound_key, inchikey, herb_ids, herb_names, target_ids, target_symbols, diseases",
        library,
    )


def main() -> None:
    args = parse_args()
    logger, log_path = configure_logging(args)
    dict_path = Path(args.dict_path).resolve()
    if not dict_path.exists():
        raise SystemExit(f"Dictionary file not found: {dict_path}")
    allowed_atoms = read_dictionary(dict_path)

    logger.info("Log file: %s", log_path)
    logger.info("Libraries requested: %s", ", ".join(args.libraries))
    logger.info("Dictionary atoms loaded: %s", len(allowed_atoms))

    for library in args.libraries:
        build_library_assets(library, allowed_atoms=allowed_atoms, args=args, logger=logger)


if __name__ == "__main__":
    main()
