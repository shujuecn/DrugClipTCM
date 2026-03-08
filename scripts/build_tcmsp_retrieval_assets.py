#!/usr/bin/env python3
"""
Build a TCMSP pilot retrieval dataset for DrugCLIP.

Pipeline:
1. Scan local TCMSP structure files (mol2/sdf).
2. Export a standardized compound table (CSV).
3. Convert valid compounds into the LMDB format expected by unimol/retrieval.py.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "TCMSP"
DEFAULT_DICT_PATH = REPO_ROOT / "data_dict" / "dict_mol.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "tcmsp_assets"

RDLogger.DisableLog("rdApp.*")

STANDARD_TABLE_FIELDS = [
    "mol_id",
    "source_db",
    "file_path",
    "parse_status",
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
    "has_3d",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standardized TCMSP table and DrugCLIP retrieval LMDB.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing TCMSP mol2/sdf structure files.",
    )
    parser.add_argument(
        "--dict-path",
        default=str(DEFAULT_DICT_PATH),
        help="DrugCLIP molecule dictionary used to track unknown atom types.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated CSV/LMDB outputs.",
    )
    parser.add_argument(
        "--table-name",
        default="tcmsp_standardized_compounds.csv",
        help="Standardized CSV filename inside output-dir.",
    )
    parser.add_argument(
        "--lmdb-name",
        default="tcmsp_retrieval.lmdb",
        help="LMDB filename inside output-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N molecules. 0 means all files.",
    )
    parser.add_argument(
        "--skip-lmdb",
        action="store_true",
        help="Only build the standardized CSV table.",
    )
    parser.add_argument(
        "--skip-unknown-atoms",
        action="store_true",
        help="Skip molecules containing atom symbols outside data_dict/dict_mol.txt.",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Keep failed rows in the CSV table instead of dropping them.",
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


def iter_structure_files(input_dir: Path, limit: int) -> list[Path]:
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


def get_atom_symbols(mol: Chem.Mol) -> list[str]:
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def get_coordinates(mol: Chem.Mol) -> np.ndarray | None:
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    coords = np.asarray(conf.GetPositions(), dtype=np.float32)
    return coords


def build_row(path: Path, allowed_atoms: set[str]) -> tuple[dict[str, object], dict[str, object] | None]:
    mol_id = path.stem
    base_row: dict[str, object] = {
        "mol_id": mol_id,
        "source_db": "TCMSP",
        "file_path": str(path),
        "parse_status": "failed",
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
        "has_3d": 0,
        "error": "",
    }

    mol, load_note = load_mol(path)
    if mol is None:
        base_row["error"] = load_note
        return base_row, None

    try:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.MolToInchiKey(mol)
    except Exception as exc:  # pylint: disable=broad-except
        base_row["error"] = f"identifier_error:{type(exc).__name__}:{exc}"
        return base_row, None

    atoms = get_atom_symbols(mol)
    coords = get_coordinates(mol)
    unknown_atoms = sorted({atom for atom in atoms if atom not in allowed_atoms})

    base_row.update(
        {
            "parse_status": "ok" if not load_note else f"ok:{load_note}",
            "canonical_smiles": smiles,
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
            "has_3d": int(coords is not None),
        }
    )

    if coords is None:
        base_row["parse_status"] = "failed:no_3d"
        base_row["error"] = "No conformer found in structure file"
        return base_row, None

    lmdb_record = {
        "atoms": atoms,
        "coordinates": [coords],
        "smi": smiles,
        "mol_id": mol_id,
        "source_db": "TCMSP",
        "inchikey": inchikey,
        "formula": base_row["formula"],
        "file_path": str(path),
    }
    return base_row, lmdb_record


def should_keep_row(row: dict[str, object], keep_failed: bool) -> bool:
    if keep_failed:
        return True
    return str(row["parse_status"]).startswith("ok")


def write_csv(rows: Iterable[dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_TABLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_lmdb(records: list[dict[str, object]], lmdb_path: Path) -> None:
    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "python-lmdb is required to write retrieval LMDB files. "
            "Install it in the active environment and rerun without --skip-lmdb."
        ) from exc

    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(lmdb_path),
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


def summarize(rows: list[dict[str, object]], records: list[dict[str, object]]) -> str:
    status_counts = Counter(str(row["parse_status"]).split(":")[0] for row in rows)
    unknown_count = sum(1 for row in rows if int(row["unknown_atom_count"]) > 0)
    lines = [
        f"Processed molecules: {len(rows)}",
        f"Rows written to table: {len(rows)}",
        f"LMDB records: {len(records)}",
        f"Parse summary: {dict(status_counts)}",
        f"Molecules with unknown atom types: {unknown_count}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    dict_path = Path(args.dict_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    csv_path = output_dir / args.table_name
    lmdb_path = output_dir / args.lmdb_name

    if not input_dir.exists():
        raise SystemExit(f"input directory not found: {input_dir}")
    if not dict_path.exists():
        raise SystemExit(f"dictionary file not found: {dict_path}")

    allowed_atoms = read_dictionary(dict_path)
    files = iter_structure_files(input_dir, args.limit)
    if not files:
        raise SystemExit(f"No mol2/sdf files found in {input_dir}")

    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for path in tqdm(files, desc="Building TCMSP assets"):
        row, record = build_row(path, allowed_atoms)
        if should_keep_row(row, args.keep_failed):
            rows.append(row)
        if record is None:
            continue
        if args.skip_unknown_atoms and int(row["unknown_atom_count"]) > 0:
            continue
        records.append(record)

    write_csv(rows, csv_path)
    if not args.skip_lmdb:
        write_lmdb(records, lmdb_path)

    print(summarize(rows, records))
    print(f"Standard table: {csv_path}")
    if args.skip_lmdb:
        print("LMDB step skipped.")
    else:
        print(f"LMDB: {lmdb_path}")


if __name__ == "__main__":
    main()
