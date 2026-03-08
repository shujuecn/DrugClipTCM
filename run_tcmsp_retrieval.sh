#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}/Uni-Core"

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data_dict}"
MOL_PATH="${MOL_PATH:-${ROOT_DIR}/results/tcmsp_assets/tcmsp_retrieval.lmdb}"
POCKET_PATH="${POCKET_PATH:-}"
WEIGHT_PATH="${WEIGHT_PATH:-}"
EMB_DIR="${EMB_DIR:-${ROOT_DIR}/results/tcmsp_retrieval}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${MOL_PATH}" ]]; then
    echo "LMDB not found: ${MOL_PATH}" >&2
    echo "Build it first with scripts/build_tcmsp_retrieval_assets.py" >&2
    exit 1
fi

if [[ -z "${POCKET_PATH}" || ! -f "${POCKET_PATH}" ]]; then
    echo "Set POCKET_PATH to an existing pocket.lmdb file." >&2
    exit 1
fi

if [[ -z "${WEIGHT_PATH}" || ! -f "${WEIGHT_PATH}" ]]; then
    echo "Set WEIGHT_PATH to an existing DrugCLIP checkpoint." >&2
    exit 1
fi

mkdir -p "${EMB_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/unimol/retrieval.py" \
    --user-dir "${ROOT_DIR}/unimol" \
    "${DATA_PATH}" \
    --valid-subset test \
    --results-path "${EMB_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    --batch-size "${BATCH_SIZE}" \
    --task drugclip \
    --arch drugclip \
    --fp16 \
    --path "${WEIGHT_PATH}" \
    --mol-path "${MOL_PATH}" \
    --pocket-path "${POCKET_PATH}" \
    --emb-dir "${EMB_DIR}"

echo "Retrieval finished. Ranked compounds: ${EMB_DIR}/ranked_compounds.txt"
