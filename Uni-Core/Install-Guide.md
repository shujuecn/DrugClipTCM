# README — Uni-Core (unicore) 标准安装流程（含 CUDA fused extensions）

> 适用环境：Linux / WSL2 + NVIDIA GPU
> 目标：安装 `unicore` 并**成功编译** `unicore_fused_*` CUDA 扩展（LayerNorm / RMSNorm / SoftmaxDropout / MultiTensor 等）

---

## 1. 前置要求

### 1.1 硬件 & 驱动

* NVIDIA GPU（建议 Compute Capability ≥ 7.0）
* `nvidia-smi` 可正常工作

### 1.2 软件

* Python ≥ 3.7（建议 3.10）
* PyTorch **CUDA 版本** 与本机 CUDA Toolkit **主次版本一致**（例如都为 11.8）
* CUDA Toolkit（必须包含 `nvcc`）
* C++ 编译器 `g++`（能编译 C++17）
* `ninja`（强烈建议）

---

## 2. 创建/进入 Python 环境

以 conda 为例：

```bash
conda create -n drugclip python=3.10 -y
conda activate drugclip
```

安装 PyTorch（示例：CUDA 11.8，对应你当前环境）：

> 你实际用什么渠道装 torch 都行，关键是 `torch.__version__` 里带 `+cu118`，且 `torch.cuda.is_available()` 为 True

---

## 3. 安装构建依赖

```bash
pip install -U pip setuptools wheel ninja
```

---

## 4. 确认 CUDA 工具链可用

```bash
which nvcc || echo "no nvcc"
nvcc --version
python -c "import torch; print('cuda?', torch.cuda.is_available()); print('torch cuda', torch.version.cuda)"
```

---

## 5. 设置关键环境变量

> 下面以 CUDA 11.8 为例；如果你是别的版本，把路径改掉即可。

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# 按你的 GPU 架构设置（示例：Ada = 8.9）
export TORCH_CUDA_ARCH_LIST="8.9"

# 并行编译线程数（按 CPU 核数调整）
export MAX_JOBS=8

# 让编译日志更详细（可选）
export VERBOSE=1
```

验证 PyTorch 能识别 CUDA_HOME（可选）：

```bash
python - <<'PY'
import os, torch
import torch.utils.cpp_extension as ce
print("env CUDA_HOME =", os.environ.get("CUDA_HOME"))
print("ce.CUDA_HOME  =", ce.CUDA_HOME)
print("torch cuda avail =", torch.cuda.is_available(), "torch.version.cuda =", torch.version.cuda)
if torch.cuda.is_available():
    print("capability =", torch.cuda.get_device_capability())
PY
```

---

## 6. 编译并启用 CUDA 扩展（关键步骤）

> **重要：本项目默认禁用 CUDA 扩展**。必须显式加 `--enable-cuda-ext` 才会生成 `unicore_fused_*` 的 `.so`。

进入项目根目录（有 `setup.py` 的目录）后执行：

```bash
# 清理
rm -rf build dist *.egg-info

# 关键：编译 CUDA 扩展（inplace 会把 .so 放到当前源码目录）
python setup.py build_ext --inplace -v --enable-cuda-ext
```

---

## 7. 检查是否产出 fused .so

```bash
find . -maxdepth 2 -name "unicore_fused_*.so" -print
```

你应该能看到类似：

* `unicore_fused_layernorm*.so`
* `unicore_fused_rmsnorm*.so`
* `unicore_fused_softmax_dropout*.so`
* `unicore_fused_multi_tensor*.so`
* `unicore_fused_rounding*.so`
* `unicore_fused_adam*.so`
* 等等

---

## 8. 验证导入是否正常

### 8.1 导入 unicore

```bash
python -c "import unicore; print('unicore import ok')"
```

### 8.2 检查 fused 扩展是否可被 Python 找到

```bash
python - <<'PY'
import importlib.util as u
mods = [
  "unicore_fused_multi_tensor",
  "unicore_fused_rounding",
  "unicore_fused_layernorm",
  "unicore_fused_layernorm_backward_gamma_beta",
  "unicore_fused_rmsnorm",
  "unicore_fused_rmsnorm_backward_gamma",
  "unicore_fused_softmax_dropout",
]
for m in mods:
    s = u.find_spec(m)
    print(f"{m:45s}", "OK -> " + str(s.origin) if s else "MISSING")
PY
```

### 8.3 快速 sanity check

```bash
python -c "import unicore, torch; print('unicore ok, torch', torch.__version__)"
```

---

## 9. 常见问题速查

### Q1：为什么之前一直提示 `fused_xxx is not installed corrected`？

A：因为 `setup.py` 默认 `DISABLE_CUDA_EXTENSION=True`，不加 `--enable-cuda-ext` 就不会编译任何扩展，因此相关模块永远 `MISSING`。

### Q2：`zsh: no matches found: *.egg-info`

A：这是 zsh 的 glob 行为；如果目录不存在会报错。属于**无害提示**。
需要的话可以改成更稳的清理方式：

```bash
rm -rf build dist ./*.egg-info
```

### Q3：`There are no g++ version bounds defined for CUDA version 11.8`

A：PyTorch 的提醒，不一定是错误。只要编译能继续并生成 `.so`，通常无需处理。

---

## 10. 一键最小可复现安装（推荐复制）

```bash
pip install -U pip setuptools wheel ninja

rm -rf build dist *.egg-info

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8
export VERBOSE=1

python setup.py build_ext --inplace -v --enable-cuda-ext

find . -maxdepth 2 -name "unicore_fused_*.so" -print
python -c "import unicore, torch; print('unicore ok, torch', torch.__version__)"
```

---

如果你希望这个 README 更“工程化”（加上 conda/pip 的 torch 安装矩阵、CUDA 版本兼容表、以及把 `--enable-cuda-ext` 写成 Makefile/脚本），我也可以给你补一版更正式的。
