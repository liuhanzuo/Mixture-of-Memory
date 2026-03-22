#!/usr/bin/env bash
# ============================================================
# setup_env.sh — MoM (Mixture-of-Memory) 环境一键配置脚本
# 
# 使用方法:
#   bash scripts/setup_env.sh
#
# 前置条件: 机器上有 conda 和 Python 3.10+
# ============================================================
set -euo pipefail

# ---- 颜色输出 ---- #
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ---- 配置 ---- #
ENV_NAME="mom"
PYTHON_VERSION="3.11"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="/apdcephfs/pig_data/Adaptive-Sparse-Trainer/models/Qwen--Qwen3-1.7b"

echo ""
echo "============================================================"
echo "  🧠 Mixture-of-Memory (MoM) 环境配置"
echo "============================================================"
echo "  项目路径: ${PROJECT_ROOT}"
echo "  模型路径: ${MODEL_PATH}"
echo "  Python:  ${PYTHON_VERSION}"
echo "  Conda 环境名: ${ENV_NAME}"
echo "============================================================"
echo ""

# ============================================================
# Step 0: 检测基础环境
# ============================================================
info "Step 0: 检测基础环境..."

# 检测 conda
if command -v conda &> /dev/null; then
    ok "conda 已安装: $(conda --version)"
else
    # 如果没有 conda 但有 python3，也可以用 venv
    if command -v python3 &> /dev/null; then
        warn "conda 未找到，将使用 python3 venv 替代"
        USE_VENV=true
    else
        fail "conda 和 python3 都未找到，请先安装其中之一"
        exit 1
    fi
fi
USE_VENV=${USE_VENV:-false}

# ============================================================
# Step 1: 创建 Python 环境
# ============================================================
info "Step 1: 创建 Python 环境..."

if [ "$USE_VENV" = true ]; then
    # 使用 venv
    VENV_PATH="${PROJECT_ROOT}/.venv"
    if [ -d "$VENV_PATH" ]; then
        ok "venv 已存在: ${VENV_PATH}"
    else
        info "创建 venv: ${VENV_PATH}"
        python3 -m venv "${VENV_PATH}"
        ok "venv 创建完成"
    fi
    source "${VENV_PATH}/bin/activate"
    PYTHON_BIN="${VENV_PATH}/bin/python"
    PIP_BIN="${VENV_PATH}/bin/pip"
else
    # 使用 conda
    # 初始化 conda (防止 conda activate 不可用)
    eval "$(conda shell.bash hook 2>/dev/null)" || true

    if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
        ok "conda 环境 '${ENV_NAME}' 已存在"
        conda activate "${ENV_NAME}" 2>/dev/null || source activate "${ENV_NAME}" 2>/dev/null || true
    else
        info "创建 conda 环境: ${ENV_NAME} (python=${PYTHON_VERSION})"
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
        conda activate "${ENV_NAME}" 2>/dev/null || source activate "${ENV_NAME}" 2>/dev/null || true
        ok "conda 环境创建完成"
    fi
    PYTHON_BIN="python"
    PIP_BIN="pip"
fi

info "Python 路径: $(which ${PYTHON_BIN})"
info "Python 版本: $(${PYTHON_BIN} --version)"

# ============================================================
# Step 2: 安装 PyTorch (CUDA 12.x)
# ============================================================
info "Step 2: 安装/检查 PyTorch..."

TORCH_OK=$(${PYTHON_BIN} -c "
try:
    import torch
    if torch.cuda.is_available():
        print('cuda_ok')
    else:
        print('no_cuda')
except ImportError:
    print('not_installed')
" 2>&1)

if [ "$TORCH_OK" = "cuda_ok" ]; then
    TORCH_VER=$(${PYTHON_BIN} -c "import torch; print(torch.__version__)")
    ok "PyTorch 已安装且 CUDA 可用: torch=${TORCH_VER}"
elif [ "$TORCH_OK" = "no_cuda" ]; then
    TORCH_VER=$(${PYTHON_BIN} -c "import torch; print(torch.__version__)")
    warn "PyTorch 已安装 (${TORCH_VER}) 但 CUDA 不可用"
    info "尝试安装带 CUDA 支持的 PyTorch..."
    ${PIP_BIN} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ok "PyTorch with CUDA 安装完成"
else
    info "PyTorch 未安装，正在安装带 CUDA 12.8 支持的版本..."
    ${PIP_BIN} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ok "PyTorch 安装完成"
fi

# ============================================================
# Step 3: 安装项目依赖
# ============================================================
info "Step 3: 安装项目依赖..."

${PIP_BIN} install -r "${PROJECT_ROOT}/requirements.txt"
ok "项目依赖安装完成"

# 确认关键包
info "验证关键包..."
${PYTHON_BIN} -c "
packages = {
    'torch': None, 'transformers': None, 'peft': None,
    'accelerate': None, 'omegaconf': None, 'datasets': None,
    'tqdm': None, 'rich': None, 'sklearn': 'scikit-learn',
    'pytest': None, 'yaml': 'PyYAML',
}
all_ok = True
for pkg, pip_name in packages.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'OK')
        print(f'  ✅ {pip_name or pkg}: {ver}')
    except ImportError:
        print(f'  ❌ {pip_name or pkg}: 未安装')
        all_ok = False
if not all_ok:
    raise SystemExit('部分包安装失败')
"
ok "所有关键包验证通过"

# ============================================================
# Step 4: 检查 GPU 环境
# ============================================================
info "Step 4: 检查 GPU 环境..."

if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
    echo ""
    
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    ok "检测到 ${GPU_COUNT} 块 GPU"
    
    ${PYTHON_BIN} -c "
import torch
print(f'  CUDA 可用: {torch.cuda.is_available()}')
print(f'  CUDA 版本: {torch.version.cuda}')
print(f'  GPU 数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem_gb = torch.cuda.get_device_properties(i).total_mem / 1024**3
    print(f'  GPU {i}: {name} ({mem_gb:.1f} GB)')
"
else
    warn "nvidia-smi 未找到 — 当前机器可能没有 GPU"
    warn "训练和真实模型推理需要 GPU，规则模式评测可以在 CPU 上运行"
fi

# ============================================================
# Step 5: 检查模型文件
# ============================================================
info "Step 5: 检查 Qwen3-1.7B 模型文件..."

if [ -d "${MODEL_PATH}" ]; then
    MODEL_SIZE=$(du -sh "${MODEL_PATH}" 2>/dev/null | cut -f1)
    ok "模型目录存在: ${MODEL_PATH} (${MODEL_SIZE})"
    
    # 检查关键文件
    for f in config.json model.safetensors.index.json tokenizer.json; do
        if [ -f "${MODEL_PATH}/${f}" ]; then
            echo -e "  ✅ ${f}"
        else
            warn "  ⚠️  ${f} 不存在"
        fi
    done
else
    fail "模型目录不存在: ${MODEL_PATH}"
    echo ""
    echo "  请下载 Qwen3-1.7B 模型到该目录，或修改配置文件:"
    echo "  - configs/model/swa_qwen.yaml"
    echo "  - configs/model/fullattn_qwen.yaml"
    echo ""
fi

# ============================================================
# Step 6: 安装项目 (editable mode)
# ============================================================
info "Step 6: 设置项目路径..."

# 检查是否有 pyproject.toml 可以 pip install -e
if [ -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    ${PIP_BIN} install -e "${PROJECT_ROOT}" --no-deps 2>/dev/null || true
    ok "项目以 editable 模式安装"
else
    # 手动添加 PYTHONPATH
    warn "未找到 pyproject.toml，请确保运行时设置 PYTHONPATH"
    echo "  export PYTHONPATH=${PROJECT_ROOT}"
fi

# ============================================================
# Step 7: 运行单元测试 (快速验证)
# ============================================================
info "Step 7: 运行单元测试..."

export PYTHONPATH="${PROJECT_ROOT}"
TEST_RESULT=$(${PYTHON_BIN} -m pytest "${PROJECT_ROOT}/tests/" -q --tb=line 2>&1 | tail -3)
echo "  ${TEST_RESULT}"

if echo "${TEST_RESULT}" | grep -q "passed"; then
    ok "单元测试通过"
else
    warn "部分测试未通过，请检查输出"
fi

# ============================================================
# Step 8: 快速模型加载测试 (如果有 GPU)
# ============================================================
GPU_AVAIL=$(${PYTHON_BIN} -c "import torch; print(torch.cuda.is_available())" 2>&1)

if [ "$GPU_AVAIL" = "True" ] && [ -d "${MODEL_PATH}" ]; then
    info "Step 8: 快速模型加载测试..."
    
    ${PYTHON_BIN} -c "
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '${MODEL_PATH}'
print(f'  加载 tokenizer...')
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f'  ✅ Tokenizer 加载完成 ({time.time()-t0:.1f}s)')
print(f'  vocab_size={tokenizer.vocab_size}')

print(f'  加载模型 (bfloat16)...')
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
)
print(f'  ✅ 模型加载完成 ({time.time()-t0:.1f}s)')

# 快速推理测试
inputs = tokenizer('Hello, I am', return_tensors='pt').to('cuda:0')
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
text = tokenizer.decode(out[0], skip_special_tokens=True)
print(f'  ✅ 推理测试通过: \"{text[:80]}...\"')

# 显存使用
mem_gb = torch.cuda.max_memory_allocated() / 1024**3
print(f'  峰值显存: {mem_gb:.2f} GB')

# 清理
del model
torch.cuda.empty_cache()
print(f'  ✅ 模型清理完成')
" 2>&1
    
    ok "模型加载和推理测试通过"
else
    info "Step 8: 跳过模型加载测试 (需要 GPU + 模型文件)"
fi

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================================"
echo -e "  ${GREEN}✅ 环境配置完成！${NC}"
echo "============================================================"
echo ""
echo "  后续步骤:"
echo ""

if [ "$USE_VENV" = true ]; then
    echo "  1. 激活环境:"
    echo "     source ${VENV_PATH}/bin/activate"
else
    echo "  1. 激活环境:"
    echo "     conda activate ${ENV_NAME}"
fi

echo ""
echo "  2. 设置项目路径 (如果未用 pip install -e):"
echo "     export PYTHONPATH=${PROJECT_ROOT}"
echo ""
echo "  3. 运行实验:"
echo "     bash scripts/run_experiments.sh"
echo ""
echo "  或者分步运行:"
echo "     # Demo 对话"
echo "     python scripts/run_chat.py --config-name swa_mom --mode demo"
echo ""
echo "     # 消融评测"
echo "     python scripts/run_ablation.py --num-samples 20"
echo ""
echo "     # 训练 L2 聚合器 (LoRA)"
echo "     python scripts/train_l2.py"
echo ""
echo "============================================================"
