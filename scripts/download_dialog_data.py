#!/usr/bin/env python3
"""
下载多轮对话数据集并转换为 train_mag.py 可用的 JSONL 格式。

支持数据集:
  1. DailyDialog   — 完全公开, 无需认证 (推荐)
  2. MSC            — 需要 HuggingFace 认证 (更优质)
  3. SODA           — 完全公开, 大规模对话 (备选)

用法:
  # 最简单 — 下载 DailyDialog (无需认证, 推荐)
  python scripts/download_dialog_data.py

  # 下载 MSC (需要先 huggingface-cli login)
  python scripts/download_dialog_data.py --dataset msc --hf_token YOUR_TOKEN

  # 下载 SODA
  python scripts/download_dialog_data.py --dataset soda

  # 下载后训练
  python scripts/train_mag.py \\
      --data_source jsonl \\
      --data_path data/raw/dailydialog_train.jsonl \\
      --max_real_samples 5000 \\
      --mag_injection_layers 6 12 18 23 \\
      --num_epochs 3 --lr 1e-4

输出格式 (JSONL, 每行一个对话):
  {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}], "personas": []}
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("download_data")


# ========== 通用工具 ==========

def download_file(url: str, save_path: str, token: str = "") -> str:
    """下载文件, 支持断点续传提示。"""
    import urllib.request
    import urllib.error

    logger.info(f"下载: {url}")
    logger.info(f"保存到: {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None

            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB
            with open(save_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  进度: {downloaded / 1e6:.1f}MB / {total / 1e6:.1f}MB ({pct:.1f}%)", end="", flush=True)
                    else:
                        print(f"\r  已下载: {downloaded / 1e6:.1f}MB", end="", flush=True)
            print()  # 换行

    except urllib.error.HTTPError as e:
        if e.code == 401:
            logger.error("认证失败 (401)。该数据集需要 HuggingFace 认证。")
            logger.error("请先运行: huggingface-cli login")
            logger.error("或使用: --hf_token YOUR_TOKEN")
            sys.exit(1)
        elif e.code == 404:
            logger.error(f"文件不存在 (404): {url}")
            sys.exit(1)
        else:
            raise

    logger.info(f"下载完成: {save_path} ({os.path.getsize(save_path) / 1e6:.1f}MB)")
    return save_path


def turns_to_messages(turns: list[str]) -> list[dict]:
    """将对话轮次列表转为 messages 格式。"""
    messages = []
    for i, text in enumerate(turns):
        text = text.strip()
        if not text:
            continue
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": text})
    return messages


# ========== DailyDialog ==========

def download_dailydialog(output_dir: str, hf_mirror: str = "") -> str:
    """下载 DailyDialog 数据集 (完全公开, 约 11K 条对话)。

    DailyDialog 是高质量的英文日常对话数据集, 每条包含 5-10 轮对话。
    使用 roskoN/dailydialog 社区镜像 (有标准 zip 文件, 无需 loading script)。
    """
    base_url = hf_mirror or "https://huggingface.co"

    # 策略 1: roskoN/dailydialog 社区镜像 (已确认有 train.zip, 仅 1.9MB)
    zip_url = f"{base_url}/datasets/roskoN/dailydialog/resolve/main/train.zip"
    zip_path = os.path.join(output_dir, "dailydialog_train.zip")

    try:
        download_file(zip_url, zip_path)
        return _convert_dailydialog_raw(zip_path, output_dir)
    except Exception as e:
        logger.warning(f"roskoN 镜像下载失败: {e}, 尝试原始源...")

    # 策略 2: 原始 IJCNLP 源
    raw_url = "http://yanran.li/files/ijcnlp_dailydialog.zip"
    zip_path2 = os.path.join(output_dir, "ijcnlp_dailydialog.zip")
    try:
        download_file(raw_url, zip_path2)
        return _convert_dailydialog_raw(zip_path2, output_dir)
    except Exception as e:
        logger.error(f"DailyDialog 所有下载方式均失败: {e}")
        sys.exit(1)


def _convert_dailydialog_parquet(parquet_path: str, output_dir: str) -> str:
    """将 DailyDialog parquet 转为 JSONL。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.info("安装 pyarrow...")
        os.system(f"{sys.executable} -m pip install pyarrow -q")
        import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    logger.info(f"读取 {len(df)} 条 DailyDialog 对话")

    output_path = os.path.join(output_dir, "dailydialog_train.jsonl")
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            dialog = row.get("dialog", row.get("dialogue", []))
            if not dialog or len(dialog) < 4:
                continue
            messages = turns_to_messages(dialog)
            if len(messages) >= 4:
                obj = {"messages": messages, "personas": []}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"转换完成: {output_path} ({count} 条对话)")
    # 清理 parquet
    os.remove(parquet_path)
    return output_path


def _convert_dailydialog_raw(zip_path: str, output_dir: str) -> str:
    """将 DailyDialog zip 转为 JSONL。

    兼容两种格式:
      - roskoN/dailydialog: zip 内直接包含 dialogues_train.txt 或类似文件
      - IJCNLP 原始: zip 内嵌套目录, 含 dialogues_train.txt
    """
    import zipfile
    import shutil

    extract_dir = os.path.join(output_dir, "_dailydialog_raw")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # 打印内容方便调试
        names = zf.namelist()
        logger.info(f"Zip 内容 ({len(names)} 个文件): {names[:10]}...")
        zf.extractall(extract_dir)

    # 寻找包含对话的 txt 文件
    # 优先匹配: dialogues_train.txt (而非 dialogues_act_train.txt)
    train_file = None
    all_txt_files = []
    candidates = []  # (优先级, 路径)
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not fname.endswith(".txt"):
                continue
            all_txt_files.append(fpath)
            fname_lower = fname.lower()

            # 最高优先级: dialogues_train.txt (精确匹配主对话文件)
            if fname_lower == "dialogues_train.txt":
                candidates.append((0, fpath))
            # 次优先: 包含 "dialog" 和 "train" 但不含 "act"/"emotion"
            elif "train" in fname_lower and ("dialog" in fname_lower) \
                    and "act" not in fname_lower and "emotion" not in fname_lower:
                candidates.append((1, fpath))
            # 低优先级: 含 "train" 的 txt
            elif "train" in fname_lower:
                candidates.append((2, fpath))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        train_file = candidates[0][1]
    elif all_txt_files:
        # 如果还没找到, 用最大的 txt 文件
        train_file = max(all_txt_files, key=os.path.getsize)
        logger.info(f"未找到明确的 train 文件, 使用最大 txt: {train_file}")

    if not train_file:
        logger.error(f"在解压目录中未找到训练对话文件: {extract_dir}")
        logger.error(f"目录内容: {os.listdir(extract_dir)}")
        sys.exit(1)

    logger.info(f"使用对话文件: {train_file}")

    output_path = os.path.join(output_dir, "dailydialog_train.jsonl")
    count = 0
    with open(train_file, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # DailyDialog 格式: utterance1 __eou__ utterance2 __eou__ ...
            turns = [t.strip() for t in line.split("__eou__") if t.strip()]
            if len(turns) < 4:
                continue
            messages = turns_to_messages(turns)
            if len(messages) >= 4:
                obj = {"messages": messages, "personas": []}
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"转换完成: {output_path} ({count} 条对话)")
    # 清理临时文件
    shutil.rmtree(extract_dir, ignore_errors=True)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    return output_path


# ========== MSC ==========

def download_msc(output_dir: str, hf_token: str = "", hf_mirror: str = "",
                 subset: str = "session_1") -> str:
    """下载 MSC (Multi-Session Chat) 数据集。

    **需要 HuggingFace 认证**: 
      1. 访问 https://huggingface.co/datasets/facebook/msc 申请访问
      2. huggingface-cli login  或  传 --hf_token
    """
    base_url = hf_mirror or "https://huggingface.co"

    # MSC parquet 路径 (datasets v2 标准)
    parquet_url = (
        f"{base_url}/datasets/facebook/msc/resolve/main/"
        f"{subset}/train/0000.parquet"
    )

    token = hf_token or os.environ.get("HF_TOKEN", "")

    parquet_path = os.path.join(output_dir, f"msc_{subset}_train.parquet")

    try:
        download_file(parquet_url, parquet_path, token=token)
    except SystemExit:
        # 认证失败, 已打印提示
        raise
    except Exception as e:
        # 尝试另一种路径格式
        alt_url = (
            f"{base_url}/datasets/facebook/msc/resolve/main/"
            f"data/{subset}-train.parquet"
        )
        try:
            download_file(alt_url, parquet_path, token=token)
        except Exception as e2:
            logger.error(f"MSC 下载失败: {e2}")
            logger.error("请确认:")
            logger.error("  1. 已在 https://huggingface.co/datasets/facebook/msc 申请访问")
            logger.error("  2. 已通过 huggingface-cli login 或 --hf_token 提供认证")
            sys.exit(1)

    return _convert_msc_parquet(parquet_path, output_dir, subset)


def _convert_msc_parquet(parquet_path: str, output_dir: str, subset: str) -> str:
    """将 MSC parquet 转为 JSONL。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.info("安装 pyarrow...")
        os.system(f"{sys.executable} -m pip install pyarrow -q")
        import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    logger.info(f"读取 {len(df)} 条 MSC 对话")

    output_path = os.path.join(output_dir, f"msc_{subset}_train.jsonl")
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            dialog = row.get("dialog", row.get("dialogue", []))
            personas = row.get("personas", row.get("persona", []))

            if not dialog or len(dialog) < 4:
                continue

            # MSC dialog 可能是字符串列表或字典列表
            if isinstance(dialog[0], str):
                messages = turns_to_messages(dialog)
            elif isinstance(dialog[0], dict):
                messages = []
                for j, utt in enumerate(dialog):
                    messages.append({
                        "role": utt.get("role", "user" if j % 2 == 0 else "assistant"),
                        "content": utt.get("text", utt.get("content", "")),
                    })
            else:
                continue

            if len(messages) < 4:
                continue

            # personas 规范化
            if personas and isinstance(personas, list) and isinstance(personas[0], list):
                personas = personas[0]

            obj = {"messages": messages, "personas": personas if isinstance(personas, list) else []}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"转换完成: {output_path} ({count} 条对话)")
    os.remove(parquet_path)
    return output_path


# ========== SODA ==========

def download_soda(output_dir: str, hf_mirror: str = "", max_samples: int = 10000) -> str:
    """下载 SODA 数据集 (完全公开, 约 1.2M 条对话, 这里只取前 max_samples 条)。

    SODA 是 Allen AI 发布的大规模社交对话数据集。
    """
    base_url = hf_mirror or "https://huggingface.co"

    # SODA 的 valid.parquet 较小 (82MB), train.parquet 太大 (688MB)
    # 优先使用 valid 集, 已够用于 MAG 训练
    parquet_url = f"{base_url}/datasets/allenai/soda/resolve/main/valid.parquet"
    parquet_path = os.path.join(output_dir, "soda_valid.parquet")

    try:
        download_file(parquet_url, parquet_path)
    except Exception as e1:
        logger.warning(f"SODA valid.parquet 下载失败: {e1}, 尝试 train...")
        # 回退到 train (688MB, 较大)
        train_url = f"{base_url}/datasets/allenai/soda/resolve/main/train.parquet"
        parquet_path = os.path.join(output_dir, "soda_train.parquet")
        try:
            download_file(train_url, parquet_path)
        except Exception as e2:
            logger.error(f"SODA 下载失败: {e2}")
            sys.exit(1)

    return _convert_soda_parquet(parquet_path, output_dir, max_samples)


def _convert_soda_parquet(parquet_path: str, output_dir: str, max_samples: int) -> str:
    """将 SODA parquet 转为 JSONL。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        os.system(f"{sys.executable} -m pip install pyarrow -q")
        import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    logger.info(f"读取 SODA: {len(df)} 条对话 (取前 {max_samples} 条)")

    output_path = os.path.join(output_dir, "soda_train.jsonl")
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if count >= max_samples:
                break
            dialog = row.get("dialogue", row.get("dialog", []))
            if not dialog or len(dialog) < 4:
                continue
            messages = turns_to_messages(dialog)
            if len(messages) >= 4:
                obj = {"messages": messages, "personas": []}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"转换完成: {output_path} ({count} 条对话)")
    os.remove(parquet_path)
    return output_path


def _convert_soda_jsonl_gz(gz_path: str, output_dir: str, max_samples: int) -> str:
    """将 SODA jsonl.gz 转为 JSONL。"""
    import gzip

    output_path = os.path.join(output_dir, "soda_train.jsonl")
    count = 0
    with gzip.open(gz_path, "rt", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if count >= max_samples:
                break
            obj = json.loads(line.strip())
            dialog = obj.get("dialogue", obj.get("dialog", []))
            if not dialog or len(dialog) < 4:
                continue
            messages = turns_to_messages(dialog)
            if len(messages) >= 4:
                out = {"messages": messages, "personas": []}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"转换完成: {output_path} ({count} 条对话)")
    os.remove(gz_path)
    return output_path


# ========== 主流程 ==========

def main():
    parser = argparse.ArgumentParser(
        description="下载多轮对话数据集并转为 JSONL 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载 DailyDialog (最简单, 无需认证)
  python scripts/download_dialog_data.py

  # 下载 MSC (需 HF 认证)
  python scripts/download_dialog_data.py --dataset msc --hf_token hf_xxx

  # 下载后训练
  python scripts/train_mag.py \\
      --data_source jsonl \\
      --data_path data/raw/dailydialog_train.jsonl \\
      --mag_injection_layers 6 12 18 23 --num_epochs 3 --lr 1e-4
        """,
    )
    parser.add_argument("--dataset", type=str, default="dailydialog",
                        choices=["dailydialog", "msc", "soda", "all"],
                        help="要下载的数据集 (默认: dailydialog)")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="保存目录 (默认: data/raw)")
    parser.add_argument("--hf_token", type=str, default="",
                        help="HuggingFace Token (下载 MSC 需要)")
    parser.add_argument("--hf_mirror", type=str, default="",
                        help="HuggingFace 镜像 URL (如: https://hf-mirror.com)")
    parser.add_argument("--msc_subset", type=str, default="session_1",
                        help="MSC subset (默认: session_1)")
    parser.add_argument("--max_soda_samples", type=int, default=10000,
                        help="SODA 最大样本数 (默认: 10000)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mirror = args.hf_mirror or os.environ.get("HF_ENDPOINT", "")

    datasets_to_download = []
    if args.dataset == "all":
        datasets_to_download = ["dailydialog", "soda"]
        if args.hf_token or os.environ.get("HF_TOKEN"):
            datasets_to_download.append("msc")
        else:
            logger.info("未提供 HF Token, 跳过 MSC (需要认证)")
    else:
        datasets_to_download = [args.dataset]

    results = {}

    for ds_name in datasets_to_download:
        logger.info(f"\n{'='*60}")
        logger.info(f"开始下载: {ds_name}")
        logger.info(f"{'='*60}")

        try:
            if ds_name == "dailydialog":
                path = download_dailydialog(args.output_dir, hf_mirror=mirror)
                results[ds_name] = path
            elif ds_name == "msc":
                path = download_msc(args.output_dir, hf_token=args.hf_token,
                                    hf_mirror=mirror, subset=args.msc_subset)
                results[ds_name] = path
            elif ds_name == "soda":
                path = download_soda(args.output_dir, hf_mirror=mirror,
                                     max_samples=args.max_soda_samples)
                results[ds_name] = path
        except SystemExit:
            logger.error(f"{ds_name} 下载失败, 跳过")
            continue
        except Exception as e:
            logger.error(f"{ds_name} 下载失败: {e}")
            continue

    # 打印结果摘要
    logger.info(f"\n{'='*60}")
    logger.info("下载完成! 摘要:")
    logger.info(f"{'='*60}")

    if not results:
        logger.error("没有成功下载任何数据集!")
        sys.exit(1)

    for ds_name, path in results.items():
        # 统计行数
        with open(path, "r") as f:
            n_lines = sum(1 for _ in f)
        size_mb = os.path.getsize(path) / 1e6
        logger.info(f"  {ds_name}: {path} ({n_lines} 条对话, {size_mb:.1f}MB)")

    # 打印使用提示
    first_path = list(results.values())[0]
    logger.info(f"\n使用方法:")
    logger.info(f"  python scripts/train_mag.py \\")
    logger.info(f"      --model_path ../models/Qwen--Qwen3-8b/ \\")
    logger.info(f"      --data_source jsonl \\")
    logger.info(f"      --data_path {first_path} \\")
    logger.info(f"      --max_real_samples 5000 \\")
    logger.info(f"      --mag_injection_layers 6 12 18 23 \\")
    logger.info(f"      --num_epochs 3 --lr 1e-4")


if __name__ == "__main__":
    main()
