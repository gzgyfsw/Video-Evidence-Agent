#!/usr/bin/env python3
"""
Qwen3-VL-4B SFT 冷启动训练
目标：让模型学会输出 <reason><confidence><answer> 三标签格式，作为 GRPO 的起点。
纯监督微调（teacher-forcing），不做奖励门控、不在训练循环里跑 generate()。
"""

import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
# 强制用 decord 解码视频：1080p 大文件用默认 torchvision 后端会卡死。
os.environ.setdefault('FORCE_QWENVL_VIDEO_READER', 'decord')

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

print("=" * 70)
print("🚀 SFT 冷启动训练 - Qwen3-VL-4B-Instruct")
print("=" * 70)

import json
import time
import torch
import gc
import logging
from datetime import datetime
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from scripts.prepare_video_safety_rl_data import rewrite_media_path
from scripts.video_safety_schema import STRUCTURED_PROMPT, build_sft_target

# 配置日志
log_dir = REPO_ROOT / "outputs" / "sft_training_logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"sft_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型路径。默认从 base model 开始；继续训练时可传入上一个 SFT checkpoint。
base_model_path = str(REPO_ROOT / "rl-qwen-data" / "Qwen3-VL-4B-Instruct")
model_path = os.environ.get("SFT_MODEL_PATH", base_model_path)

print(f"\n📦 模型: {model_path}")

# 训练配置
output_dir = REPO_ROOT / "outputs" / "sft_training_checkpoints"
num_epochs = int(os.environ.get("SFT_NUM_EPOCHS", "2"))
start_epoch = int(os.environ.get("SFT_START_EPOCH", "1"))
learning_rate = 5e-6

print(f"\n⚙️  训练配置:")
print(f"  输出目录: {output_dir}")
print(f"  训练轮数: {num_epochs}")
print(f"  起始 epoch 编号: {start_epoch}")
print(f"  学习率: {learning_rate}")

output_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print("📦 加载模型...")
print("=" * 70)

local_files_only = Path(model_path).exists()
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    local_files_only=local_files_only,
)
processor_path = model_path if Path(model_path).exists() else base_model_path
processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True, local_files_only=True)

memory_allocated = torch.cuda.memory_allocated() / 1024**3
print(f"✅ 模型加载成功！显存: {memory_allocated:.2f} GB")

print("\n" + "=" * 70)
print("📖 加载训练数据...")
print("=" * 70)

# 注意：只能用 train 切分，不能用 video_safety_rl_all.jsonl ——
# all.jsonl 包含了 video_safety_rl_test.jsonl 的全部样本，
# 如果 SFT/GRPO 都在 all.jsonl 上训练，会直接把测试集也训练进去，
# 之后"训练前 vs 训练后在测试集上对比准确率"的评估会因为数据泄漏而失效。
DATA_PATH = os.environ.get(
    "SFT_TRAIN_DATA", str(REPO_ROOT / "data" / "video_safety_rl_train.jsonl")
)
with open(DATA_PATH) as f:
    train_data = [json.loads(line) for line in f]
print(f"✅ 训练数据: {len(train_data)} 个样本 ({DATA_PATH})")

# 设置优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 长视频不做限制会导致抽帧极慢甚至卡死/OOM（部分素材是 1080p 完整片段，不是短clip）。
# 三个坑（GRPO 那边的 data_processor.py 也有同样隐患，见注释）：
#  1. 不能写死 nframes=128——qwen_vl_utils 要求 nframes <= 视频实际总帧数，
#     短视频（比如只有 46 帧）传 nframes=128 会直接抛 ValueError。用 max_frames 才是安全软上限。
#  2. max_pixels 不能小于单帧下限 VIDEO_MIN_TOKEN_NUM(128) * image_factor(32)^2 = 131072，
#     否则 smart_resize 断言失败。286*286=81796 太小会崩，这里用 384*384=147456。
#  3. 1080p 大文件用 torchvision 后端解码会卡死；已安装 decord（qwen 官方推荐后端）解决。
MAX_VIDEO_FRAMES = 128
MAX_VIDEO_PIXELS = 384 * 384


def cap_video_inputs(messages):
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        item["video"] = rewrite_media_path(item["video"], REPO_ROOT)
                        if "nframes" not in item:
                            item.setdefault("max_frames", MAX_VIDEO_FRAMES)
                        if "max_pixels" not in item:
                            item["max_pixels"] = MAX_VIDEO_PIXELS
                    elif isinstance(item, dict) and item.get("type") == "image":
                        item["image"] = rewrite_media_path(item["image"], REPO_ROOT)

# 训练循环
print("\n" + "=" * 70)
print("🚀 开始训练")
print("=" * 70)

training_start_time = time.time()
avg_loss = 0.0

for epoch in range(num_epochs):
    epoch_num = start_epoch + epoch
    print(f"\n{'=' * 70}")
    print(f"📊 Epoch {epoch_num} ({epoch + 1}/{num_epochs})")
    print(f"{'=' * 70}")

    epoch_start = time.time()
    total_loss = 0.0
    processed = 0
    category_counts = {}

    for idx, sample in enumerate(train_data):
        try:
            if idx % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # 准备数据
            messages = json.loads(sample['message'])
            true_category = sample.get('answer', sample.get('category', ''))

            messages[0]['content'][-1]['text'] = STRUCTURED_PROMPT
            cap_video_inputs(messages)

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            result = process_vision_info(messages)
            if len(result) == 3:
                image_inputs, video_inputs, _ = result
            else:
                image_inputs, video_inputs = result

            # 监督目标：prompt + 固定格式的 <reason><confidence><answer>
            target_text = text + " " + build_sft_target(true_category)
            target_inputs = processor(
                text=[target_text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            outputs = model(**target_inputs, labels=target_inputs.input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            print(f"  [{idx + 1:3d}/{len(train_data)}] {true_category:25s} | 损失={loss.item():.4f}")

            total_loss += loss.item()
            processed += 1
            category_counts[true_category] = category_counts.get(true_category, 0) + 1

            del target_inputs, outputs

        except Exception as e:
            print(f"  [{idx + 1:3d}/{len(train_data)}] ⚠️  错误: {str(e)[:100]}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    # Epoch 统计
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / processed if processed > 0 else 0

    print(f"\n📊 Epoch {epoch_num} 总结:")
    print(f"  处理样本: {processed}/{len(train_data)}")
    print(f"  平均损失: {avg_loss:.4f}")
    print(f"  耗时: {epoch_time:.1f} 秒")

    memory_used = torch.cuda.memory_allocated() / 1024**3
    print(f"  显存占用: {memory_used:.2f} GB")

    logger.info(f"Epoch {epoch_num} 完成 - 平均损失: {avg_loss:.4f}, 耗时: {epoch_time:.1f}s")

    # 保存 checkpoint
    checkpoint_path = output_dir / f"qwen3vl_4b_sft_checkpoint_epoch_{epoch_num}"
    print(f"\n💾 保存 checkpoint: {checkpoint_path}")

    try:
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
        print(f"✅ Checkpoint 已保存")
    except Exception as e:
        print(f"⚠️  Checkpoint 保存失败: {e}")

    gc.collect()
    torch.cuda.empty_cache()

total_training_time = time.time() - training_start_time

print("\n" + "=" * 70)
print("🎉 训练完成！")
print("=" * 70)

final_epoch = start_epoch + num_epochs - 1
final_checkpoint = output_dir / f"qwen3vl_4b_sft_checkpoint_epoch_{final_epoch}"

training_summary = {
    "timestamp": datetime.now().isoformat(),
    "model_path": model_path,
    "base_model_path": base_model_path,
    "train_data_path": DATA_PATH,
    "num_epochs": num_epochs,
    "start_epoch": start_epoch,
    "final_epoch": final_epoch,
    "learning_rate": learning_rate,
    "train_data_size": len(train_data),
    "final_avg_loss": float(avg_loss),
    "total_training_time": float(total_training_time),
    "checkpoint_path": str(final_checkpoint),
    "training_method": "Supervised Fine-Tuning (teacher-forcing)",
    "category_counts": category_counts,
}

training_summary_file = output_dir / "sft_training_summary.json"
with open(training_summary_file, 'w') as f:
    json.dump(training_summary, f, indent=2, ensure_ascii=False)

logger.info(f"训练完成！最终平均损失: {avg_loss:.4f}")
logger.info(f"训练摘要已保存到: {training_summary_file}")

# 建一个固定路径的软链接指向最终 checkpoint，方便 GRPO 脚本
# (rl_grpo_train_qwen3vl_4b.sh 默认从 outputs/sft_training/qwen3vl_4b_sft_final 读取)
sft_final_link = REPO_ROOT / "outputs" / "sft_training" / "qwen3vl_4b_sft_final"
sft_final_link.parent.mkdir(parents=True, exist_ok=True)
if sft_final_link.is_symlink() or sft_final_link.exists():
    sft_final_link.unlink()
sft_final_link.symlink_to(final_checkpoint)
logger.info(f"已建立软链接: {sft_final_link} -> {final_checkpoint}")

print(f"""
📁 模型保存在: {output_dir}
📊 最终 checkpoint: {final_checkpoint}
⏱️  总训练时间: {total_training_time/60:.1f} 分钟

💡 使用训练后的模型:
```python
from transformers import AutoModelForVision2Seq, AutoProcessor
model = AutoModelForVision2Seq.from_pretrained(
    "{final_checkpoint}",
    torch_dtype="bfloat16",
    device_map="auto"
)
```
""")
