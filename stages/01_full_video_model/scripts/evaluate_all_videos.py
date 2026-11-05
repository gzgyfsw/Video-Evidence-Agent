#!/usr/bin/env python3
"""
完整视频分类评估 - 测试所有样本并生成报告
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import torch

print("=" * 70)
print("🎬 完整视频分类评估")
print("=" * 70)

# 模型路径
model_path = "/home/user/MM-EUREKA-qwen/Qwen3-VL-8B-Instruct"

# 加载测试数据
print("\n📖 加载测试数据...")
with open("/home/user/MM-EUREKA-qwen/data/video_safety_test_samples.json") as f:
    test_data = json.load(f)

samples = test_data["samples"]
print(f"✅ 找到 {len(samples)} 个样本")

# 类别列表
categories = ["正常", "涉黄", "涉政", "涉暴"]

# 加载模型
print(f"\n🚀 加载模型...")
print(f"   模型路径: {model_path}")

start_time = time.time()

model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

load_time = time.time() - start_time
print(f"✅ 模型加载成功！耗时: {load_time:.1f} 秒")

# 存储结果
results = []
confusion_matrix = defaultdict(lambda: defaultdict(int))

# 分类提示词
prompt = "这个视频属于什么类别？正常、涉黄、涉政、涉暴？请直接回答类别名称。"

print("\n" + "=" * 70)
print("🔥 开始批量推理...")
print("=" * 70)

total_inference_time = 0

for idx, sample in enumerate(samples, 1):
    print(f"\n[{idx}/{len(samples)}] 测试视频: {sample['id']}")
    print(f"    路径: {sample['video_path']}")
    print(f"    真实标签: {sample['label']}")

    try:
        # 准备输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": sample["video_path"]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 处理输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = process_vision_info(messages)
        if len(result) == 3:
            image_inputs, video_inputs, _ = result
        else:
            image_inputs, video_inputs = result

        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 推理
        inference_start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        inference_time = time.time() - inference_start
        total_inference_time += inference_time

        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 提取类别
        predicted = "未识别"
        for category in categories:
            if category in output_text:
                predicted = category
                break

        # 存储结果
        result_record = {
            "id": sample['id'],
            "video_path": sample['video_path'],
            "true_label": sample['label'],
            "predicted": predicted,
            "output": output_text,
            "inference_time": inference_time,
            "correct": predicted == sample['label']
        }
        results.append(result_record)

        # 更新混淆矩阵
        confusion_matrix[sample['label']][predicted] += 1

        # 显示结果
        status = "✅" if predicted == sample['label'] else "❌"
        print(f"    预测类别: {predicted}")
        print(f"    推理时间: {inference_time:.1f}秒")
        print(f"    结果: {status}")
        if predicted != sample['label']:
            print(f"    模型输出: {output_text[:100]}...")

    except Exception as e:
        print(f"    ❌ 错误: {e}")
        results.append({
            "id": sample['id'],
            "video_path": sample['video_path'],
            "true_label": sample['label'],
            "predicted": "错误",
            "output": str(e),
            "inference_time": 0,
            "correct": False
        })

# 生成报告
print("\n" + "=" * 70)
print("📊 评估报告")
print("=" * 70)

# 计算指标
correct = sum(1 for r in results if r['correct'])
accuracy = correct / len(results) * 100

# 每个类别的精确率、召回率、F1
print("\n各类别性能:")
print("-" * 70)

for category in categories:
    tp = confusion_matrix[category][category]
    fp = sum(confusion_matrix[cat][category] for cat in categories if cat != category)
    fn = sum(confusion_matrix[category][cat] for cat in categories if cat != category)

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{category}:")
    print(f"  精确率: {precision:.1f}%")
    print(f"  召回率: {recall:.1f}%")
    print(f"  F1分数: {f1:.1f}%")

# 混淆矩阵
print("\n混淆矩阵:")
print("-" * 70)
print("      ", end="")
for pred in categories:
    print(f"{pred:>6}", end="")
print()
for true in categories:
    print(f"{true:>6}", end="")
    for pred in categories:
        count = confusion_matrix[true][pred]
        print(f"{count:>6}", end="")
    print()

# 总体统计
avg_inference_time = total_inference_time / len(results)

print("\n总体统计:")
print("-" * 70)
print(f"总样本数: {len(results)}")
print(f"正确数量: {correct}")
print(f"准确率: {accuracy:.1f}%")
print(f"平均推理时间: {avg_inference_time:.1f}秒")
print(f"总推理时间: {total_inference_time:.1f}秒")

# 错误案例
errors = [r for r in results if not r['correct']]
if errors:
    print(f"\n错误案例 ({len(errors)}个):")
    print("-" * 70)
    for err in errors:
        print(f"  {err['id']}: {err['true_label']} → {err['predicted']}")

print("\n" + "=" * 70)
print("✨ 评估完成！")
print("=" * 70)

# 保存结果
output_file = "/home/user/MM-EUREKA-qwen/results/evaluation_results.json"
Path(output_file).parent.mkdir(exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "summary": {
            "total_samples": len(results),
            "correct": correct,
            "accuracy": accuracy,
            "avg_inference_time": avg_inference_time
        },
        "confusion_matrix": dict(confusion_matrix),
        "results": results
    }, f, ensure_ascii=False, indent=2)

print(f"\n💾 详细结果已保存到: {output_file}")
