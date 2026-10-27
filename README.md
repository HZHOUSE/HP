# 右焊盘 7 模型对比包（hp2_cmp7）

同一个数据集、同一个 batch、同一个学习率、同一个 epoch 数，比 7 个主干网络。
**全部从零训练，不加载预训练权重。**

## 目录

```
hp2_cmp7/
  src/
    train_compare.py      训练 + 评估（只读 train/ 和 val/，不碰 test/）
    diagnose_samples.py   错误诊断：哪张图拖累了指标、哪些图偏暗/糊/标错
  data/
    rightpad_v2_split/    数据集 7:2:1
      train/ 2805 张   val/ 802 张   test/ 401 张
      split_manifest.csv
    rightpad_v2/
      _labels.csv         每张图的标签来源（human / model）
  requirements.txt
  README.md
```

8 个类别：`OK / SOLDER_NOT_OBVIOUS / DIRTY_SOLDER / WIRE_LINE_OFFSET /
OVER_WELD / NO_JOINT / NO_WIRE_LINE / WIRE_RESIDUE`

## 环境

```bash
pip install -r requirements.txt
# GPU 版按自己的 CUDA 改 --index-url，例如：
# pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

## 训练

**A. 题目要求的配置（lr = 5e-6）**

```bash
python src/train_compare.py \
    --data data/rightpad_v2_split --out runs/compare7_lr5e6 \
    --epochs 50 --batch-size 32 \
    --lr 5e-6 --optimizer sgd --sched step --lr-decay 2 --lr-step 10 \
    --img-size 160 --workers 8 --seed 42 --save-pred
```

> 注意：从零训练用 5e-6 基本学不动（实测 2 个 epoch 后 val acc 仍在 0.36 附近）。
> 这条命令留着做「学习率不匹配」的对照实验。

**B. 从零训练应该用的配置（推荐）**

```bash
python src/train_compare.py \
    --data data/rightpad_v2_split --out runs/compare7_lr1e2 \
    --epochs 50 --batch-size 32 \
    --lr 0.01 --optimizer sgd --sched step --lr-decay 2 --lr-step 10 \
    --img-size 160 --workers 8 --seed 42 --save-pred
```

在 4090 上 7 个模型 × 50 epoch 约 **35~45 分钟**（vgg16 最慢，约 12 分钟）。

### 参数对应

| 要求 | 参数 |
|---|---|
| 无预训练权重 | `weights=None`（默认） |
| lr = 5×10⁻⁶ | `--lr 5e-6` |
| 衰减率 r = 2 | `--lr-decay 2`，配合 `--lr-step 10` = 每 10 轮学习率减半 |
| epoch = 50 | `--epochs 50` |
| batch = 32 | `--batch-size 32` |
| 相同 data | 同一份 `data/rightpad_v2_split` |

模型：`vgg16, googlenet, resnet18, resnet34, resnet50, shufflenet, densenet`
（用 `--models resnet18,densenet` 可只跑其中几个）

## 输出

```
runs/compare7_lr1e2/
  results.csv                 每个模型一行：val_acc / val_P / val_R / val_F1
  per_class_<模型>_best_val.csv   每类的 accuracy / precision / recall / f1
  confusion_<模型>_best_val.csv   混淆矩阵
  log_<模型>.csv              每轮 loss / 各项指标 / 学习率
  pred_<模型>_best_val.csv    逐张预测（--save-pred 才有）
  pred_<模型>_last_val.csv    最后一轮的逐张预测
  config.json                 本次运行的完整配置
```

看总表：

```bash
column -s, -t < runs/compare7_lr1e2/results.csv
```

## 错误诊断（跑完训练再执行）

```bash
python src/diagnose_samples.py \
    --runs runs/compare7_lr1e2 \
    --data data/rightpad_v2_split \
    --labels data/rightpad_v2/_labels.csv \
    --out runs/compare7_lr1e2/diagnosis
```

产出：

| 文件 | 内容 |
|---|---|
| `diagnosis.md` | 结论：误差来源 TOP10、标签来源检验、每类指标、97% 可行性 |
| `class_impact.csv` | 每类 P/R/F1 + 拖累该类的 top-10 图片 |
| `error_samples.csv` | 每张图：投票、置信度、锐度/亮度/过曝/熵、标签来源 |
| `hard_samples.md` | 分组清单：疑似标错 / 模糊 / 过曝 / 偏暗 / 模型分歧 |
| `review_hard.html` | 同上，带图片的审核页（单文件，可直接发人） |

## 说明

- 训练脚本只读取 `train/` 和 `val/`，`test/` 全程不加载（`config.json` 里记录 `test_accessed=false`）。
- 模型按 `val macro-F1` 选最优轮；`best` 与 `last` 两套权重都会保存。
- 数据集里 3354 张为人工标注、3117 张为模型预标注（见 `_labels.csv` 的 `source` 列）。
