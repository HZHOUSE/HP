#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Sample-level error diagnosis: which images are actually costing us accuracy.

Reads the per-image predictions written by train_compare.py (--save-pred), the
split manifest, the label-source table and the images themselves, then answers:

  * which images drag down each class's P / R / F1
  * which samples are hard (all models wrong), blurry, badly exposed, or look
    mislabeled (most models confidently wrong)
  * does the auto-labeled part of the data hurt? (human vs model label source)
  * is 97% reachable with the current data at all?

    python src/diagnose_samples.py \
        --runs runs/diag_densenet \
        --data data/rightpad_v2_split \
        --labels data/rightpad_v2/_labels.csv \
        --out runs/diag_densenet/diagnosis

Outputs (into --out):
    error_samples.csv   one row per image: votes, confidence, quality, flags
    class_impact.csv    per class: P/R/F1, FN/FP counts and their worst images
    hard_samples.md     grouped lists (suspect label / blurry / exposure / ...)
    review_hard.html    the same groups as a visual review page
    diagnosis.md        conclusions, error sources, 97% verdict, next actions
"""
from __future__ import annotations

import argparse
import base64
import csv
import glob
import io
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass


def prf(trues, preds_list, classes):
    """Per-class precision / recall / F1 from a confusion matrix."""
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(trues, preds_list):
        cm[idx[t], idx[p]] += 1
    tp = np.diag(cm).astype(float)
    sup, pred_cnt = cm.sum(1), cm.sum(0)
    fp, fn = pred_cnt - tp, sup - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {
        "support": sup, "tp": tp, "fp": fp, "fn": fn,
        "prec": prec, "rec": rec, "f1": f1,
        "acc": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
        "macro_p": float(prec.mean()), "macro_r": float(rec.mean()),
        "macro_f1": float(f1.mean()),
    }


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------
def read_rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    return rows[0], rows[1:]


SPLIT_NAMES = ("test", "val", "train")
SELECTIONS = ("best", "last")


def parse_pred_name(base):
    """pred_resnet18_best_val -> ('resnet18', 'val', 'best'); pred_x -> ('x', 'test', '')."""
    parts = base.split("_")
    split, sel = "test", ""
    if parts and parts[-1] in SPLIT_NAMES:
        split = parts.pop()
    if parts and parts[-1] in SELECTIONS:
        sel = parts.pop()
    return "_".join(parts), split, sel


def load_all_preds(runs_dir):
    """{(model, split): {rel: (true, pred, conf)}}.

    Accepts both layouts used in this repo:
      pred_<model>.csv                      (test)
      pred_<model>_val.csv / _best_val.csv  (val, with or without selection tag)
    When a model has both best and last predictions for one split, best wins.
    """
    found = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "pred_*.csv"))):
        base = os.path.basename(p)[5:-4]                 # strip "pred_" and ".csv"
        model, split, sel = parse_pred_name(base)
        head, rows = read_rows(p)
        if not rows:
            continue
        col = {h: i for i, h in enumerate(head)}
        i_conf = col.get("conf", col.get("confidence"))
        if not {"file", "true", "pred"}.issubset(col) or i_conf is None:
            print(f"  ! skip {os.path.basename(p)}: need file/true/pred/conf(or confidence)")
            continue
        i_file, i_true, i_pred = col["file"], col["true"], col["pred"]
        d = {}
        for r in rows:
            try:
                d[r[i_file]] = (r[i_true], r[i_pred], float(r[i_conf]))
            except (ValueError, IndexError):
                continue
        key = (model, split)
        prev = found.get(key)
        # prefer "best" over "last"; otherwise first one wins
        if prev is None or (sel == "best" and prev[0] != "best"):
            found[key] = (sel, d)
    return {k: v[1] for k, v in found.items()}


def load_labels(path):
    """{rel: (source, conf)} where rel is the original raw path (series/file)."""
    if not path or not os.path.exists(path):
        return {}
    head, rows = read_rows(path)
    col = {h: i for i, h in enumerate(head)}
    if "rel" not in col:
        return {}
    i_rel = col["rel"]
    i_src = col.get("source", -1)
    i_conf = col.get("conf", -1)
    out = {}
    for r in rows:
        if i_rel >= len(r):
            continue
        src = r[i_src] if 0 <= i_src < len(r) else "unknown"
        try:
            conf = float(r[i_conf]) if 0 <= i_conf < len(r) else 1.0
        except ValueError:
            conf = 1.0
        out[r[i_rel]] = (src or "unknown", conf)
    return out


def to_rel(rel_in_class):
    """'DIRTY_SOLDER/HDZ__HDZ1_10.bmp' -> 'HDZ/HDZ1_10.bmp' (dataset name -> labels rel).

    Dataset files are named <series>__<original file>; the labels table keys on
    <series>/<original file> (the raw source folder, not the class folder).
    """
    cls, _, fname = rel_in_class.partition("/")
    if "__" in fname:
        series, rest = fname.split("__", 1)
        return f"{series}/{rest}"
    return f"{cls}/{fname}"


def data_uri(path, max_side=200, quality=82):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = max_side / max(w, h)
            if s < 1:
                im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


# --------------------------------------------------------------------------
# image quality
# --------------------------------------------------------------------------
def image_quality(path):
    """Blur / exposure / contrast / entropy, pure numpy (no cv2 on the cloud)."""
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L"), np.float32)
    except Exception:
        return None
    if g.ndim != 2 or g.shape[0] < 4 or g.shape[1] < 4:
        return None
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    hist = np.bincount(np.clip(g, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist / max(1.0, hist.sum())
    nz = p[p > 0]
    return {
        "lap_var": float(lap.var()),
        "brightness": float(g.mean()),
        "over_ratio": float((g >= 250).mean()),
        "under_ratio": float((g <= 20).mean()),
        "contrast": float(g.std()),
        "entropy": float(-(nz * np.log2(nz)).sum()),
    }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def build_samples(runs_dir, data_dir, labels, wanted_splits, max_images=0):
    preds = load_all_preds(runs_dir)
    if not preds:
        sys.exit(f"no pred_*.csv under {runs_dir} -- retrain with --save-pred")

    available = sorted({s for (_, s) in preds})
    splits = [s for s in wanted_splits if s in available]
    if not splits:
        splits = available
    if not splits:
        sys.exit(f"no predictions for splits {wanted_splits}; found {available}")

    samples = []
    all_models = set()
    for split in splits:
        split_models = sorted({m for (m, s) in preds if s == split})
        per_model = {m: preds[(m, split)] for m in split_models}
        common = set.intersection(*[set(v) for v in per_model.values()]) if per_model else set()
        if per_model:
            first = next(iter(per_model.values()))
            dropped = len(first) - len(common)
            if dropped:
                print(f"  ! {split}: {dropped} file(s) not predicted by every model, using {len(common)}")
        models = sorted(per_model)
        all_models.update(models)
        root = os.path.join(data_dir, split)
        if max_images:
            common = set(sorted(common)[:max_images])

        for rel in sorted(common):
            votes = [per_model[m][rel] for m in models]
            true = votes[0][0]          # dataset label, identical across models
            preds_list = [v[1] for v in votes]
            confs = [v[2] for v in votes]
            n_wrong = sum(1 for p in preds_list if p != true)
            vote_pred = Counter(preds_list).most_common(1)[0][0]
            path = os.path.join(root, rel)
            q = image_quality(path) if os.path.exists(path) else None
            lrel = to_rel(rel)
            src, lconf = labels.get(lrel, ("unknown", 1.0))
            row = {
                "rel": rel, "split": split, "true": true, "vote_pred": vote_pred,
                "n_models": len(models), "n_wrong": n_wrong,
                "frac_wrong": n_wrong / max(1, len(models)),
                "mean_conf": float(np.mean(confs)) if confs else 0.0,
                "max_conf": float(np.max(confs)) if confs else 0.0,
                "preds": "|".join(f"{m}={v[1]}:{v[2]:.3f}" for m, v in zip(models, votes)),
                "pred_by_model": {m: v[1] for m, v in zip(models, votes)},
                "label_source": src, "label_conf": lconf,
                "path": path,
            }
            row.update(q or {"lap_var": float("nan"), "brightness": float("nan"),
                             "over_ratio": float("nan"), "under_ratio": float("nan"),
                             "contrast": float("nan"), "entropy": float("nan")})
            samples.append(row)
    return samples, sorted(all_models), splits


def flag_samples(samples, over_thr=0.15, under_thr=90.0):
    """Attach flags. Thresholds are reported in diagnosis.md so they stay auditable."""
    vals = np.array([s["lap_var"] for s in samples if not np.isnan(s["lap_var"])])
    p10 = float(np.percentile(vals, 10)) if len(vals) >= 20 else 0.0
    p2 = float(np.percentile(vals, 2)) if len(vals) >= 20 else 0.0
    bright = np.array([s["brightness"] for s in samples if not np.isnan(s["brightness"])])
    thr = {"lap_p10": p10, "lap_p2": p2, "over": over_thr, "under": under_thr,
           "bright_p50": float(np.median(bright)) if len(bright) else 0.0,
           "bright_p5": float(np.percentile(bright, 5)) if len(bright) else 0.0,
           "over_p95": float(np.percentile([s["over_ratio"] for s in samples], 95)) if samples else 0.0}

    for s in samples:
        f = []
        if np.isnan(s["lap_var"]):
            f.append("bad_file")
        else:
            if s["lap_var"] <= p10:
                f.append("blurry")
            if s["lap_var"] <= p2:
                f.append("blurry_hard")
            if s["over_ratio"] >= over_thr:
                f.append("overexposed")
            if s["brightness"] <= under_thr:
                f.append("underexposed")
        if s["n_models"] >= 4:
            if s["frac_wrong"] >= 0.75 and s["max_conf"] >= 0.90:
                f.append("label_suspect")
            elif 0 < s["frac_wrong"] < 0.75:
                f.append("disagreement")
        elif s["n_wrong"] > 0:
            f.append("wrong")
        s["flags"] = f
        s["severity"] = round(s["n_wrong"] * s["mean_conf"], 4)
    return thr


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------
def write_error_csv(path, samples):
    cols = ["rel", "split", "true", "vote_pred", "n_models", "n_wrong", "frac_wrong",
            "mean_conf", "max_conf", "severity", "lap_var", "brightness",
            "over_ratio", "under_ratio", "contrast", "entropy",
            "label_source", "label_conf", "flags", "preds"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for s in sorted(samples, key=lambda x: -x["severity"]):
            w.writerow([s["rel"], s["split"], s["true"], s["vote_pred"], s["n_models"],
                        s["n_wrong"], round(s["frac_wrong"], 4), round(s["mean_conf"], 4),
                        round(s["max_conf"], 4), s["severity"],
                        round(s["lap_var"], 2) if not np.isnan(s["lap_var"]) else "",
                        round(s["brightness"], 1) if not np.isnan(s["brightness"]) else "",
                        round(s["over_ratio"], 4), round(s["under_ratio"], 4),
                        round(s["contrast"], 1) if not np.isnan(s["contrast"]) else "",
                        round(s["entropy"], 3) if not np.isnan(s["entropy"]) else "",
                        s["label_source"], round(s["label_conf"], 3),
                        ";".join(s["flags"]), s["preds"]])


def class_tables(samples, models):
    """per-class P/R/F1 (majority vote + per model average) and FN/FP lists."""
    trues = [s["true"] for s in samples]
    votes = [s["vote_pred"] for s in samples]
    classes = sorted(set(trues) | set(votes))

    maj = prf(trues, votes, classes)
    per_model = []
    for m in models:
        # a model may be missing from one split; only score the samples it predicted
        sub = [s for s in samples if m in s["pred_by_model"]]
        if sub:
            per_model.append(prf([s["true"] for s in sub],
                                 [s["pred_by_model"][m] for s in sub], classes))

    fn = defaultdict(list)
    fp = defaultdict(list)
    for s in samples:
        if s["true"] != s["vote_pred"]:
            fn[s["true"]].append(s)
            fp[s["vote_pred"]].append(s)

    rows = []
    for i, c in enumerate(classes):
        rows.append({
            "class": c,
            "support": int(maj["support"][i]),
            "recall": float(np.mean([m["rec"][i] for m in per_model])),
            "precision": float(np.mean([m["prec"][i] for m in per_model])),
            "f1": float(np.mean([m["f1"][i] for m in per_model])),
            "vote_recall": float(maj["rec"][i]),
            "vote_precision": float(maj["prec"][i]),
            "vote_f1": float(maj["f1"][i]),
            "fn_count": len(fn[c]), "fp_count": len(fp[c]),
            "fn": sorted(fn[c], key=lambda x: -x["mean_conf"]),
            "fp": sorted(fp[c], key=lambda x: -x["mean_conf"]),
        })
    return rows, maj, per_model, classes, fn, fp


def write_class_csv(path, rows, top):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "support", "recall", "precision", "f1",
                    "vote_recall", "vote_precision", "vote_f1",
                    "fn_count", "fp_count", "top_fn", "top_fp", "note"])
        for r in rows:
            top_fn = "; ".join(f"{s['rel']}({s['mean_conf']:.2f})" for s in r["fn"][:top])
            top_fp = "; ".join(f"{s['rel']}({s['mean_conf']:.2f})" for s in r["fp"][:top])
            note = "scarce" if r["support"] < 50 else ""
            w.writerow([r["class"], r["support"], round(r["recall"], 4),
                        round(r["precision"], 4), round(r["f1"], 4),
                        round(r["vote_recall"], 4), round(r["vote_precision"], 4),
                        round(r["vote_f1"], 4), r["fn_count"], r["fp_count"],
                        top_fn, top_fp, note])


def _md_table(head, rows):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


def write_hard_md(path, samples, rows, thr, top):
    groups = {
        "疑似标错 (多数模型都错且很自信)": [s for s in samples if "label_suspect" in s["flags"]],
        "模糊 (拉普拉斯方差处于全体最低 10%)": [s for s in samples if "blurry" in s["flags"]],
        "过曝 (饱和像素 >=15%)": [s for s in samples if "overexposed" in s["flags"]],
        "偏暗 (平均亮度 <=90)": [s for s in samples if "underexposed" in s["flags"]],
        "模型分歧 (部分对部分错)": [s for s in samples if "disagreement" in s["flags"]],
    }
    lines = ["# 困难样本清单", "",
             f"- 分析样本：{len(samples)} 张",
             f"- 模糊阈值：拉普拉斯方差 <= {thr['lap_p10']:.1f}（全体最低 10%）",
             f"- 严重模糊：<= {thr['lap_p2']:.1f}（最低 2%）", ""]
    for title, lst in groups.items():
        lines += [f"## {title}（{len(lst)} 张）", ""]
        if not lst:
            lines += ["无。", ""]
            continue
        body = []
        for s in sorted(lst, key=lambda x: -x["severity"])[:max(top, 20)]:
            body.append([s["rel"], s["true"], s["vote_pred"],
                         f"{s['mean_conf']:.2f}", f"{s['n_wrong']}/{s['n_models']}",
                         f"{s['lap_var']:.0f}" if not np.isnan(s["lap_var"]) else "-",
                         f"{s['brightness']:.0f}" if not np.isnan(s["brightness"]) else "-",
                         s["label_source"]])
        lines += _md_table(["文件", "真值", "多数票", "平均置信度", "错/模型数",
                            "锐度", "亮度", "标签来源"], body)
        lines.append("")
    for r in rows:
        lines += [f"## 类别 {r['class']}（support {r['support']}，R {r['recall']:.3f}，P {r['precision']:.3f}）", ""]
        if r["fn"]:
            lines += ["> 拖累 Recall 的图（本类被判成别类）", ""]
            lines += _md_table(["文件", "被判成", "置信度", "标签来源"],
                               [[s["rel"], s["vote_pred"], f"{s['mean_conf']:.2f}", s["label_source"]]
                                for s in r["fn"][:top]])
            lines.append("")
        if r["fp"]:
            lines += ["> 拖累 Precision 的图（别类被判成本类）", ""]
            lines += _md_table(["文件", "真实类", "置信度", "标签来源"],
                               [[s["rel"], s["true"], f"{s['mean_conf']:.2f}", s["label_source"]]
                                for s in r["fp"][:top]])
            lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


HTML_HEAD = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>困难样本复核</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f6f7f9;margin:0;padding:22px;color:#222}
h1{font-size:20px;margin:0 0 4px} .sub{color:#666;font-size:13px;margin-bottom:16px}
h2{font-size:15px;margin:26px 0 10px;padding:6px 10px;background:#eef1f5;border-left:4px solid #4a7fd4;border-radius:3px}
.grid{display:flex;flex-wrap:wrap;gap:12px}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:8px;width:188px}
.card img{width:100%;border-radius:5px;display:block}
.tag{margin-top:6px;font-size:12px;line-height:1.5}
.t{color:#1a7f37;font-weight:600}.p{color:#c62828;font-weight:600}.c{color:#888}
.warn{display:inline-block;background:#fff3cd;color:#8a6d3b;border-radius:3px;padding:0 5px;font-size:11px}
</style></head><body>
<h1>困难样本复核</h1><div class="sub">__SUB__</div>
__BODY__
</body></html>
"""


def write_html(path, samples, rows, top, data_dir):
    def card(s):
        uri = data_uri(s["path"])
        if not uri:
            return ""
        tags = "".join(f'<span class="warn">{f}</span> ' for f in s["flags"])
        return (f'<div class="card"><img src="{uri}">'
                f'<div class="tag"><span class="t">{s["true"]}</span> → '
                f'<span class="p">{s["vote_pred"]}</span><br>'
                f'<span class="c">conf {s["mean_conf"]:.2f} · '
                f'{s["n_wrong"]}/{s["n_models"]} 错 · 锐度 '
                f'{s["lap_var"]:.0f} · {s["label_source"]}</span><br>{tags}'
                f'<span class="c">{s["rel"]}</span></div></div>')

    groups = [
        ("疑似标错", [s for s in samples if "label_suspect" in s["flags"]]),
        ("模糊", [s for s in samples if "blurry" in s["flags"]]),
        ("过曝 / 偏暗", [s for s in samples if "overexposed" in s["flags"] or "underexposed" in s["flags"]]),
        ("模型分歧", [s for s in samples if "disagreement" in s["flags"]]),
    ]
    body = []
    for title, lst in groups:
        lst = sorted(lst, key=lambda x: -x["severity"])[:max(top, 24)]
        body.append(f"<h2>{title}（{len(lst)} 张，最多显示 {max(top, 24)}）</h2><div class='grid'>")
        body.append("".join(card(s) for s in lst) or "<p>无。</p>")
        body.append("</div>")
    for r in rows:
        for kind, lst in (("拖累 Recall", r["fn"]), ("拖累 Precision", r["fp"])):
            if not lst:
                continue
            body.append(f"<h2>{r['class']} · {kind}（{len(lst)} 张）</h2><div class='grid'>")
            body.append("".join(card(s) for s in lst[:top]))
            body.append("</div>")
    sub = f"分析 {len(samples)} 张 · 类别 {len(rows)} 个 · 每类最多显示 {top} 张"
    html = HTML_HEAD.replace("__SUB__", sub).replace("__BODY__", "\n".join(body))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


def write_diagnosis(path, samples, rows, models, thr, maj, splits):
    n = len(samples)
    acc = float(np.mean([s["true"] == s["vote_pred"] for s in samples])) if n else 0.0
    wrong = [s for s in samples if s["true"] != s["vote_pred"]]

    # error sources by (true -> vote) pair
    pairs = Counter((s["true"], s["vote_pred"]) for s in wrong)
    # error sources by flag
    flag_cnt = Counter()
    for s in samples:
        for f in s["flags"]:
            if f in ("label_suspect", "blurry", "overexposed", "underexposed", "disagreement"):
                flag_cnt[f] += 1

    # label-source quality
    src_stat = defaultdict(lambda: [0, 0])
    for s in samples:
        src_stat[s["label_source"]][1] += 1
        if s["true"] == s["vote_pred"]:
            src_stat[s["label_source"]][0] += 1

    hard = ("label_suspect", "blurry_hard", "overexposed", "underexposed")
    bad = [s for s in samples if any(f in hard for f in s["flags"])]
    n_bad = len(bad)
    acc_clean = (float(np.mean([s["true"] == s["vote_pred"] for s in samples
                                if not any(f in hard for f in s["flags"])]))
                 if n - n_bad else 0.0)
    ceiling = (n - n_bad + n_bad) / n if n else 0.0   # all bad ones fixed
    feasible = ceiling >= 0.97 and acc_clean >= 0.94

    lines = ["# 误差诊断报告", "",
             f"- 分析样本：**{n}** 张（split: {', '.join(splits)}）",
             f"- 模型数：**{len(models)}**（{', '.join(models)}）",
             f"- 多数票准确率：**{acc:.4f}**（错 {len(wrong)} 张）",
             f"- 模糊阈值：拉普拉斯方差 <= {thr['lap_p10']:.1f}（最低 10%）；严重 <= {thr['lap_p2']:.1f}（最低 2%）",
             ""]

    lines += ["## 1. 主要误差来源", ""]
    if pairs:
        rows_md = [[f"{a} → {b}", c, f"{c / len(wrong) * 100:.1f}%"] for (a, b), c in pairs.most_common(10)]
        lines += _md_table(["真实类 → 判成类", "张数", "占全部错误"], rows_md)
        lines.append("")
    else:
        lines += ["没有错误样本。", ""]

    lines += ["## 2. 样本属性统计", "",
              "| 属性 | 张数 | 占全部 | 占错误 |", "|---|---:|---:|---:|"]
    for f in ("label_suspect", "blurry", "overexposed", "underexposed", "disagreement"):
        c = flag_cnt[f]
        e = sum(1 for s in wrong if f in s["flags"])
        lines.append(f"| {f} | {c} | {c / max(1, n) * 100:.1f}% | {e / max(1, len(wrong)) * 100:.1f}% |")
    lines.append("")

    lines += ["## 3. 标签来源检验", "",
              "| 标签来源 | 张数 | 准确率 |", "|---|---:|---:|"]
    for src, (ok, tot) in sorted(src_stat.items(), key=lambda x: -x[1][1]):
        lines.append(f"| {src} | {tot} | {ok / max(1, tot):.4f} |")
    if "model" in src_stat and "human" in src_stat:
        m_ok, m_tot = src_stat["model"]
        h_ok, h_tot = src_stat["human"]
        gap = (h_ok / max(1, h_tot)) - (m_ok / max(1, m_tot))
        lines.append("")
        lines.append(f"> 人工标注准确率比模型标注高 **{gap * 100:+.1f} 个百分点**。"
                     + ("模型标注值得保留。" if gap < 0.03 else
                        "模型标注部分噪声偏大，建议复核 `_review.csv`。"))
    lines.append("")

    lines += ["## 4. 每类指标（多模型平均）", "",
              "| 类别 | support | Recall | Precision | F1 | FN | FP | 状态 |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda x: x["recall"]):
        note = "⚠ 样本不足" if r["support"] < 50 else ""
        lines.append(f"| {r['class']} | {r['support']} | {r['recall']:.3f} | "
                     f"{r['precision']:.3f} | {r['f1']:.3f} | {r['fn_count']} | "
                     f"{r['fp_count']} | {note} |")
    lines.append("")
    worst = [r for r in sorted(rows, key=lambda x: x["recall"])[:3]]
    lines.append("> 最差三类：" + "、".join(f"**{r['class']}**（R {r['recall']:.3f}）" for r in worst))
    lines.append("")

    lines += ["## 5. 97% 可行性", "",
              f"- 当前准确率：**{acc:.4f}**",
              f"- 疑似硬伤样本（标错/严重模糊/曝光异常）：**{n_bad}** 张（{n_bad / max(1, n) * 100:.1f}%）",
              f"- 剔除这些图后的准确率：**{acc_clean:.4f}**",
              f"- 假设这些图全部修好后的上限：**{ceiling:.4f}**",
              "",
              ("> ✅ **97% 可行**：先修上面这批图，再叠加增强/预训练。"
               if feasible else
               "> ❌ **当前数据不可达 97%**：硬伤样本过多或清洗后准确率仍不足。"
               "先把第 1、2 节的样本处理掉，再谈模型。"),
              ""]

    lines += ["## 6. 下一步（对应路线图）", ""]
    acts = []
    if flag_cnt["label_suspect"] >= 5:
        acts.append(f"- **修标签**：{flag_cnt['label_suspect']} 张疑似标错，见 `review_hard.html` 第一组")
    if flag_cnt["overexposed"] + flag_cnt["underexposed"] >= 5:
        acts.append(f"- **阶段 3 优先级提升**：{flag_cnt['overexposed'] + flag_cnt['underexposed']} 张曝光异常 → "
                    "直方图/gamma 增强排在最前")
    if flag_cnt["blurry"] >= 5:
        acts.append(f"- **模糊样本**：{flag_cnt['blurry']} 张 → 评估是剔除还是补拍")
    scarce = [r["class"] for r in rows if r["support"] < 50]
    if scarce:
        acts.append(f"- **阶段 4 离线增强**：{'、'.join(scarce)} 样本不足 50 张")
    top_pair = pairs.most_common(1)[0] if pairs else None
    if top_pair and top_pair[1] / max(1, len(wrong)) >= 0.25:
        (a, b), c = top_pair
        acts.append(f"- **类间混淆**：{a} ↔ {b} 占全部错误 {c / len(wrong) * 100:.0f}% → "
                    "考虑补数据、合并类或加注意力（阶段 6）")
    if not acts:
        acts.append("- 错误分布均匀，直接进入阶段 1（预训练对比）")
    lines += acts + [""]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"acc": acc, "n_bad": n_bad, "acc_clean": acc_clean,
            "ceiling": ceiling, "feasible": feasible, "pairs": pairs.most_common(3),
            "flags": flag_cnt, "src": dict(src_stat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="dir with pred_*.csv (and pred_*_val.csv)")
    ap.add_argument("--data", required=True, help="split root, e.g. data/rightpad_v2_split")
    ap.add_argument("--labels", default=None, help="dataset _labels.csv (label source)")
    ap.add_argument("--out", default=None, help="default <runs>/diagnosis")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--top", type=int, default=10, help="images shown per class")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--over-thr", type=float, default=0.15,
                    help="saturated-pixel ratio (>=250) above which an image counts as overexposed")
    ap.add_argument("--under-thr", type=float, default=90.0,
                    help="mean brightness below which an image counts as too dark")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    runs = os.path.abspath(args.runs)
    data = os.path.abspath(args.data)
    out = os.path.abspath(args.out or os.path.join(runs, "diagnosis"))
    os.makedirs(out, exist_ok=True)

    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    labels = load_labels(args.labels)
    print(f"runs  : {runs}\ndata  : {data}\nlabels: {len(labels)} 条")

    samples, models, splits = build_samples(runs, data, labels, wanted, args.max_images)
    if not samples:
        sys.exit("no samples to analyse")
    print(f"样本 {len(samples)} 张 / 模型 {len(models)} 个 / split {splits}")

    thr = flag_samples(samples, args.over_thr, args.under_thr)
    rows, maj, per_model, classes, fn, fp = class_tables(samples, models)

    write_error_csv(os.path.join(out, "error_samples.csv"), samples)
    write_class_csv(os.path.join(out, "class_impact.csv"), rows, args.top)
    write_hard_md(os.path.join(out, "hard_samples.md"), samples, rows, thr, args.top)
    if not args.no_html:
        write_html(os.path.join(out, "review_hard.html"), samples, rows, args.top, data)
    summary = write_diagnosis(os.path.join(out, "diagnosis.md"), samples, rows, models, thr, maj, splits)

    print("\n" + "=" * 62)
    print(f"多数票准确率 {summary['acc']:.4f}  硬伤样本 {summary['n_bad']} 张  "
          f"清洗后 {summary['acc_clean']:.4f}  上限 {summary['ceiling']:.4f}")
    print(f"97% 可行性: {'是' if summary['feasible'] else '否'}")
    print("主要混淆:", ", ".join(f"{a}->{b}({c})" for (a, b), c in summary["pairs"]) or "无")
    print("标记统计:", dict(summary["flags"]))
    print(f"\n输出目录: {out}")
    for f in ("error_samples.csv", "class_impact.csv", "hard_samples.md",
              "review_hard.html", "diagnosis.md"):
        p = os.path.join(out, f)
        if os.path.exists(p):
            print(f"  {f:<22}{os.path.getsize(p) / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
