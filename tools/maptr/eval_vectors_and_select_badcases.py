#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import json
import argparse
import shutil
import math
import csv
import numpy as np

# 依赖 SciPy：KDTree + 匈牙利匹配
try:
    from scipy.spatial import cKDTree
    from scipy.optimize import linear_sum_assignment

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def resample_polyline(pts, K=64):
    """将折线按弧长均匀重采样到 K 点，稳定 Chamfer 计算。"""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, K, axis=0)

    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    total = L[-1]
    if total < 1e-6:
        return np.repeat(pts[:1], K, axis=0)

    t = np.linspace(0, total, K, dtype=np.float32)
    out = np.zeros((K, 2), dtype=np.float32)
    j = 0
    for i, ti in enumerate(t):
        while j < len(L) - 2 and L[j + 1] < ti:
            j += 1
        denom = (L[j + 1] - L[j]) if (L[j + 1] - L[j]) > 1e-6 else 1.0
        a = (ti - L[j]) / denom
        out[i] = (1 - a) * pts[j] + a * pts[j + 1]
    return out


def chamfer_distance(A, B, use_kdtree=True):
    """双向 Chamfer（平均）：mean(d(A->B)) + mean(d(B->A)) / 2"""
    if A.size == 0 or B.size == 0:
        return float("inf"), float("inf")
    if use_kdtree and SCIPY_OK:
        da = cKDTree(B).query(A, k=1)[0].astype(np.float32)
        db = cKDTree(A).query(B, k=1)[0].astype(np.float32)
    else:
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        da = np.sqrt(d2.min(axis=1))
        db = np.sqrt(d2.min(axis=0))
    return 0.5 * (da.mean() + db.mean()), float(
        np.percentile(np.concatenate([da, db]), 95)
    )


def match_instances(gt_list, pred_list, K=64, gate=1.5):
    """按类别分别做 1-to-1 Hungarian 匹配；代价=重采样后 Chamfer。"""
    from collections import defaultdict

    g_by_c = defaultdict(list)
    p_by_c = defaultdict(list)
    for g in gt_list:
        g_by_c[g["cls"]].append(g)
    for p in pred_list:
        p_by_c[p["cls"]].append(p)

    all_pairs = []
    FP = 0
    FN = 0
    per_class = {}

    for cls in sorted(set(list(g_by_c.keys()) + list(p_by_c.keys()))):
        G = g_by_c.get(cls, [])
        P = p_by_c.get(cls, [])
        nG, nP = len(G), len(P)
        if nG == 0 and nP == 0:
            continue
        if nG == 0:
            FP += nP
            per_class[cls] = dict(
                matches=0, FP=nP, FN=0, chamfer_mean=np.nan, chamfer95=np.nan
            )
            continue
        if nP == 0:
            FN += nG
            per_class[cls] = dict(
                matches=0, FP=0, FN=nG, chamfer_mean=np.nan, chamfer95=np.nan
            )
            continue

        Gs = [resample_polyline(g["polyline"], K=K) for g in G]
        Ps = [resample_polyline(p["polyline"], K=K) for p in P]

        C = np.zeros((nG, nP), dtype=np.float32)
        C95 = np.zeros_like(C)
        for i in range(nG):
            for j in range(nP):
                c, c95 = chamfer_distance(Gs[i], Ps[j])
                C[i, j] = c
                C95[i, j] = c95

        gi, pj = linear_sum_assignment(C)
        matches = []
        cls_c_list = []
        cls_c95_list = []
        matched_pred = set()
        matched_gt = set()
        for a, b in zip(gi, pj):
            c = C[a, b]
            if c <= gate:
                matches.append((a, b))
                matched_pred.add(b)
                matched_gt.add(a)
                cls_c_list.append(c)
                cls_c95_list.append(C95[a, b])

        FP_cls = nP - len(matched_pred)
        FN_cls = nG - len(matched_gt)
        FP += FP_cls
        FN += FN_cls
        for a, b in matches:
            all_pairs.append(
                dict(
                    cls=cls,
                    gt_idx=a,
                    pred_idx=b,
                    chamfer=C[a, b],
                    chamfer95=C95[a, b],
                )
            )

        cm = float(np.mean(cls_c_list)) if cls_c_list else np.nan
        c95m = float(np.mean(cls_c95_list)) if cls_c95_list else np.nan
        per_class[cls] = dict(
            matches=len(matches),
            FP=FP_cls,
            FN=FN_cls,
            chamfer_mean=cm,
            chamfer95=c95m,
        )

    return all_pairs, FP, FN, per_class


def evaluate_one_sample(json_path, K=64, gate=1.5):
    with open(json_path, "r") as f:
        J = json.load(f)
    gt = J.get("gt", [])
    pred = J.get("pred", [])

    pairs, FP, FN, per_class = match_instances(gt, pred, K=K, gate=gate)
    n_pred = len(pred)
    n_gt = len(gt)
    n_match = len(pairs)
    prec = n_match / max(n_pred, 1)
    rec = n_match / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)

    if pairs:
        chamfers = np.array([p["chamfer"] for p in pairs], dtype=np.float32)
        chamfer95 = np.array([p["chamfer95"] for p in pairs], dtype=np.float32)
        c_mean = float(chamfers.mean())
        c95_mean = float(chamfer95.mean())
    else:
        c_mean = float("nan")
        c95_mean = float("nan")

    fp_ratio = FP / max(n_pred, 1)
    fn_ratio = FN / max(n_gt, 1)
    badness = (
        0.7 * (c_mean if math.isfinite(c_mean) else gate * 1.5)
        + 0.2 * fn_ratio
        + 0.1 * fp_ratio
    )

    return dict(
        sample=J.get("sample_token", osp.basename(osp.dirname(json_path))),
        json_path=json_path,
        n_pred=n_pred,
        n_gt=n_gt,
        matches=n_match,
        FP=FP,
        FN=FN,
        precision=prec,
        recall=rec,
        f1=f1,
        chamfer_mean=c_mean,
        chamfer95_mean=c95_mean,
        badness=badness,
        per_class=per_class,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "vis_dir",
        help="你的 vis_pred 输出目录（包含若干子目录，每个子目录下有 vectors.json）",
    )
    ap.add_argument("--K", type=int, default=64, help="重采样点数")
    ap.add_argument(
        "--gate",
        type=float,
        default=1.5,
        help="Hungarian 匹配门限（米）",
    )
    ap.add_argument("--topk", type=int, default=200, help="复制最差的前 K 个样本")
    ap.add_argument("--out_csv", default="vis_metrics.csv")
    ap.add_argument("--bad_dir", default="")
    ap.add_argument(
        "--copy_what",
        nargs="+",
        default=[
            "vectors.json",
            "PRED_MAP_plot.png",
            "GT_fixednum_pts_MAP.png",
            "GT_polyline_pts_MAP.png",
            "surroud_view.jpg",
            "SAMPLE_VIS.png",
        ],
    )
    args = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(args.vis_dir)):
        d = osp.join(args.vis_dir, name)
        if not osp.isdir(d):
            continue
        jp = osp.join(d, "vectors.json")
        if not osp.exists(jp):
            continue
        try:
            r = evaluate_one_sample(jp, K=args.K, gate=args.gate)
            r_flat = {k: v for k, v in r.items() if k != "per_class"}
            for cls, pc in r["per_class"].items():
                for kk, vv in pc.items():
                    r_flat[f"c{cls}_{kk}"] = vv
            rows.append(r_flat)
        except Exception as e:
            print(f"[WARN] failed on {jp}: {e}")

    rows.sort(key=lambda x: x.get("badness", 0.0), reverse=True)

    out_csv = osp.join(args.vis_dir, args.out_csv)
    if rows:
        keys = sorted(set().union(*[r.keys() for r in rows]))
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[OK] CSV saved to: {out_csv}")
    else:
        print("[WARN] no rows to write")

    if args.bad_dir:
        os.makedirs(args.bad_dir, exist_ok=True)
        for r in rows[: args.topk]:
            src = osp.dirname(r["json_path"])
            dst = osp.join(args.bad_dir, osp.basename(src))
            if osp.exists(dst):
                shutil.rmtree(dst)
            os.makedirs(dst, exist_ok=True)
            for b in args.copy_what:
                p = osp.join(src, b)
                if osp.exists(p):
                    try:
                        shutil.copy2(p, osp.join(dst, osp.basename(p)))
                    except Exception:
                        pass
        msg = f"[OK] Copied worst {min(args.topk, len(rows))} to: {args.bad_dir}"  # noqa: E501
        print(msg)


if __name__ == "__main__":
    main()
