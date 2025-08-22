#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 vectors.json（pred/gt 折线集合）评估每个样本的向量级误差，
输出 bad_cases.csv，并可复制最差样本到指定目录。
"""
import os, os.path as osp, json, math, argparse, shutil, csv
from collections import defaultdict

import numpy as np
try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def poly_length(P):  # 折线长度，用作权重
    P = np.asarray(P, dtype=float)
    if len(P) < 2:
        return 0.0
    return float(np.linalg.norm(P[1:] - P[:-1], axis=1).sum())


def chamfer(A, B):
    """ Chamfer 距离（点集到点集，双向均值）单位：米 """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if len(A) == 0 and len(B) == 0:
        return 0.0
    if len(A) == 0 or len(B) == 0:
        return 1e9  # 极差
    if HAVE_SCIPY:
        da = cKDTree(A).query(B, k=1)[0]
        db = cKDTree(B).query(A, k=1)[0]
    else:
        da = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)).min(axis=1)
        db = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(da.mean() + db.mean()) * 0.5


def chamfer_percentile(A, B, q=95):
    """ 取双向最近距离的分位数（q%，例如 95%） """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if len(A) == 0 or len(B) == 0:
        return 1e9
    if HAVE_SCIPY:
        da = cKDTree(A).query(B, k=1)[0]
        db = cKDTree(B).query(A, k=1)[0]
    else:
        da = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)).min(axis=1)
        db = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(np.percentile(np.concatenate([da, db]), q))


def cost_matrix(pred_list, gt_list, cap=5.0):
    """
    构造代价矩阵（行：GT，列：Pred），元素为 Chamfer（米）。
    cap: 对过大的代价进行截断，利于稳定匹配。
    """
    G, P = len(gt_list), len(pred_list)
    C = np.full((G, P), fill_value=cap, dtype=float)
    for i, g in enumerate(gt_list):
        gpoly = np.asarray(g['polyline'], dtype=float)
        for j, p in enumerate(pred_list):
            ppoly = np.asarray(p['polyline'], dtype=float)
            d = chamfer(gpoly, ppoly)
            C[i, j] = min(d, cap)
    return C


def hungarian_match(C, thr):
    """ 代价矩阵 C 上做匈牙利匹配，并筛去 >thr 的匹配 """
    from scipy.optimize import linear_sum_assignment
    gi, pj = linear_sum_assignment(C)  # 最小化总代价
    matches = []
    for g, p in zip(gi, pj):
        if C[g, p] <= thr:
            matches.append((g, p, C[g, p]))
    return matches


def per_sample_metrics(pred, gt, class_names=None, cost_cap=5.0, match_thr=1.0):
    """
    对一个样本，分类别做匹配并统计指标。
    - cost_cap: 计算代价时的截断（米）
    - match_thr: 认为“匹配成功”的阈值（米）
    """
    if class_names is None:
        class_names = ['__all__']
        tag = lambda x: '__all__'
    else:
        tag = lambda x: x.get('category', 'unknown')

    bucket_pred = defaultdict(list)
    bucket_gt = defaultdict(list)
    for p in pred:
        bucket_pred[tag(p)].append(p)
    for g in gt:
        bucket_gt[tag(g)].append(g)

    TP = FP = FN = 0
    all_pairs = []
    for cls in class_names:
        P = bucket_pred.get(cls, [])
        G = bucket_gt.get(cls, [])
        if len(P) == 0 and len(G) == 0:
            continue
        if len(P) == 0:
            FN += len(G)
            continue
        if len(G) == 0:
            FP += len(P)
            continue

        C = cost_matrix(P, G, cap=cost_cap)
        try:
            matches = hungarian_match(C, thr=match_thr)
        except Exception:
            used_p = set()
            matches = []
            for gi in range(C.shape[0]):
                pj = int(np.argmin(C[gi]))
                if pj not in used_p and C[gi, pj] <= match_thr:
                    matches.append((gi, pj, C[gi, pj]))
                    used_p.add(pj)

        matched_g = set(g for g, _, _ in matches)
        matched_p = set(p for _, p, _ in matches)
        TP += len(matches)
        FN += len(G) - len(matched_g)
        FP += len(P) - len(matched_p)

        for g, p, d in matches:
            gpoly = np.asarray(G[g]['polyline'], dtype=float)
            ppoly = np.asarray(P[p]['polyline'], dtype=float)
            w = max(poly_length(gpoly), poly_length(ppoly)) + 1e-6
            p95 = chamfer_percentile(gpoly, ppoly, q=95)
            all_pairs.append({'d': d, 'p95': p95, 'w': w, 'cls': cls})

    if TP == 0 and (FP > 0 or FN > 0):
        mean_c = 1e9
        p95_c = 1e9
    else:
        if all_pairs:
            wsum = sum(x['w'] for x in all_pairs)
            mean_c = sum(x['d'] * x['w'] for x in all_pairs) / max(wsum, 1e-6)
            p95_c = sum(x['p95'] * x['w'] for x in all_pairs) / max(wsum, 1e-6)
        else:
            mean_c = p95_c = 0.0

    P_inst = TP / max(TP + FP, 1e-6)
    R_inst = TP / max(TP + FN, 1e-6)
    F1_inst = 2 * P_inst * R_inst / max(P_inst + R_inst, 1e-6)

    badness = (1 - F1_inst) + min(1.0, mean_c / 1.0) + 0.5 * min(1.0, p95_c / 2.0)

    return dict(TP=TP, FP=FP, FN=FN, mean_chamfer=mean_c, p95_chamfer=p95_c,
                precision=P_inst, recall=R_inst, f1=F1_inst, badness=badness)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('visdir', help='vis_pred 根目录（包含各样本子目录）')
    ap.add_argument('--dump-name', default='vectors.json')
    ap.add_argument('--topk', type=int, default=200)
    ap.add_argument('--copy-to', default='', help='复制最差样本到此目录')
    ap.add_argument('--cost-cap', type=float, default=5.0)
    ap.add_argument('--match-thr', type=float, default=1.0)
    ap.add_argument('--classes', default='',
                    help='逗号分隔类别名；留空表示不分类别统一匹配')
    ap.add_argument('--csv', default='bad_cases.csv')
    args = ap.parse_args()

    class_names = [x.strip() for x in args.classes.split(',')] if args.classes else None

    rows = []
    for name in sorted(os.listdir(args.visdir)):
        sd = osp.join(args.visdir, name)
        if not osp.isdir(sd):
            continue
        jf = osp.join(sd, args.dump_name)
        if not osp.exists(jf):
            rows.append(dict(sample=name, badness=1e9, note='no_vectors'))
            continue
        try:
            with open(jf, 'r') as f:
                obj = json.load(f)
        except Exception:
            rows.append(dict(sample=name, badness=1e9, note='json_error'))
            continue

        pred = obj.get('pred', [])
        gt = obj.get('gt', [])
        m = per_sample_metrics(pred, gt, class_names=class_names,
                               cost_cap=args.cost_cap, match_thr=args.match_thr)
        m.update({'sample': name})
        rows.append(m)

    rows.sort(key=lambda r: r.get('badness', 1e9), reverse=True)
    worst = rows[:args.topk]

    csv_path = osp.join(args.visdir, args.csv)
    if worst:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(worst[0].keys()))
            w.writeheader()
            w.writerows(worst)
    else:
        open(csv_path, 'w').close()

    if args.copy_to:
        os.makedirs(args.copy_to, exist_ok=True)
        for r in worst:
            src = osp.join(args.visdir, r['sample'])
            dst = osp.join(args.copy_to, r['sample'])
            if osp.isdir(src):
                if osp.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

    print(f"Done. worst={len(worst)} CSV={csv_path} copied_to={args.copy_to or 'N/A'}")


if __name__ == '__main__':
    main()
