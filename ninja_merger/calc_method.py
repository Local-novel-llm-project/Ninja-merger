# Copyright (c) 2024 TylorShine
#                    UmiseTokikaze
#                    Exveria
# license: Apache-2.0

import gc
import torch

import numpy as np

# try:
#     from torch_linear_assignment import (
#         batch_linear_assignment as linear_sum_assignment,
#         assignment_to_indices
#     )
#     is_torch_linear_assignment_found = True
# except ImportError:
#     print("WARN: torch_linear_assignment not found, using scipy version instead")
from scipy.optimize import linear_sum_assignment
    # is_torch_linear_assignment_found = False


def Add(v, base_state_dict, sub_state_dict, velocity):
    return ((base_state_dict + sub_state_dict).to(v) * velocity)

def Sub(v, base_state_dict, sub_state_dict, velocity):
    return ((base_state_dict - sub_state_dict).to(v) * velocity)

def Mul(v, base_state_dict, sub_state_dict, velocity):
    return ((base_state_dict * sub_state_dict).to(v) * velocity)

def Div(v, base_state_dict, sub_state_dict, velocity):
    return ((base_state_dict / sub_state_dict).to(v) * velocity)

def Mix(v, base_state_dict, sub_state_dict, velocity):
    return (base_state_dict*(1.0 - velocity) + sub_state_dict*velocity).to(v)

def Avg(v, base_state_dict, sub_state_dict, base_weight, sub_weight, velocity):
    return (base_state_dict * base_weight + sub_state_dict * sub_weight) / (base_weight + sub_weight)

def Concatenation(v, base_state_dict, sub_state_dict, velocity):
    return torch.cat((base_state_dict, sub_state_dict), dim=0)

def MaxPool(v, base_state_dict, sub_state_dict,velocity):
    return torch.max(base_state_dict, sub_state_dict)

def MinPool(v, base_state_dict, sub_state_dict,velocity):
    return torch.min(base_state_dict, sub_state_dict)

def GeometricMean(v, base_state_dict, sub_state_dict,velocity):
    return torch.sqrt(base_state_dict * sub_state_dict)

def StdSub(v, base_state_dict, sub_state_dict, velocity):
    eps = 1e-7
    base_std, base_mean = torch.std_mean(base_state_dict)
    sub_std, sub_mean = torch.std_mean(sub_state_dict)
    return (((base_state_dict - base_mean) / max(base_std, eps)) - ((sub_state_dict - sub_mean) / max(sub_std, eps))).to(v) * velocity


def PostAdd(v, processed_value, velocity):
    return v + processed_value*velocity

def PostSub(v, processed_value, velocity):
    return v - processed_value*velocity

def PostSubFrom(v, processed_value, velocity):
    return processed_value - v*velocity

def PostMul(v, processed_value, velocity):
    return v * processed_value*velocity

def PostDiv(v, processed_value, velocity):
    return v / processed_value*velocity

def PostDivBy(v, processed_value, velocity):
    return processed_value*velocity / v

def PostMix(v, processed_value, velocity):
    return (v*(1.0 - velocity) + processed_value*velocity).to(v)

def PostConcatenation(v, processed_value, velocity):
    return torch.cat((v, processed_value*velocity), dim=0)

def PostMaxPool(v, processed_value, velocity):
    return torch.max(v, processed_value*velocity)

def PostMinPool(v, processed_value, velocity):
    return torch.min(v, processed_value*velocity)

def PostGeometricMean(v, processed_value, velocity):
    return torch.sqrt(v * processed_value * velocity)

def PostAngle(v, processed_value, velocity):
    return torch.acos(torch.clamp((v*processed_value).sum(dim=-1)/(torch.norm(v,dim=-1)*torch.norm(processed_value,dim=-1)),-1.0,1.0))


def Passthrough(proc_func, v1, v2s, velocity, **kwargs):
    if len(v2s) > 1:
        return torch.stack([proc_func(v1, v2, velocity) for v2 in v2s], dim=-1).sum(dim=-1, keepdim=True)
    else:
        return proc_func(v1, v2s[0], velocity)


def NormStdMean(proc_func, v1, v2s, velocity, **kwargs):
    eps = 1e-7
    processed = proc_func(v1, torch.sum(torch.stack(v2s, dim=-1), dim=-1, keepdim=True), velocity)
    std, mean = torch.std_mean(processed)
    return (processed - mean) / max(std, eps)

def MatchStdMean(proc_func, v1, v2s, velocity, **kwargs):
    eps = 1e-7
    orig_std, orig_mean = torch.std_mean(v1)
    processed = proc_func(v1, torch.sum(torch.stack(v2s, dim=-1), dim=-1, keepdim=True) , velocity)
    std, mean = torch.std_mean(processed)
    return (processed - (mean - orig_mean)) * (max(std, eps)/max(orig_std, eps))

def ProcStdMean(proc_func, v1, v2s, velocity, **kwargs):
    eps = 1e-7
    orig_std, orig_mean = torch.std_mean(v1)
    v2 = torch.sum(torch.stack(v2s, dim=-1), dim=-1, keepdim=True)
    v_std, v_mean = torch.std_mean(v2)
    processed = proc_func((v1 - orig_mean)/max(orig_std, eps), (v2 - v_mean)/max(v_std, eps), velocity)
    return processed*max(orig_std, eps) + orig_mean

def all_mul(*args):
    result = args[0]
    for arg in args[1:]:
        result = result * arg
    return result

def NormAngleMerge(proc_func, v1, v2s, velocity, **kwargs):
    # norm_prods = [
    #     torch.norm(v2, dim=-1) * torch.norm(v2s[j], dim=-1)
    #     for i, v2 in enumerate(v2s)
    #     for j in range(i+1, len(v2s))
    #     ]
    thetas = [
        ((v2 * v2s[j]).sum(dim=-1) / ((torch.norm(v2, dim=-1) * torch.norm(v2s[j], dim=-1)).clamp(min=1e-7)))
        for i, v2 in enumerate(v2s)
        for j in range(i+1, len(v2s))
        ]
    
    # norm_prod = None
    # for i in range(1, len(v2s)):
    #     if norm_prod is None:
    #         norm_prod = torch.norm(v2s[i-1], dim=-1) * torch.norm(v2s[i], dim=-1)
    #     else:
    #         norm_prod *= torch.norm(v2s[i], dim=-1)
    # norm_prod = all_mul(*[torch.norm(v, dim=-1) for v in v2s])
    # theta = (all_mul(*v2s).sum(dim=-1) / (norm_prod.clamp(min=1e-7))).unsqueeze(-1)
    theta = torch.stack(thetas, dim=-1).mean(dim=-1).unsqueeze(-1)
    t = len(v2s)*torch.cos(theta)/(1. + (len(v2s) - 1) * torch.cos(theta))
    # avg = torch.mul(torch.stack(v2s, dim=-1), t/len(v2s)).sum(dim=-1)
    # avg = torch.stack(v2s, dim=-1).sum(dim=-1) * t
    avg = sum(v2s) / len(v2s)
    # print(len(thetas), v1.shape, theta.shape, t.shape)
    processed = proc_func(v1 * (1. - t), avg * t, velocity)
    return processed


def GitReBasin(v1, v2s, preprocess_velocity=None, **kwargs):
    # v1: ベースモデルのパラメータテンソル
    # v2s: マージしたいモデルのパラメータテンソルのリスト
    # velocity: 補間係数

    # v2 = v2s[0]  # 複数のモデルがある場合への対応（ここでは1つのみ対応）
    is_not_list = False
    if not isinstance(v2s, list):
        is_not_list = True
        v2s = [v2s]

    # layer_name = kwargs.get("layer_name", None)
    # if layer_name is None:
    #     raise ValueError("layer_nameをkwargsで指定してください。")

    # permutation_dict = kwargs.get("permutation_dict", {})

    print(f"===== Processing layer =====")
    print(f"v1 shape: {v1.shape}")
    # print(f"v2s[0] shape: {v2s[0].shape}")
    
    for iv2 in range(len(v2s)):
        # バイアス項や1次元テンソルの場合、パーミュテーションは不要
        if v1.dim() < 2 or v2s[iv2].dim() < 2:
            # aligned_v2 = v2
            print("This layer does not require permutation (bias or 1D tensor).")
        else:
            # if layer_name in permutation_dict:
            #     # 既に計算済みのパーミュテーションを適用
            #     col_ind = permutation_dict[layer_name]
            #     print(f"Using existing permutation for layer '{layer_name}'.")
            # else:
            
            # テンソルを2次元に変形
            v1_flat = v1.view(v1.size(0), -1)
            v2_flat = v2s[iv2].view(v2s[iv2].size(0), -1)
            print(f"v1_flat shape: {v1_flat.shape}")
            print(f"v2_flat shape: {v2_flat.shape}")

            # デバイスの設定
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # テンソルをデバイスに移動し、データ型をfloat32に変更
            v1_flat = v1_flat.to(device).to(torch.float32)
            v2_flat = v2_flat.to(device).to(torch.float32)
            print(f"v1_flat device: {v1_flat.device}")
            print(f"v2_flat device: {v2_flat.device}")

            # 正規化の計算
            norm_v1 = torch.norm(v1_flat, dim=1, keepdim=True) + 1e-8
            v1_norm = v1_flat / norm_v1

            norm_v2 = torch.norm(v2_flat, dim=1, keepdim=True) + 1e-8
            v2_norm = v2_flat / norm_v2

            # コサイン類似度の計算
            similarity = torch.matmul(v1_norm, v2_norm.t())
            # similarity = torch.cdist(v1_flat, v2_flat, p=2)

            print(f"Computed similarity matrix of shape: {similarity.shape}")
            
            del norm_v1, norm_v2

            # コスト行列を作成（類似度の負をコストとする）
            # cost_matrix = -similarity.cpu().numpy().astype(np.float32)
            # print("Converted similarity matrix to cost matrix.")

            # 線形割当問題を解く
            # row_ind, col_ind = linear_sum_assignment(cost_matrix)
            row_ind, col_ind = linear_sum_assignment(-similarity.cpu().numpy().astype(np.float32))
            # if is_torch_linear_assignment_found:
            #     row_ind, col_ind = assignment_to_indices(linear_sum_assignment(-similarity[None, :]))
            # else:
            #     row_ind, col_ind = linear_sum_assignment(-similarity)
            # permutation_dict[layer_name] = col_ind
            # print(f"Calculated new permutation for layer '{layer_name}'.")

            # v2 をパーミュテーション
            # aligned_v2 = v2[col_ind]
            v2s[iv2] = v2s[iv2][col_ind]
            # print(f"Applied permutation to v2 for layer '{layer_name}'.")
            
            # del similarity, cost_matrix, row_ind, col_ind
            # gc.collect()

    # # ベースモデルとパーミューテーション後のモデルを線形補間
    # result = (1 - velocity) * v1 + velocity * aligned_v2
    # print(f"Interpolated parameters for layer '{layer_name}'.\n")
    
    if is_not_list:
        return v1, v2s[0]

    return v1, v2s