# Copyright (c) 2024 TylorShine
#                    UmiseTokikaze
#                    Exveria
# license: Apache-2.0

import torch


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