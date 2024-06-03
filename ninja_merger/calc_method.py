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

def Avg(v, base_state_dict, sub_state_dict, base_weight, sub_weight,velocity):
    return (base_state_dict * base_weight + sub_state_dict * sub_weight) / (base_weight + sub_weight)

def Concatenation(v, base_state_dict, sub_state_dict,velocity):
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
    return v + processed_value

def PostSub(v, processed_value, velocity):
    return v - processed_value

def PostSubFrom(v, processed_value, velocity):
    return processed_value - v

def PostMul(v, processed_value, velocity):
    return v * processed_value

def PostDiv(v, processed_value, velocity):
    return v / processed_value

def PostDivBy(v, processed_value, velocity):
    return processed_value / v

def PostMix(v, processed_value, velocity):
    return (v*(1.0 - velocity) + processed_value*velocity).to(v)

def PostConcatenation(v, processed_value, velocity):
    return torch.cat((v, processed_value), dim=0)

def PostMaxPool(v, processed_value, velocity):
    return torch.max(v, processed_value)

def PostMinPool(v, processed_value, velocity):
    return torch.min(v, processed_value)

def PostGeometricMean(v, processed_value, velocity):
    return torch.sqrt(v * processed_value)


def Passthrough(proc_func, v1, v2, velocity, **kwargs):
    return proc_func(v1, v2, velocity)


def NormStdMean(proc_func, v1, v2, velocity, **kwargs):
    eps = 1e-7
    processed = proc_func(v1, v2, velocity)
    std, mean = torch.std_mean(processed)
    return (processed - mean) / max(std, eps)

def MatchStdMean(proc_func, v1, v2, velocity, **kwargs):
    eps = 1e-7
    orig_std, orig_mean = torch.std_mean(v1)
    processed = proc_func(v1, v2, velocity)
    std, mean = torch.std_mean(processed)
    return (processed - (mean - orig_mean)) * (max(std, eps)/max(orig_std, eps))

def ProcStdMean(proc_func, v1, v2, velocity, **kwargs):
    eps = 1e-7
    orig_std, orig_mean = torch.std_mean(v1)
    v_std, v_mean = torch.std_mean(v2)
    processed = proc_func((v1 - orig_mean)/max(orig_std, eps), (v2 - v_mean)/max(v_std, eps), velocity)
    return processed*max(orig_std, eps) + orig_mean
