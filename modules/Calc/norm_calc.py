import torch


def NormStdMean(proc_func, v1, v2s, velocity, **kwargs):
    eps = 1e-7
    processed = proc_func(
        v1, torch.sum(torch.stack(v2s, dim=-1), dim=-1, keepdim=True), velocity
    )
    std, mean = torch.std_mean(processed)
    return (processed - mean) / max(std, eps)


def MatchStdMean(proc_func, v1, v2s, velocity, **kwargs):
    eps = 1e-7
    orig_std, orig_mean = torch.std_mean(v1)
    processed = proc_func(
        v1, torch.sum(torch.stack(v2s, dim=-1), dim=-1, keepdim=True), velocity
    )
    std, mean = torch.std_mean(processed)
    return (processed - (mean - orig_mean)) * (max(std, eps) / max(orig_std, eps))


def ProcStdMean(proc_func, v1, v2s, velocity, **kwargs):
    eps = 1e-7
    orig_std, orig_mean = torch.std_mean(v1)
    v2 = torch.sum(torch.stack(v2s, dim=-1), dim=-1, keepdim=True)
    v_std, v_mean = torch.std_mean(v2)
    processed = proc_func(
        (v1 - orig_mean) / max(orig_std, eps), (v2 - v_mean) / max(v_std, eps), velocity
    )
    return processed * max(orig_std, eps) + orig_mean


def NormAngleMerge(proc_func, v1, v2s, velocity, **kwargs):
    # norm_prods = [
    #     torch.norm(v2, dim=-1) * torch.norm(v2s[j], dim=-1)
    #     for i, v2 in enumerate(v2s)
    #     for j in range(i+1, len(v2s))
    #     ]
    thetas = [
        (
            (v2 * v2s[j]).sum(dim=-1)
            / ((torch.norm(v2, dim=-1) * torch.norm(v2s[j], dim=-1)).clamp(min=1e-7))
        )
        for i, v2 in enumerate(v2s)
        for j in range(i + 1, len(v2s))
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
    t = len(v2s) * torch.cos(theta) / (1.0 + (len(v2s) - 1) * torch.cos(theta))
    # avg = torch.mul(torch.stack(v2s, dim=-1), t/len(v2s)).sum(dim=-1)
    # avg = torch.stack(v2s, dim=-1).sum(dim=-1) * t
    avg = sum(v2s) / len(v2s)
    # print(len(thetas), v1.shape, theta.shape, t.shape)
    processed = proc_func(v1 * (1.0 - t), avg * t, velocity)
    return processed
