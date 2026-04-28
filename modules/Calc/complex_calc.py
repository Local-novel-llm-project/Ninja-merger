# complex_calc.py
import torch
from rich import print


def _complex_real_dtype(dtype):
    if dtype in (torch.float16, torch.float32, torch.float64):
        return dtype
    return torch.float32


def _to_complex_tensor(val, device, dtype):
    """
    数値または複素数を複素数テンソルに変換するヘルパー関数
    """
    real_dtype = _complex_real_dtype(dtype)
    if isinstance(val, complex):
        return torch.complex(
            torch.tensor(val.real, device=device, dtype=real_dtype),
            torch.tensor(val.imag, device=device, dtype=real_dtype),
        )
    else:
        return torch.complex(
            torch.tensor(float(val), device=device, dtype=real_dtype),
            torch.tensor(0.0, device=device, dtype=real_dtype),
        )


def ComplexAdd(v1, avg, t, velocity):
    """
    複素数の velocity を用いた加算。
    """
    # テストの期待値に合わせて計算式を修正
    return v1 + (avg - v1) * t * velocity


def ComplexMix(v1, v2, t, velocity):
    """
    複素数の velocity と 複素数の t を用いた線形結合。
    """
    # t, velocity が数値の場合、テンソルに変換
    if not isinstance(t, torch.Tensor):
        device = v1.device if hasattr(v1, "device") else torch.device("cpu")
        dtype = torch.float32  # または v1.dtype
        t = _to_complex_tensor(t, device, dtype)

    if not isinstance(velocity, torch.Tensor):
        device = v1.device if hasattr(v1, "device") else torch.device("cpu")
        dtype = torch.float32
        velocity = _to_complex_tensor(velocity, device, dtype)

    # v1, v2の計算
    v1_term = (
        v1
        * (1.0 - torch.abs(t) * torch.abs(velocity))
        * torch.exp(-1j * torch.angle(t) - 1j * torch.angle(velocity))
    )
    v2_term = (
        v2
        * torch.abs(t)
        * torch.abs(velocity)
        * torch.exp(1j * torch.angle(t) * 1j * torch.angle(velocity))
    )

    return v1_term + v2_term


def ComplexAngleNorm(
    proc_func,
    v1,
    v2s,
    velocity,
    *,
    force_merge_single: bool = False,
    v2s_empty_default: str = "v1",
    v2s_single_default: str = "auto",
    **kwargs,
):
    """
    proc_funcに対応した NormAngleMerge。
    """

    # v1, v2s が複素数の場合、内積、ノルム、角度の計算を修正
    def dot_product(x, y):
        return torch.sum(x * y.conj(), dim=-1)

    def norm(x):
        return torch.linalg.norm(x, dim=-1)

    # v2sが空の場合
    if not v2s:
        if v2s_empty_default == "v1":
            return v1
        elif v2s_empty_default == "zero":
            return torch.zeros_like(v1)  # ゼロテンソル
        else:  # none
            return None

    if len(v2s) == 1:
        if v2s_single_default == "auto":
            thetas = [
                (dot_product(v1, v2) / (norm(v1) * norm(v2)).clamp(min=1e-7))
                for v2 in v2s
            ]
            theta = torch.stack(thetas, dim=-1).mean(dim=-1, keepdim=True)
            t = (
                1
                * torch.cos(torch.angle(theta))
                / (1.0 + (1 - 1) * torch.cos(torch.angle(theta)))
            )  # 角度だけ利用
            t = _to_complex_tensor(t, v1.device, v1.dtype)

            avg = sum(v2s) / len(v2s)
            processed = proc_func(v1, avg, t, velocity)
            return processed

        elif v2s_single_default == "v1":
            return v1
        elif v2s_single_default == "v2":
            return v2s[0]
        elif v2s_single_default == "average":
            return (v1 + v2s[0]) / 2  # 平均で
        else:  # none
            return None

    # 通常処理
    thetas = [
        (dot_product(v2, v2s[j]) / (norm(v2) * norm(v2s[j])).clamp(min=1e-7))
        for i, v2 in enumerate(v2s)
        for j in range(i + 1, len(v2s))
    ]
    theta = torch.stack(thetas, dim=-1).mean(dim=-1, keepdim=True)
    t = (
        len(v2s)
        * torch.exp(1j * torch.angle(theta))
        / (1.0 + (len(v2s) - 1) * torch.exp(1j * torch.angle(theta)))
    )  # 角度だけ利用

    avg = sum(v2s) / len(v2s)
    processed = proc_func(v1, avg, t, velocity)

    return processed


def ComplexAngleMerge(v1, v2s, velocity, *, complex_mix_func, t_calc_func, **kwargs):
    """
    複素数の velocity と、柔軟な t 計算、および混合関数を用いた Complex Angle Merge。
    t も複素数として扱う。
    """
    # t を計算 (複素数)
    t = t_calc_func(v1, v2s, velocity, **kwargs)

    # v2sの平均
    avg = sum(v2s) / len(v2s)

    # 混合
    processed = complex_mix_func(v1, avg, t, velocity)
    return processed


def norm_angle_t_calc(v1, v2s, velocity, **kwargs):
    """
    NormAngleMerge の t 計算 (複素数対応)。
    theta を実数に戻さない。
    """

    def dot_product(x, y):
        return torch.sum(x * y.conj(), dim=-1)

    def norm(x):
        return torch.linalg.norm(x, dim=-1)

    if not v2s:
        print("Warning: v2s is empty in norm_angle_t_calc. Skipping this layer.")
        return torch.complex(
            torch.tensor(0.0, device=v1.device, dtype=torch.float32),  # float32に
            torch.tensor(0.0, device=v1.device, dtype=torch.float32),  # float32に
        )
    if len(v2s) == 1:
        # v2s_single_defaultの値で分岐
        v2s_single_default = kwargs.get("v2s_single_default", "auto")
        if v2s_single_default == "auto":
            thetas = [
                (dot_product(v1, v2) / (norm(v1) * norm(v2)).clamp(min=1e-7))
                for v2 in v2s
            ]
            theta = torch.stack(thetas, dim=-1).mean(dim=-1, keepdim=True)
            t = (
                1
                * torch.cos(torch.angle(theta))
                / (1.0 + (1 - 1) * torch.cos(torch.angle(theta)))
            )  # len(v2s) -> 1
        elif v2s_single_default == "v1":
            t = 0.0
        elif v2s_single_default == "v2":
            t = 1.0
        elif v2s_single_default == "average":
            t = 0.5
        else:
            t = 0.0

        return _to_complex_tensor(t, v1.device, torch.float32)  # float32に

    thetas = [
        (dot_product(v2, v2s[j]) / (norm(v2) * norm(v2s[j])).clamp(min=1e-7))
        for i, v2 in enumerate(v2s)
        for j in range(i + 1, len(v2s))
    ]
    theta = torch.stack(thetas, dim=-1).mean(dim=-1, keepdim=True)
    t = (
        len(v2s)
        * torch.exp(1j * torch.angle(theta))
        / (  # 角度だけ利用
            1.0 + (len(v2s) - 1) * torch.exp(1j * torch.angle(theta))
        )
    )  # 角度だけ利用
    return t


def calculate_average_angle(base_model, sub_model):
    angles = []
    base_state_dict = base_model.state_dict()
    sub_state_dict = sub_model.state_dict()

    for key in base_state_dict:
        if key in sub_state_dict:
            diff = sub_state_dict[key] - base_state_dict[key]

            # 実数テンソルの場合は、複素数テンソルに変換 (float32に変換)
            if diff.dtype in [
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ]:
                diff = diff.to(torch.float32)  # float32 に変換
                # 複素数対応: 実数の場合のみ、微小な虚部を追加
                diff = torch.complex(
                    diff,
                    torch.randn(diff.shape, device=diff.device, dtype=torch.float32)
                    * 1e-7,
                )  # 微小な虚部を追加

            angle = torch.angle(diff)
            angles.append(angle.mean())

    if not angles:
        print(
            "Warning: No common keys found between base and sub models, or all diffs are zero."
        )
        return 0.0

    # ベクトルとして平均
    angles = torch.stack(angles)
    # 複素数対応: 複素数の平均角度を計算
    avg_angle = torch.angle(torch.exp(1j * angles).mean()).item()

    return avg_angle
