# complex_calc.py
import torch
from rich import print


def ComplexAdd(v1, avg, t, velocity):
    """
    複素数の velocity を用いた加算。

    Args:
        v1: ベースモデルの重み (Tensor)。
        avg: マージ対象モデルの重みの平均 (Tensor)。
        t: v1 と avg の混合割合 (Tensor)。
        velocity: 複素数 (torch.complex64 または torch.complex128)。

    Returns:
        マージされた重み (Tensor)。
    """
    return v1 * (1.0 - t) + avg * t + velocity


def ComplexMix(v1, v2, t, velocity):
    """
    複素数の velocity と 複素数の t を用いた線形結合。
    """
    v1_term = v1 * (1.0 - t)
    v2_term = v2 * t

    # velocityをTensorに変換（もし数値型なら）
    if not isinstance(velocity, torch.Tensor):
        device = v1.device if hasattr(v1, "device") else torch.device("cpu")
        dtype = torch.float32

        if isinstance(velocity, complex):
            # 複素数の場合
            velocity = torch.complex(
                torch.tensor(velocity.real, device=device, dtype=dtype),
                torch.tensor(velocity.imag, device=device, dtype=dtype),
            )
        else:
            # 実数の場合、複素数に変換（虚部=0）
            velocity = torch.complex(
                torch.tensor(float(velocity), device=device, dtype=dtype),
                torch.tensor(0.0, device=device, dtype=dtype),
            )

    velocity_abs = torch.abs(velocity)
    velocity_angle = torch.angle(velocity)
    v2_term_rotated = v2_term * torch.exp(1j * velocity_angle)  # v2_term に位相回転
    return v1_term * (1.0 - velocity_abs) + v2_term_rotated * velocity_abs


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

    Args:
        proc_func: 処理関数 (複素数対応)。
        v1: ベースモデルの重み (実数または複素数テンソル)。
        v2s: マージ対象モデルの重みのリスト (実数または複素数テンソル)。
        velocity: 複素数 (torch.complex64 または torch.complex128)。
        **kwargs: その他の引数。

    Returns:
        マージされた重み (複素数テンソル)。
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
                1 * torch.cos(theta) / (1.0 + (1 - 1) * torch.cos(theta))
            )  # len(v2s) -> 1
            avg = sum(v2s) / len(v2s)
            processed = proc_func(v1 * (1.0 - t), avg * t, velocity)
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
    t = len(v2s) * torch.cos(theta) / (1.0 + (len(v2s) - 1) * torch.cos(theta))
    avg = sum(v2s) / len(v2s)
    processed = proc_func(v1 * (1.0 - t), avg * t, velocity)

    return processed


def ComplexAngleMerge(v1, v2s, velocity, *, complex_mix_func, t_calc_func, **kwargs):
    """
    複素数の velocity と、柔軟な t 計算、および混合関数を用いた Complex Angle Merge。
    t も複素数として扱う。

    Args:
        v1: ベースモデルの重み (実数または複素数テンソル)。
        v2s: マージ対象モデルの重みのリスト (実数または複素数テンソル)。
        velocity: 複素数 (torch.complex64 または torch.complex128)。
        complex_mix_func: 2つのテンソルと複素数 velocity, 複素数tを受け取り、混合結果を返す関数。
        t_calc_func: v1, v2s, velocity, kwargs を受け取り、t (混合割合, 複素数) を計算する関数。
        **kwargs: その他の引数 (t_calc_func に渡される)。

    Returns:
        マージされた重み (複素数テンソル)。
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
            torch.tensor(0.0, device=v1.device, dtype=v1.dtype),
            torch.tensor(0.0, device=v1.device, dtype=v1.dtype),
        )
    if len(v2s) == 1:
        print("Warning: len(v2s) == 1 in norm_angle_t_calc.  t=1+0j")
        return torch.complex(
            torch.tensor(1.0, device=v1.device, dtype=v1.dtype),
            torch.tensor(0.0, device=v1.device, dtype=v1.dtype),
        )

    thetas = [
        (dot_product(v2, v2s[j]) / (norm(v2) * norm(v2s[j])).clamp(min=1e-7))
        for i, v2 in enumerate(v2s)
        for j in range(i + 1, len(v2s))
    ]
    theta = torch.stack(thetas, dim=-1).mean(dim=-1, keepdim=True)
    # theta = torch.real(theta)  # 実数に戻さない！
    t = (
        len(v2s)
        * torch.exp(1j * theta)
        / (1.0 + (len(v2s) - 1) * torch.exp(1j * theta))
    )
    return t


def calculate_average_angle(base_model, sub_model):
    angles = []
    base_state_dict = base_model.state_dict()
    sub_state_dict = sub_model.state_dict()

    # print(f"base_model keys (first 5): {list(base_state_dict.keys())[:5]}")
    # print(f"sub_model keys (first 5): {list(sub_state_dict.keys())[:5]}")

    for key in base_state_dict:
        if key in sub_state_dict:
            diff = sub_state_dict[key] - base_state_dict[key]
            # print(f"diff for {key} (first element): {diff.flatten()[0]}")

            # 実数テンソルの場合は、複素数テンソルに変換 (float32に変換)
            if diff.dtype in [
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ]:
                diff = diff.to(torch.float32)  # float32 に変換
                diff = torch.complex(
                    diff,
                    torch.randn(diff.shape, device=diff.device, dtype=torch.float32)
                    * 1e-7,
                )  # 微小な虚部を追加

            angle = torch.angle(diff)
            # print(f"angle for {key} (first 5 elements): {angle.flatten()[:5]}")

            angles.append(angle.mean())

    if not angles:
        print(
            "Warning: No common keys found between base and sub models, or all diffs are zero."
        )
        return 0.0

    # ベクトルとして平均
    angles = torch.stack(angles)
    x = torch.cos(angles).mean()
    y = torch.sin(angles).mean()
    avg_angle = torch.atan2(y, x).item()

    return avg_angle
