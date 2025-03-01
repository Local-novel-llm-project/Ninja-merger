import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def Passthrough(proc_func, v1, v2s, velocity, **kwargs):
    if not isinstance(v2s, list):
        raise TypeError(
            "v2s must be a list of tensors."
        )  # v2s がリストでない場合は例外を発生

    if len(v2s) > 1:
        return torch.stack([proc_func(v1, v2, velocity) for v2 in v2s], dim=-1).sum(
            dim=-1, keepdim=True
        )
    elif len(v2s) == 1:  # v2s が空リストでなく、要素が1つだけの場合の処理
        return proc_func(v1, v2s[0], velocity)
    else:  # v2sが空の場合
        return v1  # v1をそのまま返す（v2sが空の場合のデフォルトの動作


def GitReBasin(v1, v2s, preprocess_velocity=None, **kwargs):
    # v1: ベースモデルのパラメータテンソル
    # v2s: マージしたいモデルのパラメータテンソルのリスト
    # velocity: 補間係数

    # v2 = v2s[0]  # 複数のモデルがある場合への対応（ここでは1つのみ対応）
    is_not_list = False
    if not isinstance(v2s, list):
        is_not_list = True
        v2s = [v2s]

    print("===== Processing layer =====")
    print(f"v1 shape: {v1.shape}")

    for iv2 in range(len(v2s)):
        # バイアス項や1次元テンソルの場合、パーミュテーションは不要
        if v1.dim() < 2 or v2s[iv2].dim() < 2:
            # aligned_v2 = v2
            print("This layer does not require permutation (bias or 1D tensor).")
        else:
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

            # コスト行列を作成（類似度の負をコストとする）し、線形割当問題を解く
            row_ind, col_ind = linear_sum_assignment(
                -similarity.cpu().numpy().astype(np.float32)
            )

            # v2 をパーミュテーション
            v2s[iv2] = v2s[iv2][col_ind]

    if is_not_list:
        return v1, v2s[0]

    return v1, v2s


def QuantileMatch(proc_func, v1, v2s, velocity, *, num_quantiles=100, **kwargs):
    """
    Applies quantile matching to align the distribution of v2s to v1.

    Args:
        proc_func: The function to apply to v1 and the combined v2s.
        v1: The tensor representing the target distribution (base model).
        v2s: A list of tensors representing the source distributions (sub models).
        velocity: The velocity parameter passed to proc_func.
        num_quantiles: The number of quantiles to use for matching.

    Returns:
        The tensor with the adjusted distribution.
    """
    eps = 1e-7
    device = v1.device
    v1 = v1.detach().cpu().float()
    # Combine multiple v2s into a single tensor
    v2_combined = torch.cat(
        [v2.detach().cpu().float().flatten() for v2 in v2s]
    ).flatten()

    # Calculate quantiles for v1 and v2_combined
    # Use a smaller number of quantiles if the tensor is too large.
    print(
        f"v1.numel(): {v1.numel()}, v2_combined.numel(): {v2_combined.numel()}"
    )  # Debug print
    num_quantiles = min(
        num_quantiles, int(v1.numel() ** 0.5), int(v2_combined.numel() ** 0.5)
    )  # Example heuristic

    if num_quantiles < 2:
        return v1.to(device)  # Not enough data to perform matching.

    linspace_values = torch.linspace(0, 1, num_quantiles + 1).float()
    try:
        v1_quantiles = torch.quantile(v1, linspace_values)
        v2_quantiles = torch.quantile(v2_combined, linspace_values)
    except RuntimeError as e:
        if "input tensor is too large" in str(e):
            print(
                f"Error: Input tensor is still too large. v1.shape: {v1.shape}, v2_combined.shape: {v2_combined.shape}, num_quantiles: {num_quantiles}"
            )
            # Handle the error (e.g., skip this layer, use a different method)
            return v1.to(device)  # Fallback: Return the original tensor
        else:
            raise e  # Re-raise other RuntimeErrors

    # Apply the processing function
    # Ensure v2_combined has the same number of dimensions as v1 before reshaping
    if v2_combined.ndim != v1.ndim:
        v2_combined = v2_combined.reshape(-1, *([1] * (v1.ndim - 1)))

    try:
        processed = proc_func(v1, v2_combined.reshape(v1.shape), velocity).flatten()
    except RuntimeError as e:
        print(f"Error during processing: {e}")
        print(
            f"v1.shape: {v1.shape}, v2_combined.shape (before reshape): {v2_combined.shape}"
        )
        return v1.to(device)  # or some other appropriate fallback

    # Map processed values to v1 quantiles
    v2_sorted, v2_indices = torch.sort(v2_combined)
    v1_sorted, _ = torch.sort(v1)

    interp_values = torch.interp(
        processed,
        v2_sorted,
        torch.cat([v1_sorted, v1_sorted[[-1]]]),  # v1 quantiles mapping to v1 value
    ).reshape(v1.shape)

    return interp_values.to(device)
