# custom_calc.py
from typing import List

import torch


def WidenMerge(
    target: torch.nn.Module,
    base_models: List[torch.nn.Module],
    sub_models: List[torch.nn.Module],
    key: str,
    t: float = 1.0,
    s: float = 1.0,
) -> torch.Tensor:
    """
    WIDEN を用いてモデルをマージする関数。

    Args:
        target: ターゲットモデル(パラメータを直接操作するため).
        base_models: ベースモデルのリスト.
        sub_models: マージするモデルのリスト.
        key: 処理対象のパラメータ名
        t: 重要度を判定する閾値 (論文中の t)。
        s: スコアキャリブレーションの係数 (論文中の s)。

    Returns:
        merged_weight: マージされた重み (torch.Tensor)。

    Raises:
        ValueError: sub_models が空の場合。
    """

    if not sub_models:
        raise ValueError("sub_models cannot be empty.")

    # base_model = base_models[0]  # 今回は、base_modelは複数与えられる可能性を考慮しない
    w_base = target.state_dict()[key].to(
        next(target.parameters()).device
    )  # デバイス統一

    # 有効なsub_modelのパラメータのみを取得
    w_subs = [
        sub_model.state_dict()[key].to(next(target.parameters()).device)
        for sub_model in sub_models
        if key in sub_model.state_dict()
        and sub_model.state_dict()[key].shape == w_base.shape
    ]

    if not w_subs:  # 有効なsub_modelが一つもない場合
        return w_base  # 処理をしない

    # 1次元パラメータと2次元以上パラメータで処理を分ける
    if w_base.ndim == 1:
        # 1次元パラメータ
        with torch.no_grad():
            delta_w = torch.stack(
                [torch.abs(w_sub - w_base) for w_sub in w_subs]
            )  # (num_subs, num_elements)
            # ランキング (0-1正規化)
            rank_w = torch.argsort(torch.argsort(delta_w, dim=1), dim=1).float() / (
                delta_w.size(1) - 1
            )  # (num_subs, num_elements)

            # Softmax
            w_scores = torch.softmax(rank_w, dim=0)  # (num_subs, num_elements)
            # スコアキャリブレーション
            w_mean = torch.mean(w_scores, dim=0, keepdim=True)
            w_scores = torch.where(
                w_scores > t * w_mean, torch.full_like(w_scores, s), w_scores
            )
            # 重みの統合
            w_merged = w_base + torch.sum(
                w_scores * (torch.stack(w_subs) - w_base), dim=0
            )

    else:
        # 2次元パラメータ
        with torch.no_grad():
            m_base = torch.norm(w_base, dim=-1, keepdim=True)
            d_base = w_base / (m_base + 1e-8)

            # shape: (num_subs, num_neurons, embedding_dim)
            m_subs = torch.stack(
                [torch.norm(w_sub, dim=-1, keepdim=True) for w_sub in w_subs]
            )
            d_subs = torch.stack(
                [
                    w_sub / (torch.norm(w_sub, dim=-1, keepdim=True) + 1e-8)
                    for w_sub in w_subs
                ]
            )

            # 2. 重みの相違の推定
            delta_m = torch.abs(m_subs - m_base)  # (num_subs, num_neurons, 1)
            delta_d = 1.0 - torch.sum(d_subs * d_base, dim=-1, keepdim=True).clamp(
                -1, 1
            )  # (num_subs, num_neurons, 1)

            # 3. 重みのランク付け (0-1 正規化)
            rank_m = torch.argsort(torch.argsort(delta_m, dim=1), dim=1).float() / (
                delta_m.size(1) - 1
            )  # (num_subs, num_neurons, 1)
            rank_d = torch.argsort(torch.argsort(delta_d, dim=1), dim=1).float() / (
                delta_d.size(1) - 1
            )  # (num_subs, num_neurons, 1)

            # 4. モデルのマージ
            # 4.1 重要度の計算 (Softmax)
            m_scores = torch.softmax(rank_m, dim=0)  # (num_subs, num_neurons, 1)
            d_scores = torch.softmax(rank_d, dim=0)  # (num_subs, num_neurons, 1)

            # 4.2 スコアキャリブレーション
            # 平均を計算
            m_mean = torch.mean(m_scores, dim=0, keepdim=True)
            d_mean = torch.mean(d_scores, dim=0, keepdim=True)

            # 閾値を超える要素を特定し、スコアを s に設定
            m_scores = torch.where(
                m_scores > t * m_mean, torch.full_like(m_scores, s), m_scores
            )  # (num_subs, num_neurons, 1)
            d_scores = torch.where(
                d_scores > t * d_mean, torch.full_like(d_scores, s), d_scores
            )  # (num_subs, num_neurons, 1)
            # 4.3 重みの統合
            w_merged = w_base + torch.sum(
                (m_scores + d_scores) * 0.5 * (torch.stack(w_subs) - w_base), dim=0
            )
    return w_merged
