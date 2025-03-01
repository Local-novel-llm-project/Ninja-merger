import torch


def calculate_correlation_matrices(
    base_model, sub_model, layers, corr_method="pearson", device="cpu"
):
    """
    ベースモデルとサブモデルの指定されたレイヤーの活性化の相関行列を計算する。

    Args:
        base_model: ベースモデル (torch.nn.Module)。
        sub_model: サブモデル (torch.nn.Module)。
        layers: 相関を計算するレイヤー名のリスト。
        corr_method: 相関の計算方法 ("pearson" または "spearman")。
        device: 計算を行うデバイス ("cpu" または "cuda")。

    Returns:
        base_corr_matrices: ベースモデルの各レイヤーの相関行列のリスト。
        sub_corr_matrices: サブモデルの各レイヤーの相関行列のリスト。
           (モデルがNoneの場合は空のリストを返す)
    """

    def get_activations(model, layers, device):
        """モデルの指定されたレイヤーの活性化を取得する。"""
        activations = {}
        handles = []

        def hook_fn(module, input, output, layer_name):
            activations[layer_name] = output.detach().to("cpu")  # CPUに戻す

        for layer_name in layers:
            # --- 変更箇所: パラメータ名 (.weight など) をモジュール名に変換 ---
            if layer_name.endswith(".weight") or layer_name.endswith(".bias"):
                module_name = ".".join(layer_name.split(".")[:-1])
            else:
                module_name = layer_name
            # --- ここまで ---
            layer = dict([*model.named_modules()])[module_name]  # 正しい
            handle = layer.register_forward_hook(
                lambda module,
                input,
                output,
                layer_name=module_name: hook_fn(  # hook_fn にはモジュール名を渡す
                    module, input, output, layer_name
                )
            )
            handles.append(handle)

        # ダミーの入力データでフォワードパスを実行
        dummy_input = torch.randint(
            0, model.config.vocab_size, (1, 64), device="cpu"
        )  # バッチサイズ1, 系列長64
        with torch.no_grad():
            model(dummy_input.to(device))

        for handle in handles:
            handle.remove()

        return activations

    def compute_corr_matrix(activation, corr_method):
        """単一のレイヤーの活性化から相関行列を計算する。"""
        # (バッチサイズ, 系列長, ニューロン数) -> (ニューロン数, バッチサイズ * 系列長)
        activation = activation.view(-1, activation.size(-1)).T

        activation = activation.to("cuda")  # GPUに転送

        if corr_method == "pearson":
            corr_matrix = torch.corrcoef(activation)
        elif corr_method == "spearman":
            # 順位を計算 (計算負荷が高い)
            ranks = activation.argsort(dim=1).argsort(dim=1).float()
            corr_matrix = torch.corrcoef(ranks)
        else:
            raise ValueError(f"Invalid corr_method: {corr_method}")

        return corr_matrix.cpu()  # CPUに戻す

    base_corr_matrices = []
    sub_corr_matrices = []

    if base_model is not None:
        base_activations = get_activations(base_model, layers, device)
        for layer_name in layers:
            # --- 変更箇所: パラメータ名 (.weight など) をモジュール名に変換 ---
            if layer_name.endswith(".weight") or layer_name.endswith(".bias"):
                module_name = ".".join(layer_name.split(".")[:-1])
            else:
                module_name = layer_name
            # --- ここまで ---
            base_corr_matrices.append(
                compute_corr_matrix(base_activations[module_name], corr_method)
            )  # キーを修正

    if sub_model is not None:
        sub_activations = get_activations(sub_model, layers, device)
        for layer_name in layers:
            # --- 変更箇所: パラメータ名 (.weight など) をモジュール名に変換 ---
            if layer_name.endswith(".weight") or layer_name.endswith(".bias"):
                module_name = ".".join(layer_name.split(".")[:-1])
            else:
                module_name = layer_name
            # --- ここまで ---
            sub_corr_matrices.append(
                compute_corr_matrix(sub_activations[module_name], corr_method)
            )  # キーを修正

    return base_corr_matrices, sub_corr_matrices


def merge_correlation_matrices(base_corr_matrices, sub_corr_matrices, method="average"):
    """
    ベースモデルとサブモデルの相関行列をマージする。

    Args:
        base_corr_matrices: ベースモデルの相関行列のリスト。
        sub_corr_matrices: サブモデルの相関行列のリスト。
        method: マージ方法 ("average", "geometric_mean", "quantum_inspired")。

    Returns:
        merged_corr_matrices: マージされた相関行列のリスト。
    """

    merged_corr_matrices = []
    if len(base_corr_matrices) != len(sub_corr_matrices):
        # QEICでは通常起こらないはずだが、念の為
        raise ValueError("The number of layers in base and sub models do not match.")

    for base_corr, sub_corr in zip(base_corr_matrices, sub_corr_matrices):
        if method == "average":
            merged_corr = (base_corr + sub_corr) / 2
        elif method == "geometric_mean":
            merged_corr = torch.sqrt(base_corr * sub_corr)  # 要素ごとの積の平方根
        elif method == "quantum_inspired":
            # 簡略化のため、t=0.5 (算術平均と幾何平均の中間) に固定
            merged_corr = torch.sqrt(torch.sqrt(base_corr) + torch.sqrt(sub_corr))
        else:
            raise ValueError(f"Invalid merge_method: {method}")

        merged_corr_matrices.append(merged_corr)

    return merged_corr_matrices


def QeicAdd(v, base_state_dict, sub_state_dict, velocity, **kwargs):
    """
    QEICに基づく重み付け加算を行う。
    """
    return qeic_base(v, base_state_dict, sub_state_dict, velocity, "add", **kwargs)


def QeicMix(v, base_state_dict, sub_state_dict, velocity, **kwargs):
    """
    QEICに基づく重み付け混合を行う。
    """
    return qeic_base(v, base_state_dict, sub_state_dict, velocity, "mix", **kwargs)


def QeicSub(v, base_state_dict, sub_state_dict, velocity, **kwargs):
    """
    QEICに基づく減算を行う（負の相関を考慮）。
    """
    return qeic_base(v, base_state_dict, sub_state_dict, velocity, "sub", **kwargs)


def qeic_base(v, base_state_dict, sub_state_dict, velocity, mode, **kwargs):
    layers = kwargs.get("layers", [])
    corr_method = kwargs.get("corr_method", "pearson")
    merge_method = kwargs.get("merge_method", "average")
    alpha_mode = kwargs.get("alpha_mode", "correlation")
    beta_mode = kwargs.get("beta_mode", "abs")
    sub_threshold = kwargs.get("sub_threshold", -0.1)  # 負の相関の閾値
    # base_model = kwargs.get("base_model") # 使わない
    # sub_model = kwargs.get("sub_model") # 使わない
    # device = v.device # 使わない
    base_corr_matrices = kwargs.get("base_corr_matrices")
    sub_corr_matrices = kwargs.get("sub_corr_matrices")

    if not layers:
        return v  # レイヤーが指定されていない場合は、元の重みを返す

    merged_corr_matrices = merge_correlation_matrices(
        base_corr_matrices, sub_corr_matrices, merge_method
    )

    if not merged_corr_matrices:
        print(
            "[yellow]Warning: No correlation matrices were merged. Skipping QEIC.[/yellow]"
        )
        return v  # マージする相関行列がない場合は、元の重みを返す

    # 最初の相関行列を使用 (複数ある場合は、最初のものを使う)
    merged_corr = merged_corr_matrices[0]

    # --- 変更箇所 (ここから) ---
    v = v.to("cuda")
    base_state_dict = base_state_dict.to("cuda")
    sub_state_dict = sub_state_dict.to("cuda")
    merged_corr = merged_corr.to("cuda")
    # --- 変更箇所 (ここまで) ---

    # 相関行列のサイズとvのサイズが一致しない場合
    if merged_corr.shape[0] != v.shape[0]:
        print("v.shape", v.shape)
        print("merged_corr_matrices.shape", merged_corr.shape)
        return v  # マージは行わない

    # 重み付け係数の計算
    if mode == "add" or mode == "mix":
        if alpha_mode == "correlation":
            alpha = merged_corr.clamp(min=0, max=1)  # 相関係数を 0-1 の範囲にクリップ
            if mode == "mix":
                alpha = alpha * velocity  # mix の場合は、velocityをかける。
        elif alpha_mode == "fixed":
            alpha = torch.full_like(merged_corr, 0.5)  # 固定値 (0.5)
        else:
            raise ValueError(f"Invalid alpha_mode: {alpha_mode}")
    elif mode == "sub":
        if beta_mode == "abs":
            alpha = -torch.abs(merged_corr)  # 負の相関の絶対値
        elif beta_mode == "threshold":
            alpha = torch.where(
                merged_corr < sub_threshold,
                -torch.abs(merged_corr),
                torch.zeros_like(merged_corr),
            )
        else:
            raise ValueError(f"Invalid beta_mode: {beta_mode}")

    # 重みのマージ
    if mode == "sub":
        merged_weight = v + alpha * (base_state_dict - sub_state_dict)  # 減算
    elif mode == "add":
        merged_weight = (1 - alpha) * v + alpha * (
            base_state_dict + sub_state_dict
        )  # 重み付け加算
    else:
        merged_weight = (
            1 - alpha
        ) * base_state_dict + alpha * sub_state_dict  # 重み付け混合

    return merged_weight.cpu()  # CPUに戻す
