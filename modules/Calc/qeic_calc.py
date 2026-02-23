import torch


def calculate_correlation_matrices(
    base_model: torch.nn.Module,
    sub_model: torch.nn.Module,
    layers: list[str],
    corr_method: str = "pearson",
    device: str = "cpu",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Calculates correlation matrices of activations for specified layers.

    This function computes the neuron-wise correlation matrices for a given
    list of layers in both a base and a sub model. It does this by injecting
    hooks, running a dummy forward pass to capture activations, and then
    computing the correlation.

    Args:
        base_model (torch.nn.Module): The base model.
        sub_model (torch.nn.Module): The sub model.
        layers (list[str]): A list of layer names to compute correlations for.
        corr_method (str, optional): The correlation method, either 'pearson'
            or 'spearman'. Defaults to "pearson".
        device (str, optional): The device to run the forward pass on.
            Defaults to "cpu".

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: A tuple containing two
            lists: the first for the base model's correlation matrices, and
            the second for the sub model's.
    """

    requested_device = (
        device if isinstance(device, torch.device) else torch.device(str(device))
    )
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")

    def get_activations(model, layers, run_device):
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
            model(dummy_input.to(run_device))

        for handle in handles:
            handle.remove()

        return activations

    def compute_corr_matrix(activation, corr_method, compute_device):
        """単一のレイヤーの活性化から相関行列を計算する。"""
        # (バッチサイズ, 系列長, ニューロン数) -> (ニューロン数, バッチサイズ * 系列長)
        activation = activation.view(-1, activation.size(-1)).T.to(
            device=compute_device, dtype=torch.float32
        )

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
        base_activations = get_activations(base_model, layers, requested_device)
        for layer_name in layers:
            # --- 変更箇所: パラメータ名 (.weight など) をモジュール名に変換 ---
            if layer_name.endswith(".weight") or layer_name.endswith(".bias"):
                module_name = ".".join(layer_name.split(".")[:-1])
            else:
                module_name = layer_name
            # --- ここまで ---
            base_corr_matrices.append(
                compute_corr_matrix(
                    base_activations[module_name], corr_method, requested_device
                )
            )  # キーを修正

    if sub_model is not None:
        sub_activations = get_activations(sub_model, layers, requested_device)
        for layer_name in layers:
            # --- 変更箇所: パラメータ名 (.weight など) をモジュール名に変換 ---
            if layer_name.endswith(".weight") or layer_name.endswith(".bias"):
                module_name = ".".join(layer_name.split(".")[:-1])
            else:
                module_name = layer_name
            # --- ここまで ---
            sub_corr_matrices.append(
                compute_corr_matrix(
                    sub_activations[module_name], corr_method, requested_device
                )
            )  # キーを修正

    return base_corr_matrices, sub_corr_matrices


def merge_correlation_matrices(
    base_corr_matrices: list[torch.Tensor],
    sub_corr_matrices: list[torch.Tensor],
    method: str = "average",
) -> list[torch.Tensor]:
    """Merges correlation matrices from base and sub models.

    Args:
        base_corr_matrices (list[torch.Tensor]): A list of correlation matrices
            from the base model.
        sub_corr_matrices (list[torch.Tensor]): A list of correlation matrices
            from the sub model.
        method (str, optional): The merging method. Can be 'average',
            'geometric_mean', or 'quantum_inspired'. Defaults to "average".

    Returns:
        list[torch.Tensor]: A list of the merged correlation matrices.

    Raises:
        ValueError: If the number of matrices in the base and sub lists differ,
            or if an invalid merge method is provided.
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


def QeicAdd(
    v: torch.Tensor,
    base_state_dict: torch.Tensor,
    sub_state_dict: torch.Tensor,
    velocity: float,
    **kwargs,
) -> torch.Tensor:
    """Performs QEIC-based weighted addition.

    This is a wrapper around `qeic_base` for the 'add' operation.
    """
    return qeic_base(v, base_state_dict, sub_state_dict, velocity, "add", **kwargs)


def QeicMix(
    v: torch.Tensor,
    base_state_dict: torch.Tensor,
    sub_state_dict: torch.Tensor,
    velocity: float,
    **kwargs,
) -> torch.Tensor:
    """Performs QEIC-based weighted mixing.

    This is a wrapper around `qeic_base` for the 'mix' operation.
    """
    return qeic_base(v, base_state_dict, sub_state_dict, velocity, "mix", **kwargs)


def QeicSub(
    v: torch.Tensor,
    base_state_dict: torch.Tensor,
    sub_state_dict: torch.Tensor,
    velocity: float,
    **kwargs,
) -> torch.Tensor:
    """Performs QEIC-based subtraction, considering negative correlation.

    This is a wrapper around `qeic_base` for the 'sub' operation.
    """
    return qeic_base(v, base_state_dict, sub_state_dict, velocity, "sub", **kwargs)


def qeic_base(
    v: torch.Tensor,
    base_state_dict: torch.Tensor,
    sub_state_dict: torch.Tensor,
    velocity: float,
    mode: str,
    **kwargs,
) -> torch.Tensor:
    """Core function for QEIC-based merging.

    Calculates a merged weight tensor based on the correlation between neurons
    in the base and sub models.

    Args:
        v (torch.Tensor): The original tensor from the target model.
        base_state_dict (torch.Tensor): The corresponding tensor from the base model.
        sub_state_dict (torch.Tensor): The corresponding tensor from the sub model.
        velocity (float): The velocity parameter for mixing.
        mode (str): The merge mode ('add', 'mix', or 'sub').
        **kwargs: A dictionary of additional parameters, including:
            - layers (list[str]): The layers being processed.
            - corr_method (str): The correlation calculation method.
            - merge_method (str): The method for merging correlation matrices.
            - alpha_mode (str): The method for calculating the weighting factor alpha.
            - beta_mode (str): The method for calculating the weighting factor for subtraction.
            - sub_threshold (float): The threshold for negative correlation in 'sub' mode.
            - base_corr_matrices (list[torch.Tensor]): Pre-calculated correlation matrices for the base model.
            - sub_corr_matrices (list[torch.Tensor]): Pre-calculated correlation matrices for the sub model.

    Returns:
        torch.Tensor: The merged tensor.
    """
    layers = kwargs.get("layers", [])
    corr_method = kwargs.get("corr_method", "pearson")
    merge_method = kwargs.get("merge_method", "average")
    alpha_mode = kwargs.get("alpha_mode", "correlation")
    beta_mode = kwargs.get("beta_mode", "abs")
    sub_threshold = kwargs.get("sub_threshold", -0.1)  # 負の相関の閾値
    # base_model = kwargs.get("base_model") # 使わない
    # sub_model = kwargs.get("sub_model") # 使わない
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

    compute_device = kwargs.get("device", v.device)
    if not isinstance(compute_device, torch.device):
        compute_device = torch.device(str(compute_device))
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        compute_device = torch.device("cpu")

    out_device = v.device
    out_dtype = v.dtype

    v = v.to(device=compute_device, dtype=torch.float32)
    base_state_dict = base_state_dict.to(device=compute_device, dtype=torch.float32)
    sub_state_dict = sub_state_dict.to(device=compute_device, dtype=torch.float32)
    merged_corr = merged_corr.to(device=compute_device, dtype=torch.float32)

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

    return merged_weight.to(device=out_device, dtype=out_dtype)
