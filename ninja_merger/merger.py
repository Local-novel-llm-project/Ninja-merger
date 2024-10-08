# Copyright (c) 2024 TylorShine
#                    UmiseTokikaze
#                    Exveria
# license: Apache-2.0

import torch
from rich import print
from rich.progress import track

from . import calc_method
from .layer_utility import (
    is_layer_dropped,
    is_layer_included,
    parse_layer_specifications,
)


def merge(
    skip_layernorm,
    target_model,
    base_models,
    sub_models,
    velocity,
    post_velocity,
    skip_layers,
    operation,
    post_operation,
    preprocess,
    post_preprocess,
    normalization,
    include_layers,
    exclude_layers,
    drop_layers,
    unmatch_size_layer_op,
    dry_run=False,
    is_llava_next=False,
):

    if target_model is None:
        target = base_models[0].state_dict()
        post_operation = "mix"
        post_velocity = 1.0
    else:
        target = target_model.state_dict()
        
    sub_state_dict = sub_models.state_dict()
    
    preprocess_dict = {
        "none": lambda x, y, v, **kwargs: (x, y),
        "pcd": calc_method.GitReBasin,
    }
    operation_dict = {
        "add": calc_method.Add,
        "sub": calc_method.Sub,
        "mul": calc_method.Mul,
        "div": calc_method.Div,
        "mix": calc_method.Mix,
        "avg": calc_method.Avg,
        "concat": calc_method.Concatenation,
        "maxpool": calc_method.MaxPool,
        "minpool": calc_method.MinPool,
        "geometric_mean": calc_method.GeometricMean,
        "std_sub": calc_method.StdSub,
    }
    normalization_dict = {
        "none": calc_method.Passthrough,
        "norm_std_mean": calc_method.NormStdMean,
        "match_std_mean": calc_method.MatchStdMean,
        "proc_std_mean": calc_method.ProcStdMean,
        "angle_merge": calc_method.NormAngleMerge,
    }
    post_operation_dict = {
        "add": calc_method.PostAdd,
        "sub": calc_method.PostSub,
        "subfrom": calc_method.PostSubFrom,
        "mul": calc_method.PostMul,
        "div": calc_method.PostDiv,
        "divby": calc_method.PostDivBy,
        "mix": calc_method.PostMix,
        "concat": calc_method.PostConcatenation,
        "maxpool": calc_method.PostMaxPool,
        "minpool": calc_method.PostMinPool,
        "geometric_mean": calc_method.PostGeometricMean,
    }
    included_layers = []
    excluded_layers = []
    dropped_layers = []

    include_ranges, include_specific = parse_layer_specifications(include_layers)
    exclude_ranges, exclude_specific = parse_layer_specifications(exclude_layers)
    drop_ranges, drop_specific = parse_layer_specifications(drop_layers)

    for k in track(target.keys(), description="Merging layers"):
        target_k = k
        if is_llava_next:
            k = k.replace("language_model.", "", 1)

        if is_layer_dropped(k, drop_ranges, drop_specific):
            dropped_layers.append(k)
            continue
        if (k in skip_layers) or (skip_layernorm and "layernorm" in k):
            excluded_layers.append(k)
            continue
        if not is_layer_included(
            k, include_ranges, include_specific, exclude_ranges, exclude_specific
        ):
            excluded_layers.append(k)
            continue
        included_layers.append(k)
        v = target[target_k]
        if not dry_run:
            if unmatch_size_layer_op == "skip":
                v.copy_(
                    normalization_dict[normalization](
                        post_operation_dict[post_operation],
                        *preprocess_dict[post_preprocess](
                            v,
                            [operation_dict[operation](
                                v, *preprocess_dict[preprocess](b.state_dict()[k], sub_state_dict[k]), velocity
                            ) for b in base_models],
                        ),
                        post_velocity,
                    )
                )
            else:  # only_common_range
                min_size = min(
                    v.shape, *[b.state_dict()[k].shape for b in base_models], sub_state_dict[k].shape
                )

                def get_slice_to(t, indice):
                    if len(indice) == 1:
                        return t[: indice[0]]
                    elif len(indice) == 2:
                        return t[: indice[0], : indice[1]]
                    elif len(indice) == 3:
                        return t[: indice[0], : indice[1], : indice[2]]
                    # TODO: support N dimentional
                    return t
                
                get_slice_to(v, min_size).copy_(
                    normalization_dict[normalization](
                        post_operation_dict[post_operation],
                        *preprocess_dict[post_preprocess](
                            get_slice_to(v, min_size),
                            [operation_dict[operation](
                                get_slice_to(v, min_size),
                                *preprocess_dict[preprocess](
                                    get_slice_to(b.state_dict()[k], min_size),
                                    get_slice_to(sub_state_dict[k], min_size),
                                ),
                                velocity,
                            ) for b in base_models],
                        ),
                        post_velocity,
                    )
                )

    for k in track(dropped_layers, description="Dropping layers"):
        target_k = k
        if is_llava_next:
            k = k.replace("language_model.", "", 1)
        del target[target_k]

    print("[bold]Included Layers:[/bold]")
    print(included_layers)
    print("[bold]Excluded Layers:[/bold]")
    print(excluded_layers)
    print("[bold]Dropped Layers:[/bold]")
    print(dropped_layers)

    return target
