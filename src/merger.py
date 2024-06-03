import torch
from tqdm import tqdm

import calc_method
from layer_utility import is_layer_dropped, is_layer_included, parse_layer_specifications


def merge(
    skip_layernorm, target_state_dict, base_state_dict, sub_state_dict,
    velocity, post_velocity, skip_layers,
    operation, post_operation, normalization,
    include_layers, exclude_layers, drop_layers,
    unmatch_size_layer_op, dry_run=False, is_llava_next=False):
    
    if target_state_dict is None:
        target = base_state_dict
        post_operation = "mix"
        post_velocity = 1.0
    else:
        target = target_state_dict

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

    with tqdm(target.keys()) as pbar:
        for k in pbar:
            target_k = k
            if is_llava_next:
                k = k.replace("language_model.", "", 1)

            if is_layer_dropped(k, drop_ranges, drop_specific):
                dropped_layers.append(k)
                continue
            if (k in skip_layers) or (skip_layernorm and "layernorm" in k):
                excluded_layers.append(k)
                continue
            if not is_layer_included(k, include_ranges, include_specific, exclude_ranges, exclude_specific):
                excluded_layers.append(k)
                continue
            included_layers.append(k)
            v = target[target_k]
            pbar.set_description(k)
            if not dry_run:
                if unmatch_size_layer_op == "skip":
                    v.copy_(normalization_dict[normalization](
                        post_operation_dict[post_operation],
                        v,
                        operation_dict[operation](v, base_state_dict[k], sub_state_dict[k], velocity),
                        post_velocity
                    ))
                else:   # only_common_range
                    min_size = min(v.shape, base_state_dict[k].shape, sub_state_dict[k].shape)
                    def get_slice_to(t, indice):
                        if len(indice) == 1:
                            print(indice)
                            return t[:indice[0]]
                        elif len(indice) == 2:
                            return t[:indice[0], :indice[1]]
                        elif len(indice) == 3:
                            return t[:indice[0], :indice[1], :indice[2]]
                        # TODO: support N dimentional
                        return t
                    min_size = min(v.shape, base_state_dict[k].shape, sub_state_dict[k].shape)
                    get_slice_to(v, min_size).copy_(normalization_dict[normalization](
                        post_operation_dict[post_operation],
                        get_slice_to(v, min_size),
                        operation_dict[operation](get_slice_to(v, min_size), get_slice_to(base_state_dict[k], min_size), get_slice_to(sub_state_dict[k], min_size), velocity),
                        post_velocity
                    ))

    with tqdm(dropped_layers) as pbar:
        for k in pbar:
            target_k = k
            if is_llava_next:
                k = k.replace("language_model.", "", 1)
            pbar.set_description(f"drop {k}")
            del target[target_k]


    print("Included Layers:")
    print(included_layers)
    print("Excluded Layers:")
    print(excluded_layers)
    print("Dropped Layers:")
    print(dropped_layers)

    return target