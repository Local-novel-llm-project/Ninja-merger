import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel, AutoModelForVision2Seq, PreTrainedModel, PretrainedConfig
from peft import PeftConfig, PeftModel

def load_model(model, lora_name, device,torch_dtype):
    model = AutoModelForCausalLM.from_pretrained(
    # cfg = PretrainedConfig().from_pretrained(model)
    # model = PTModel.from_pretrained(
    # model = AutoModelForVision2Seq.from_pretrained(
        model,
        # config=cfg,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    if lora_name is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("start merging LoRA")
        model = PeftModel.from_pretrained(model, lora_name, device=device)
        model = model.merge_and_unload()
        print("LoRA merged")
    state_dict = model.state_dict()
    return model, state_dict


def load_vlm_model(model, lora_name, device,torch_dtype):
    # model = AutoModelForCausalLM.from_pretrained(
    model = AutoModelForVision2Seq.from_pretrained(
    # cfg = PretrainedConfig().from_pretrained(model)
    # model = PTModel.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True
    )
    if lora_name is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("start merging LoRA")
        model = PeftModel.from_pretrained(model, lora_name, device=device)
        model = model.merge_and_unload()
        print("LoRA merged")
    state_dict = model.state_dict()
    return model, state_dict


def load_llava_model(model, lora_name, device,torch_dtype):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    # model = AutoModelForCausalLM.from_pretrained(
    tokenizer, model, image_processor, context_len = load_pretrained_model(
    # cfg = PretrainedConfig().from_pretrained(model)
    # model = PTModel.from_pretrained(
        model_path=model,
        model_base=None,
        model_name=get_model_name_from_path(model),
        torch_dtype=torch_dtype,
        device_map=device,
    )
    if lora_name is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("start merging LoRA")
        model = PeftModel.from_pretrained(model, lora_name, device=device)
        model = model.merge_and_unload()
        print("LoRA merged")
    state_dict = model.state_dict()
    return model, state_dict, tokenizer, image_processor


def load_left_right_models(model_dict,merge_models_device):
    velocity = model_dict["velocity"]
    base_weight, base_state_dict = load_model(model_dict["left"], None, merge_models_device)
    sub_weight, sub_state_dict = load_model(model_dict["right"], None, merge_models_device)
    # del base_weight, sub_weight
    return (base_weight, base_state_dict), (sub_weight, sub_state_dict), velocity

def load_processor(model_name):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    # cfg = PretrainedConfig().from_pretrained(model_name)
    # tokenizer = PTModel.from_pretrained(model_name, config=cfg, torch_dtype=torch_dtype)

    return processor


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = AutoProcessor.from_pretrained(model_name)
    # cfg = PretrainedConfig().from_pretrained(model_name)
    # tokenizer = PTModel.from_pretrained(model_name, config=cfg, torch_dtype=torch_dtype)
    return tokenizer