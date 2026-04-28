from modules.Utils.models import DummyConfig, prune_config_for_dropped_layers


def test_prune_config_for_gemma4_text_only():
    config = DummyConfig(
        config_dict={
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "audio_config": {"model_type": "gemma4_audio"},
            "vision_config": {"model_type": "gemma4_vision"},
            "audio_token_id": 10,
            "boa_token_id": 11,
            "eoa_token_id": 12,
            "eoa_token_index": 13,
            "image_token_id": 20,
            "video_token_id": 21,
            "boi_token_id": 22,
            "eoi_token_id": 23,
            "vision_soft_tokens_per_image": 280,
            "text_config": {
                "model_type": "gemma4_text",
                "bos_token_id": 2,
                "eos_token_id": 1,
                "vocab_size": 262144,
                "hidden_size": 2560,
            },
        }
    )

    state_dict = {
        "model.language_model.embed_tokens.weight": object(),
        "model.language_model.layers.0.mlp.down_proj.weight": object(),
    }

    prune_config_for_dropped_layers(config, state_dict)

    assert config.architectures == ["Gemma4ForCausalLM"]
    assert config.model_type == "gemma4_text"
    assert config.vocab_size == 262144
    assert not hasattr(config, "text_config")
    assert not hasattr(config, "audio_config")
    assert not hasattr(config, "vision_config")
    assert not hasattr(config, "audio_token_id")
    assert not hasattr(config, "image_token_id")


def test_prune_config_for_audio_only_removal():
    config = DummyConfig(
        config_dict={
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "audio_config": {"model_type": "gemma4_audio"},
            "vision_config": {"model_type": "gemma4_vision"},
            "audio_token_id": 10,
            "boa_token_id": 11,
            "eoa_token_id": 12,
            "eoa_token_index": 13,
            "image_token_id": 20,
            "boi_token_id": 22,
            "eoi_token_id": 23,
            "vision_soft_tokens_per_image": 280,
            "text_config": {"model_type": "gemma4_text"},
        }
    )

    state_dict = {
        "model.language_model.embed_tokens.weight": object(),
        "model.vision_tower.encoder.layers.0.input_layernorm.weight": object(),
        "model.embed_vision.embedding_projection.weight": object(),
    }

    prune_config_for_dropped_layers(config, state_dict)

    assert config.architectures == ["Gemma4ForConditionalGeneration"]
    assert hasattr(config, "vision_config")
    assert not hasattr(config, "audio_config")
    assert not hasattr(config, "audio_token_id")
    assert hasattr(config, "image_token_id")
