import json

import pytest

from scripts.eval_artifacts import validate_adapter_base
from vpo_rm.trainer import TrainerConfig, VPOTrainer


def adapter_for(tmp_path, model):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(model)}))
    return adapter


def test_exact_huggingface_model_id_is_a_valid_adapter_binding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = adapter_for(tmp_path, "org/base-checkpoint")
    validate_adapter_base(adapter, "org/base-checkpoint")
    with pytest.raises(ValueError, match="incompatible"):
        validate_adapter_base(adapter, "org/other-checkpoint")


def test_factory_rejects_wrong_adapter_before_any_tokenizer_or_model_load(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    adapter = adapter_for(tmp_path, "org/base-checkpoint")
    config = TrainerConfig(model_name="org/other-checkpoint", init_adapter=str(adapter),
                           actor_device="cpu", reward_device="cpu", output_dir=str(tmp_path / "run"))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: pytest.fail("Tokenizer loading began"))
    with pytest.raises(ValueError, match="incompatible"):
        VPOTrainer.from_pretrained(config)


def test_factory_accepts_matching_local_base_and_reaches_tokenizer(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    model = tmp_path / "model"
    model.mkdir()
    adapter = adapter_for(tmp_path, model)
    config = TrainerConfig(model_name=str(model), init_adapter=str(adapter),
                           actor_device="cpu", reward_device="cpu", output_dir=str(tmp_path / "run"))

    class TokenizerReached(RuntimeError):
        pass

    def stop_at_tokenizer(*args, **kwargs):
        raise TokenizerReached()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", stop_at_tokenizer)
    with pytest.raises(TokenizerReached):
        VPOTrainer.from_pretrained(config)
