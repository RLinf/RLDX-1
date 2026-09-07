"""Offline loading contracts: backbone and checkpoint revisions are distinct."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rldx.policy import policy_loader as loader


REVISION = "4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7"


@pytest.mark.parametrize("revision", [None, REVISION])
def test_model_revision_only_reaches_nested_backbone(monkeypatch, tmp_path, revision):
    config_load = Mock(return_value=SimpleNamespace(use_memory=False))
    model = SimpleNamespace(eval=Mock())
    model_load = Mock(return_value=model)
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", config_load)
    monkeypatch.setattr(loader.AutoModel, "from_pretrained", model_load)

    assert loader._load_model(tmp_path, "cpu", False, revision) is model
    config_load.assert_called_once_with(tmp_path, trust_remote_code=True)
    kwargs = model_load.call_args.kwargs
    assert "revision" not in kwargs
    if revision is None:
        assert "transformers_loading_kwargs" not in kwargs
    else:
        assert kwargs["transformers_loading_kwargs"] == {"revision": revision}
    model.eval.assert_called_once_with()


@pytest.mark.parametrize("subdirectory", [False, True])
@pytest.mark.parametrize("revision", [None, REVISION])
def test_processor_revision_only_reaches_nested_support(
    monkeypatch, tmp_path, subdirectory, revision
):
    directory = tmp_path / "processor" if subdirectory else tmp_path
    directory.mkdir(exist_ok=True)
    processor = SimpleNamespace(physics_keys=[], conversation_image_first=True, eval=Mock())
    load = Mock(return_value=processor)
    monkeypatch.setattr(loader.AutoProcessor, "from_pretrained", load)
    model = SimpleNamespace(config=SimpleNamespace())

    assert loader._load_processor(tmp_path, model, revision) is processor
    assert load.call_args.args == (directory,)
    kwargs = load.call_args.kwargs
    assert "revision" not in kwargs
    if revision is None:
        assert kwargs == {}
    else:
        assert kwargs["transformers_loading_kwargs"]["revision"] == revision


def test_missing_support_does_not_retry_with_main(monkeypatch, tmp_path):
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", Mock(return_value=SimpleNamespace()))
    load = Mock(side_effect=OSError("pinned support files are not cached"))
    monkeypatch.setattr(loader.AutoModel, "from_pretrained", load)
    with pytest.raises(OSError, match="not cached"):
        loader._load_model(tmp_path, "cpu", False, REVISION)
    assert load.call_count == 1


def test_factory_forwards_revision_and_leaves_remote_configuration_to_server(monkeypatch):
    from rldx.eval.rollout_policy import create_rldx_sim_policy

    policy = Mock(return_value="policy")
    wrapper = Mock(return_value="wrapped")
    stub = types.ModuleType("rldx.policy.rldx_policy")
    stub.RLDXPolicy = policy
    stub.RLDXSimPolicyWrapper = wrapper
    monkeypatch.setitem(sys.modules, stub.__name__, stub)

    assert (
        create_rldx_sim_policy("checkpoint", "embodiment", backbone_revision=REVISION) == "wrapped"
    )
    assert policy.call_args.kwargs["backbone_revision"] == REVISION
    with pytest.raises(ValueError, match="policy server"):
        create_rldx_sim_policy(
            "checkpoint", "embodiment", "localhost", 5555, backbone_revision=REVISION
        )


def test_loader_propagates_same_revision_to_model_and_processor(monkeypatch, tmp_path):
    class StopAfterProcessor(Exception):
        pass

    model = object()
    load_model = Mock(return_value=model)
    load_processor = Mock(side_effect=StopAfterProcessor)
    monkeypatch.setattr(loader, "_load_model", load_model)
    monkeypatch.setattr(loader, "_apply_model_tweaks", Mock())
    monkeypatch.setattr(loader, "_load_processor", load_processor)
    with pytest.raises(StopAfterProcessor):
        loader.PolicyLoader.load(
            embodiment_tag="test",
            model_path=str(tmp_path),
            device="cpu",
            backbone_revision=REVISION,
        )
    load_model.assert_called_once_with(Path(tmp_path), "cpu", False, REVISION)
    load_processor.assert_called_once_with(Path(tmp_path), model, REVISION)
