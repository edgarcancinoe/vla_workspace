import pickle
import sys
import types
import importlib
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(WORKSPACE_ROOT / "repos" / "lerobot" / "src"))
sys.path.append(str(PROJECT_ROOT))


def install_stub_modules():
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.Env = object
    gymnasium.vector = SimpleNamespace(VectorEnv=object)
    sys.modules.setdefault("gymnasium", gymnasium)
    services_pb2 = types.ModuleType("lerobot.transport.services_pb2")

    class Empty:
        pass

    class TransferState:
        TRANSFER_BEGIN = 0
        TRANSFER_MIDDLE = 1
        TRANSFER_END = 2

    services_pb2.Empty = Empty
    services_pb2.TransferState = TransferState
    services_pb2_grpc = types.ModuleType("lerobot.transport.services_pb2_grpc")

    class AsyncInferenceServicer:
        pass

    services_pb2_grpc.AsyncInferenceServicer = AsyncInferenceServicer
    services_pb2_grpc.add_AsyncInferenceServicer_to_server = lambda servicer, server: None
    sys.modules["lerobot.transport.services_pb2"] = services_pb2
    sys.modules["lerobot.transport.services_pb2_grpc"] = services_pb2_grpc


install_stub_modules()

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation

from thesis_vla.inference.resident_eval import resolve_episode_prompts

resident_policy_server_module = importlib.import_module("thesis_vla.inference.resident_policy_server")
ResidentPolicyServer = resident_policy_server_module.ResidentPolicyServer
ResidentPolicyServerConfig = resident_policy_server_module.ResidentPolicyServerConfig


class FakeContext:
    def peer(self):
        return "pytest-client"


class FakeRequest:
    def __init__(self, payload):
        self.data = payload


class FakePolicy:
    def __init__(self):
        self.config = SimpleNamespace(type="xvla", image_features={}, normalization_mapping={})
        self.model = SimpleNamespace(action_space=SimpleNamespace(), chunk_size=2)
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, observation):
        return torch.tensor([[[1.0], [2.0]]], dtype=torch.float32)


def make_policy_request(rename_map=None, lerobot_features=None):
    return FakeRequest(
        pickle.dumps(
            RemotePolicyConfig(
                policy_type="xvla",
                pretrained_name_or_path="repo/model",
                lerobot_features=lerobot_features or {"observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}},
                actions_per_chunk=2,
                device="cpu",
                rename_map=rename_map or {"observation.images.main": "observation.images.image"},
            )
        )
    )


def test_resolve_episode_prompts_cycles_and_prefers_task_prompts():
    prompts = resolve_episode_prompts("fallback", ["one", "two"], 5)
    assert prompts == ["one", "two", "one", "two", "one"]
    assert resolve_episode_prompts("fallback", [], None) == ["fallback"]


def test_run_policy_local_uses_shared_runtime_helpers():
    source = (PROJECT_ROOT / "apps" / "eval" / "run_policy_local.py").read_text()
    assert "load_runtime_policy(" in source
    assert "build_runtime_policy_processors(" in source


def test_resident_server_loads_policy_once_and_rebuilds_processors(monkeypatch):
    calls = {"load": 0, "build": 0, "sync": 0}

    def fake_load_runtime_policy(*args, **kwargs):
        calls["load"] += 1
        return FakePolicy(), True

    def fake_build_runtime_policy_processors(*args, **kwargs):
        calls["build"] += 1
        return lambda observation: observation, lambda action: action

    def fake_sync_xvla_policy_with_features(*args, **kwargs):
        calls["sync"] += 1

    monkeypatch.setattr("thesis_vla.inference.resident_policy_server.load_runtime_policy", fake_load_runtime_policy)
    monkeypatch.setattr("thesis_vla.inference.resident_policy_server.build_runtime_policy_processors", fake_build_runtime_policy_processors)
    monkeypatch.setattr("thesis_vla.inference.resident_policy_server.sync_xvla_policy_with_features", fake_sync_xvla_policy_with_features)

    server = ResidentPolicyServer(ResidentPolicyServerConfig(policy_type="xvla", pretrained_name_or_path="repo/model", policy_device="cpu"))
    request = make_policy_request()
    server.SendPolicyInstructions(request, FakeContext())
    server.SendPolicyInstructions(request, FakeContext())

    assert calls["load"] == 1
    assert calls["build"] == 2
    assert calls["sync"] == 2
    assert server.policy_load_count == 1


def test_resident_server_ready_and_prompt_propagation(monkeypatch):
    fake_policy = FakePolicy()
    seen_tasks = []

    def fake_load_runtime_policy(*args, **kwargs):
        return fake_policy, True

    def fake_preprocessor(observation):
        seen_tasks.append(observation["task"])
        return observation

    monkeypatch.setattr("thesis_vla.inference.resident_policy_server.load_runtime_policy", fake_load_runtime_policy)
    monkeypatch.setattr("thesis_vla.inference.resident_policy_server.build_runtime_policy_processors", lambda *args, **kwargs: (fake_preprocessor, lambda action: action))
    monkeypatch.setattr("thesis_vla.inference.resident_policy_server.sync_xvla_policy_with_features", lambda *args, **kwargs: None)

    server = ResidentPolicyServer(ResidentPolicyServerConfig(policy_type="xvla", pretrained_name_or_path="repo/model", policy_device="cpu"))
    server.SendPolicyInstructions(make_policy_request(), FakeContext())
    server.Ready(None, FakeContext())
    server.lerobot_features = {"observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}}
    server.actions_per_chunk = 2
    server._predict_action_chunk(TimedObservation(timestamp=0.0, timestep=0, observation={"joint.pos": 0.0, "task": "pick red"}))
    server._predict_action_chunk(TimedObservation(timestamp=0.0, timestep=1, observation={"joint.pos": 1.0, "task": "pick blue"}))

    assert seen_tasks == ["pick red", "pick blue"]
    assert fake_policy.reset_calls >= 2
