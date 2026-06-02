import logging
import pickle  # nosec
from concurrent import futures
from dataclasses import asdict, dataclass, field
from pprint import pformat

import draccus
import grpc

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore

from thesis_vla.inference.runtime_policy import (
    RuntimePolicyOverrides,
    build_runtime_policy_processors,
    load_runtime_policy,
    sync_xvla_policy_with_features,
)


@dataclass
class ResidentPolicyServerConfig(PolicyServerConfig):
    policy_type: str = field(default="xvla")
    pretrained_name_or_path: str = field(default="")
    policy_device: str = field(default="cpu")
    chunk_size: int | None = field(default=None)
    n_action_steps: int | None = field(default=None)
    max_action_tokens: int | None = field(default=None)
    num_xvla_obs_steps: int = field(default=1)
    binary_gripper_inference: bool = field(default=False)


class ResidentPolicyServer(PolicyServer):
    def __init__(self, config: ResidentPolicyServerConfig):
        super().__init__(config)
        self.config = config
        self.policy_load_count = 0
        self._load_policy()

    def _load_policy(self) -> None:
        if self.policy is not None:
            return
        self.policy, _ = load_runtime_policy(
            policy_type=self.config.policy_type,
            pretrained_path=self.config.pretrained_name_or_path,
            device=self.config.policy_device,
            overrides=RuntimePolicyOverrides(
                chunk_size=self.config.chunk_size,
                n_action_steps=self.config.n_action_steps,
                max_action_tokens=self.config.max_action_tokens,
                num_xvla_obs_steps=self.config.num_xvla_obs_steps,
                binary_gripper_inference=self.config.binary_gripper_inference,
            ),
        )
        self.device = self.config.policy_device
        self.policy_type = self.config.policy_type
        self.policy_load_count += 1

    def _assert_requested_policy_identity(self, policy_specs: RemotePolicyConfig) -> None:
        expected = (self.config.policy_type, self.config.pretrained_name_or_path, self.config.policy_device)
        actual = (policy_specs.policy_type, policy_specs.pretrained_name_or_path, policy_specs.device)
        if actual != expected:
            raise ValueError(
                "ResidentPolicyServer is pinned to a single checkpoint per process. "
                f"Expected {expected}, got {actual}."
            )

    def Ready(self, request, context):  # noqa: N802
        response = super().Ready(request, context)
        self.fps_tracker.reset()
        if self.policy is not None:
            self.policy.reset()
        return response

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        policy_specs = pickle.loads(request.data)  # nosec
        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        self._assert_requested_policy_identity(policy_specs)
        self._load_policy()
        self.device = self.config.policy_device
        self.policy_type = self.config.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        if self.policy_type == "xvla":
            sync_xvla_policy_with_features(self.policy, self.lerobot_features, policy_specs.rename_map)

        self.preprocessor, self.postprocessor = build_runtime_policy_processors(
            policy=self.policy,
            pretrained_path=self.config.pretrained_name_or_path,
            device=self.config.policy_device,
            rename_map=policy_specs.rename_map,
            stats=None,
            use_dataset_stats=False,
        )
        self.policy.reset()
        return services_pb2.Empty()


@draccus.wrap()
def serve(cfg: ResidentPolicyServerConfig):
    logging.info(pformat(asdict(cfg)))
    policy_server = ResidentPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info(f"ResidentPolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("ResidentPolicyServer terminated")


if __name__ == "__main__":
    serve()
