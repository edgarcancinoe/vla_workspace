"""Policy-specific adapters."""


def ensure_thesis_policy_registration() -> None:
    from thesis_vla.policies.xvla_guided import configuration_xvla_guided  # noqa: F401


ensure_thesis_policy_registration()
