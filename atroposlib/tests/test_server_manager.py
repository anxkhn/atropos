"""Tests for ServerManager SLURM node-distribution URL construction.

These cover the case where the documented ``INFER_TP`` environment variable is
set (SLURM.md tells users to ``export INFER_TP=$TP_SIZE``). Because environment
variables are always strings, ``8 // os.environ.get("INFER_TP", 1)`` must cast
the value to ``int`` before the integer division, otherwise it raises
``TypeError: unsupported operand type(s) for //: 'int' and 'str'``.
"""

import os

import pytest

from atroposlib.envs.server_handling.server_baseline import (
    APIServerConfig,
    ServerBaseline,
)
from atroposlib.envs.server_handling.server_manager import ServerManager


class _FakeScontrol:
    """Stand-in for the file object returned by ``os.popen``."""

    def __init__(self, hostnames):
        self._output = "\n".join(hostnames) + "\n"

    def read(self):
        return self._output


@pytest.fixture
def slurm_env(monkeypatch):
    """Patch ``os.popen`` to return two SLURM hostnames without calling scontrol."""

    def fake_popen(cmd):
        return _FakeScontrol(["node-train", "node-infer"])

    monkeypatch.setattr(os, "popen", fake_popen)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node-[train,infer]")
    monkeypatch.setenv("NUM_TRAINING_NODES", "1")


def test_slurm_list_configs_infer_tp_set(slurm_env, monkeypatch):
    """A set INFER_TP must not crash the list-configs SLURM path (line ~175)."""
    monkeypatch.setenv("INFER_TP", "2")

    configs = [APIServerConfig(model_name="test-model", api_key="x")]
    manager = ServerManager(configs=configs, slurm=True)

    # 1 inference node (node-infer) x (8 // 2) = 4 servers on ports 9000-9003.
    urls = [server.config.base_url for server in manager.servers]
    assert urls == [f"http://node-infer:{9000 + i}/v1" for i in range(4)]


def test_slurm_baseline_config_infer_tp_set(slurm_env, monkeypatch):
    """A set INFER_TP must not crash the ServerBaseline SLURM path (line ~123)."""
    monkeypatch.setenv("INFER_TP", "2")

    manager = ServerManager(configs=ServerBaseline(model_name="test-model"), slurm=True)

    urls = [server.config.base_url for server in manager.servers]
    assert urls == [f"http://node-infer:{9000 + i}/v1" for i in range(4)]


def test_slurm_infer_tp_unset_defaults_to_eight(slurm_env):
    """With INFER_TP unset the default of 1 yields 8 servers per node (no regression)."""
    configs = [APIServerConfig(model_name="test-model", api_key="x")]
    manager = ServerManager(configs=configs, slurm=True)

    urls = [server.config.base_url for server in manager.servers]
    assert urls == [f"http://node-infer:{9000 + i}/v1" for i in range(8)]
