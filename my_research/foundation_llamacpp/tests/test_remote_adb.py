import subprocess
from pathlib import Path

from my_research.foundation_llamacpp.runner import remote


def test_push_retries_transient_adb_protocol_fault(monkeypatch):
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                1,
                cmd,
                stderr="adb: error: connect failed: protocol fault (couldn't read status): Success",
            )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(remote, "run", fake_run)
    monkeypatch.setattr(remote.time, "sleep", lambda _seconds: None)

    remote.push(["adb"], Path("opencl_phase_mtmd"), "/data/local/tmp/svlm")

    assert len(calls) == 2
    assert calls[0] == calls[1]
