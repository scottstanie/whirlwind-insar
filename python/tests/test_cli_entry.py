"""The ``whirlwind`` console script wiring (``_native.cli_main`` +
``whirlwind._climain:main``). The CLI logic itself lives in Rust
(``crates/whirlwind-cli``) - these tests cover the Python entry path: exit
codes, stdout/stderr routing, and an end-to-end simulate_ifg -> unwrap
roundtrip through the in-process entry point.

``capfd`` (fd-level capture) is required: the Rust side writes to the real
file descriptors, which ``capsys`` cannot see.
"""

import sys

import numpy as np
import pytest

from whirlwind._climain import main
from whirlwind._native import cli_main


def test_help_exits_zero(capfd):
    assert cli_main(["--help"]) == 0
    out = capfd.readouterr().out
    assert "Usage: whirlwind" in out


def test_version_matches_package(capfd):
    import whirlwind as ww

    assert cli_main(["--version"]) == 0
    assert capfd.readouterr().out.strip() == f"whirlwind {ww.__version__}"


def test_usage_error_exit_code(capfd):
    assert cli_main(["--not-a-flag"]) == 2
    assert "unexpected argument" in capfd.readouterr().err


def test_removed_simulate_subcommand_points_to_replacement(capfd):
    assert cli_main(["simulate", "--shape", "64x64", "--out", "sim"]) == 2
    assert "simulate_ifg" in capfd.readouterr().err


def test_runtime_error_exit_code(tmp_path, capfd):
    code = cli_main(
        [
            "--phase",
            str(tmp_path / "missing.tif"),
            "--cor",
            str(tmp_path / "missing2.tif"),
            "--nlooks",
            "10",
            "--out",
            str(tmp_path / "out.tif"),
        ]
    )
    assert code == 1
    assert "Error" in capfd.readouterr().err


def test_console_script_main(monkeypatch, capfd):
    monkeypatch.setattr(sys, "argv", ["whirlwind", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert "Usage: whirlwind" in capfd.readouterr().out


def test_simulate_unwrap_roundtrip(tmp_path, capfd):
    import whirlwind as ww

    # Synthesize via the Python API (the old `simulate` subcommand's home) and
    # hand flat float32 inputs to the CLI: .f32 in and out are np.tofile /
    # np.fromfile-compatible, so no raster dependency is needed.
    truth = ww.diagonal_ramp(64, 64)
    gamma = np.full((64, 64), 0.85, dtype=np.float32)
    igram, cor = ww.simulate_ifg(truth, gamma, 10, 42)
    np.angle(igram).astype(np.float32).tofile(tmp_path / "wrapped.f32")
    np.clip(cor, 0.0, 1.0).astype(np.float32).tofile(tmp_path / "cor.f32")

    assert (
        cli_main(
            [
                "--phase",
                str(tmp_path / "wrapped.f32"),
                "--cols",
                "64",
                "--cor",
                str(tmp_path / "cor.f32"),
                "--nlooks",
                "10",
                "--out",
                str(tmp_path / "unw.f32"),
            ]
        )
        == 0
    )
    capfd.readouterr()  # drain progress chatter

    unw = np.fromfile(tmp_path / "unw.f32", dtype=np.float32).reshape(64, 64)
    assert np.isfinite(unw).all()
    assert unw.std() > 0.0, "unwrapped phase should not be constant"
