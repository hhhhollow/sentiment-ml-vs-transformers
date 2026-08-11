from __future__ import annotations

import json

from sentiment_benchmark.cli import main


def test_download_cli_reports_verified_hash(monkeypatch, tmp_path, capsys) -> None:
    output = tmp_path / "archive.zip"
    output.write_bytes(b"fixture")
    monkeypatch.setattr("sentiment_benchmark.cli.download_dataset", lambda path, force: output)

    assert main(["download", "--output", str(output)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output)
    assert len(payload["sha256"]) == 64


def test_run_all_cli_delegates_full_benchmark(monkeypatch, capsys) -> None:
    observed = {}

    def fake_run(paths, config, force_download, device):
        observed.update(
            fast=config.fast,
            force_download=force_download,
            device=device,
        )
        return {"champion_model": "fixture"}

    monkeypatch.setattr("sentiment_benchmark.experiment.run_benchmark", fake_run)

    assert main(["run-all", "--fast", "--force-download", "--device", "cpu"]) == 0

    assert observed == {"fast": True, "force_download": True, "device": "cpu"}
    assert json.loads(capsys.readouterr().out)["champion_model"] == "fixture"
