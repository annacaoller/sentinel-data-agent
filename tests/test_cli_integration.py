from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from sentinel.cli.main import app

runner = CliRunner()


def test_cli_run_and_status(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):

        examples_dir = Path("examples")
        examples_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = examples_dir / "example.csv"
        dataset_path.write_text(
            "id,value,category\n"
            "1,10,A\n"
            "2,20,B\n"
            "3,15,A\n"
            "4,30,C\n",
            encoding="utf-8",
        )

        result_run = runner.invoke(app, ["run", str(dataset_path)])

        assert result_run.exit_code == 0
        assert "Run created" in result_run.stdout

        match = re.search(r"run_id:\s*(\S+)", result_run.stdout)
        assert match is not None

        run_id = match.group(1)

        result_status = runner.invoke(app, ["status", run_id])

        assert result_status.exit_code == 0

        payload = json.loads(result_status.stdout)

        assert payload["run_id"] == run_id
        assert payload["status"] == "report_completed"
        assert payload["current_phase"] == "final_report"
        assert payload["progress_pct"] == 100