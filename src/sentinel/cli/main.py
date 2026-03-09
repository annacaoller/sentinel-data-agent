from __future__ import annotations

import sys
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentinel.orchestrator.bootstrap import BootstrapOrchestrator
from sentinel.state.store import JsonRunStore

app = typer.Typer(
    name="sentinel",
    add_completion=False,
    help="Sentinel Data Agent: deterministic DAG-based autonomous data analysis with auditability.",
)

console = Console()
store = JsonRunStore()
bootstrap_orchestrator = BootstrapOrchestrator(store)


@app.command()
def run(
    dataset: str = typer.Argument(..., help="Path to dataset file (csv, parquet, etc)."),
    objective: str = typer.Option(
        "auto",
        "--objective",
        help="Optional analysis objective. 'auto' lets the agent infer a goal.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Allow overwriting an existing run directory if collision happens.",
        is_flag=True,
    ),
) -> None:
    """
    Start a new Sentinel run.
    """

    try:
        result = bootstrap_orchestrator.execute(dataset=dataset, objective=objective)

        console.print(
            Panel.fit(
                f"[bold green]Run created[/bold green]\n\n"
                f"run_id: [bold]{result.run_id}[/bold]\n"
                f"dataset: {result.profile['dataset_path']}\n"
                f"objective: {objective}\n"
                f"rows: {result.profile['num_rows']}\n"
                f"columns: {result.profile['num_columns']}\n"
                f"profile: {result.profile_path}\n"
                f"quality: {result.quality_path}\n"
                f"target inference: {result.target_inference_path}\n"
                f"schema: {result.schema_path}\n"
                f"execution plan: {result.execution_plan_path}\n"
                f"eda summary: {result.eda_summary_path}\n"
                f"correlation analysis: {result.correlation_analysis_path}\n"
                f"feature readiness: {result.feature_readiness_path}\n"
                f"baseline classification: {result.baseline_classification_path}\n"
                f"final report: {result.final_report_path}\n"
                f"checkpoint: {result.checkpoint_path}\n"
                f"selected target: {result.target_inference['selected_target']}\n"
                f"target confidence: {result.target_inference['confidence']}",
                title="Sentinel",
            )
        )

    except Exception as exc:
        console.print(f"[red]Run failed during bootstrap:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command()
def status(
    run_id: str = typer.Argument(..., help="Run identifier."),
) -> None:
    """
    Show run status and progress.
    """
    try:
        state = store.read_run_state(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(code=1)

    console.print_json(data=state)


@app.command("list")
def list_runs(
    limit: int = typer.Option(default=10, help="Number of runs to list."),
) -> None:
    """
    List recent runs.
    """
    runs = list(store.list_runs(limit=limit))

    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return

    table = Table(title="Sentinel Runs")
    table.add_column("run_id", style="bold")
    table.add_column("status")
    table.add_column("updated_at")
    table.add_column("dataset_path")

    for r in runs:
        table.add_row(
            r.get("run_id", "unknown"),
            r.get("status", "unknown"),
            r.get("updated_at", ""),
            r.get("dataset_path", ""),
        )

    console.print(table)


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run id to locate the report for."),
    open_in_browser: bool = typer.Option(
        False,
        "--open-in-browser",
        help="If true, attempts to open the HTML report in the default browser (if available).",
        is_flag=True,
    ),
) -> None:
    """
    Show report location for a run.
    """
    try:
        paths = store.expected_report_paths(run_id)
    except Exception:
        console.print(f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(code=1)

    console.print(f"Run artifacts: {paths['run_dir']}")

    final_report_json = paths["final_report_json"]
    report_html = paths["report_html"]
    report_json = paths["report_json"]

    if final_report_json.exists():
        console.print(f"[green]Final JSON report:[/green] {final_report_json}")
    else:
        console.print(f"[yellow]Final JSON report not found yet.[/yellow] Expected: {final_report_json}")

    if report_html.exists():
        console.print(f"[green]HTML report:[/green] {report_html}")
        if open_in_browser:
            try:
                import webbrowser

                webbrowser.open(report_html.as_uri())
            except Exception as exc:
                console.print(f"[yellow]Could not open browser:[/yellow] {exc}")
    else:
        console.print(f"[yellow]HTML report not found yet.[/yellow] Expected: {report_html}")

    if report_json.exists():
        console.print(f"[green]Legacy report JSON:[/green] {report_json}")
    else:
        console.print(f"[yellow]Legacy report JSON not found.[/yellow] Expected: {report_json}")


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run id to resume from latest checkpoint."),
) -> None:
    """
    Resume a run from the latest checkpoint.

    Current MVP behavior:
    - locate latest checkpoint
    - report its path
    - full resume execution logic will be implemented later
    """
    try:
        latest = store.latest_checkpoint_path(run_id)
    except Exception:
        console.print(f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(code=1)

    if latest is None:
        console.print("[yellow]No checkpoints found for this run.[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"Latest checkpoint: [bold]{latest}[/bold]")


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    app(prog_name="sentinel", args=argv)


if __name__ == "__main__":
    main()