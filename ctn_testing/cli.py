"""CTN Testing CLI."""

import os
import subprocess
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="ctn-test")
def cli():
    """CTN Testing CLI - Run evaluations and browse results."""
    pass


@cli.command()
@click.argument(
    "results_dir",
    type=click.Path(exists=True),
    default="domains/constraint_adherence/results",
    required=False,
)
@click.option("--port", default=8501, help="Port for Streamlit server")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def browse(results_dir: str, port: int, no_browser: bool):
    """Launch results browser UI.

    RESULTS_DIR is the path to a folder containing test run directories.
    Each run directory should have a manifest.json file.

    Examples:

        ctn-test browse

        ctn-test browse domains/constraint_adherence/results

        ctn-test browse ./my-results --port 8502
    """
    app_path = Path(__file__).parent / "browser" / "app.py"

    if not app_path.exists():
        click.echo(f"Error: Browser app not found at {app_path}", err=True)
        sys.exit(1)

    click.echo("Starting CTN Results Browser...")
    click.echo(f"  Results dir: {results_dir}")
    click.echo(f"  Port: {port}")
    click.echo()

    # Set PYTHONPATH to include the project root so absolute imports work
    # when Streamlit runs app.py as a script
    env = os.environ.copy()
    project_root = str(Path(__file__).parent.parent)
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = project_root + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = project_root

    # Build streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
    ]

    if no_browser:
        cmd.extend(["--server.headless", "true"])

    # Pass results_dir as argument to the app
    cmd.extend(["--", results_dir])

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        click.echo("\nBrowser stopped.")
    except subprocess.CalledProcessError as e:
        click.echo(f"Error running browser: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--sdk-url", default=None, help="Override SDK server URL")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output (no per-item progress)")
def run(config_path: str, sdk_url: str | None, seed: int | None, quiet: bool):
    """Run an evaluation from a config file.

    CONFIG_PATH is the path to an evaluation config YAML file.

    Examples:

        ctn-test run domains/constraint_adherence/configs/phase1.yaml

        ctn-test run config.yaml --sdk-url http://localhost:9999

        ctn-test run config.yaml -q  # Quiet mode
    """
    from .runners.evaluation import ConstraintEvaluator, ProgressInfo

    click.echo(f"Running evaluation: {config_path}")
    click.echo()

    # Track errors for summary
    errors: list[tuple[str, str, str]] = []  # (constraint, prompt, error)
    run_success = 0
    run_failed = 0
    judge_success = 0
    judge_failed = 0
    current_stage = ""

    def truncate(text: str, max_len: int = 40) -> str:
        """Truncate text with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def progress_callback(info: ProgressInfo) -> None:
        nonlocal current_stage, run_success, run_failed, judge_success, judge_failed

        # Print stage header when switching
        if info.stage != current_stage:
            if current_stage:
                click.echo()  # Newline after previous stage
            if info.stage == "running":
                click.echo("Generating responses:")
            elif info.stage == "judging":
                click.echo("\nJudging responses:")
            current_stage = info.stage

        # Track stats
        if info.stage == "running":
            if info.success:
                run_success += 1
            else:
                run_failed += 1
                if info.constraint_name and info.prompt_text and info.error_msg:
                    errors.append((info.constraint_name, info.prompt_text, info.error_msg))
        else:
            if info.success:
                judge_success += 1
            else:
                judge_failed += 1

        if quiet:
            # Quiet mode: just show progress bar style
            status = click.style(".", fg="green") if info.success else click.style("x", fg="red")
            click.echo(status, nl=False)
            if info.current == info.total:
                click.echo()
            return

        # Detailed output
        index = f"[{info.current}/{info.total}]"
        timing = f"({info.duration_secs:.1f}s)" if info.duration_secs else ""

        if info.stage == "running":
            # Show constraint and prompt
            constraint = info.constraint_name or "unknown"
            if constraint == "baseline":
                constraint_display = "baseline"
            else:
                constraint_display = f"@{constraint}"
            prompt = truncate(info.prompt_text or "", 35)

            if info.success:
                status = click.style("OK", fg="green")
                click.echo(f"  {index} {constraint_display}: {prompt}... {status} {timing}")
            else:
                status = click.style("ERROR", fg="red")
                error_detail = truncate(info.error_msg or "unknown error", 40)
                click.echo(f"  {index} {constraint_display}: {prompt}... {status} ({error_detail})")

        elif info.stage == "judging":
            # Show baseline vs test
            baseline = info.baseline_constraint or "baseline"
            test = info.test_constraint or "unknown"
            prompt = truncate(info.prompt_text or "", 30)

            if info.success:
                status = click.style("OK", fg="green")
                click.echo(f"  {index} {baseline} vs @{test}: {prompt}... {status} {timing}")
            else:
                status = click.style("SKIP", fg="yellow")
                error_detail = truncate(info.error_msg or "skipped", 30)
                click.echo(
                    f"  {index} {baseline} vs @{test}: {prompt}... {status} ({error_detail})"
                )

    try:
        evaluator = ConstraintEvaluator(
            config_path=Path(config_path),
            sdk_base_url=sdk_url,
            random_seed=seed,
        )

        result = evaluator.run(progress_callback=progress_callback)

        # Summary
        click.echo()
        click.echo("Summary:")
        total_run = run_success + run_failed
        total_judge = judge_success + judge_failed
        click.echo(
            f"  Responses: {run_success}/{total_run} succeeded"
            + (f", {run_failed} errors" if run_failed else "")
        )
        if total_judge > 0:
            click.echo(
                f"  Judgments: {judge_success}/{total_judge} succeeded"
                + (f", {judge_failed} skipped" if judge_failed else "")
            )

        # Error details
        if errors:
            click.echo()
            click.echo(click.style("Errors:", fg="red"))
            for constraint, prompt, error in errors[:10]:  # Limit to 10
                click.echo(f"  - @{constraint}: {truncate(prompt, 30)} ({error})")
            if len(errors) > 10:
                click.echo(f"  ... and {len(errors) - 10} more errors")

        click.echo()
        if result.run_dir:
            click.echo(f"Results saved to: {result.run_dir}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("list")
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def list_runs(results_dir: str, fmt: str):
    """List all runs in a results directory.

    Examples:

        ctn-test list domains/constraint_adherence/results

        ctn-test list ./results --format json
    """
    import json

    from .browser.data import list_runs

    runs = list_runs(Path(results_dir))

    if not runs:
        click.echo("No runs found.")
        return

    if fmt == "json":
        data = [
            {
                "run_id": r.run_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "strategy": r.strategy,
                "config_name": r.config_name,
                "prompts_count": r.prompts_count,
                "constraints": r.constraints,
                "errors_count": len(r.errors),
            }
            for r in runs
        ]
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(f"Found {len(runs)} run(s):\n")
        for run in runs:
            strategy_str = f" [{run.strategy}]" if run.strategy else ""
            errors_str = click.style(f" ({len(run.errors)} errors)", fg="red") if run.errors else ""
            click.echo(f"  {run.run_id}{strategy_str}{errors_str}")
            click.echo(f"    Config: {run.config_name}")
            click.echo(f"    Prompts: {run.prompts_count}, Constraints: {len(run.constraints)}")
            click.echo()


@cli.command()
@click.argument("run_path", type=click.Path(exists=True))
def analyze(run_path: str):
    """Analyze a single run and show statistics.

    Examples:

        ctn-test analyze domains/constraint_adherence/results/2024-01-15T10-30-00
    """
    from .runners.evaluation import EvaluationResult
    from .statistics.constraint_analysis import format_report, full_analysis

    try:
        result = EvaluationResult.load(Path(run_path))

        if not result.comparisons:
            click.echo("No judging comparisons found in this run.")
            return

        analyses = full_analysis(result)
        report = format_report(analyses)
        click.echo(report)

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error analyzing run: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
