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

    click.echo(f"Starting CTN Results Browser...")
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
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
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
def run(config_path: str, sdk_url: str | None, seed: int | None):
    """Run an evaluation from a config file.

    CONFIG_PATH is the path to an evaluation config YAML file.

    Examples:

        ctn-test run domains/constraint_adherence/configs/phase1.yaml

        ctn-test run config.yaml --sdk-url http://localhost:9999
    """
    from .runners.evaluation import ConstraintEvaluator

    click.echo(f"Running evaluation: {config_path}")

    try:
        evaluator = ConstraintEvaluator(
            config_path=Path(config_path),
            sdk_base_url=sdk_url,
            random_seed=seed,
        )

        def progress_callback(stage: str, current: int, total: int, success: bool = True, error_msg: str | None = None):
            status = click.style("[ok]", fg="green") if success else click.style("[error]", fg="red")
            click.echo(f"\r{stage}: {current}/{total} {status}", nl=False)
            if current == total:
                click.echo()

        result = evaluator.run(progress_callback=progress_callback)

        click.echo()
        click.echo(f"Completed: {len(result.run_results)} responses, {len(result.comparisons)} comparisons")

        if result.run_dir:
            click.echo(f"Results saved to: {result.run_dir}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def list(results_dir: str, fmt: str):
    """List all runs in a results directory.

    Examples:

        ctn-test list domains/constraint_adherence/results

        ctn-test list ./results --format json
    """
    from .browser.data import list_runs
    import json

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
    from .statistics.constraint_analysis import full_analysis, format_report

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
