import typer 

from typing import Annotated
from importlib.metadata import version

from .inspector import Inspector
from .report import CliReport

app = typer.Typer()

def version_callback(value: bool):
    if value:
        typer.echo(
            f"CLI Dataset Inspector {version('cli-dataset-inspector')}"
        )
        raise typer.Exit()

@app.command()
def inspect_dataset(
    path: Annotated[
        str,
        typer.Argument(help="Path to the dataset")
    ],
    target: Annotated[
        str,
        typer.Option(
            "--target",
            help="Name of the target"
        )
    ] = None,
    task: Annotated[
        str,
        typer.Option(
            "--task",
            help="Name of task: Classification ot regression"
        )
    ] = None,
    jupyter: Annotated[
        bool,
        typer.Option(
            "--jupyter",
            "-j",
            help="Append a jupyter notebook"
        )
    ] = False,
    baseline: Annotated[
        bool,
        typer.Option(
            "--baseline",
            "-bl",
            help="Learn a baseline model for task"
        )
    ] = False,
    json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Format JSON"
        )
    ] = False,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-op",
            help="Path for output notebook"
        )
    ] = None,
    show_version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
):
    """
    Function for inspecting the dataset
    """
    inspector = Inspector(
        path=path,
        target=target,
        task=task,
        jupyter=jupyter,
        baseline=baseline,
        json=json,
        output=output
    )

    try:
        data = inspector.inspect()
        cli_reporter = CliReport(data=data)

        cli_reporter.print_shape_table()
        cli_reporter.print_duplicates_table()
        cli_reporter.print_missing_table()
        cli_reporter.print_memory_table()

    except OSError as error:
        typer.echo(
            f"✗ Failed to inspect: {error}",
            err=True,
        )

        raise typer.Exit(code=1)

    typer.echo(
        f"✓ Dataset inspected!"
    )


def main():
    app()