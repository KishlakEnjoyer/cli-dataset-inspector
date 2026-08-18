from rich.console import Console
from rich.table import Table

import pandas as pd

console = Console()

class CliReport:
    def __init__(self, data: dict):
        self.data = data

    def print_shape_table(self):
        shape_table = Table(title="Dataset Shape")
        shape_table.add_column("Rows", justify="right")
        shape_table.add_column("Columns", justify="right")

        shape_table.add_row(
            str(self.data['shape']['rows']),
            str(self.data['shape']['columns'])
        )

        console.print(shape_table)

    def print_duplicates_table(self):
        dup_table = Table(title="Duplicates")

        dup_table.add_column("Dataset duplicates", justify="right")
        dup_table.add_column("Duplicates percent", justify="right")

        dup_table.add_row(
            str(self.data["duplicates"]["count"]),
            f'{self.data["duplicates"]["percent"]:.2f}%'
        )

        console.print(dup_table)

    

    def print_memory_table(self):
        memory_table = Table()
        memory_table.add_column("Memory usage", justify="right")

        memory_mb = self.data["memory"] / (1024 ** 2)

        memory_table.add_row(
            f"{memory_mb:.2f} MB"
        )

        console.print(memory_table)

    def print_common_info(self):
        console.print("Common information")

        self.print_shape_table()
        self.print_memory_table()
        self.print_duplicates_table()

    def print_columns_summary_table(self):
        table = Table(title="Columns summary")

        table.add_column("Column")
        table.add_column("Type")
        table.add_column("Filled", justify="right")
        table.add_column("Missing", justify="right")
        table.add_column("Missing %", justify="right")
        table.add_column("Unique", justify="right")
        table.add_column("Unique %", justify="right")

        for column in self.data["columns_summary"]:
            table.add_row(
                column["name"],
                column["dtype"],
                str(column["filled"]),
                str(column["missing"]),
                f'{column["missing_percent"]:.2f}%',
                str(column["unique"]),
                f'{column["unique_percent"]:.2f}%',
            )

        console.print(table)

    def print_numeric_summary_table(self):
        table = Table(title="Numeric columns")

        table.add_column("Column")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Outliers", justify="right")

        for column in self.data["numeric_summary"]:
            table.add_row(
                column["name"],
                f'{column["min"]:.2f}',
                f'{column["max"]:.2f}',
                f'{column["mean"]:.2f}',
                f'{column["median"]:.2f}',
                f'{column["std"]:.2f}',
                str(column["outliers"]),
            )

        console.print(table)

    def print_categorical_summary_table(self):
        table = Table(title="Categorical columns")

        table.add_column("Column")
        table.add_column("Categories", justify="right")
        table.add_column("Most frequent")
        table.add_column("Count", justify="right")
        table.add_column("Top values")

        for column in self.data["categorical_summary"]:

            top_values = ", ".join(
                f"{value}: {count}"
                for value, count in column["top_values"].items()
            )

            table.add_row(
                column["name"],
                str(column["categories"]),
                column["most_frequent"],
                str(column["most_frequent_count"]),
                top_values,
            )

        console.print(table)

    def print_target_table(self):
        target = self.data.get("target")

        if target is None:
            return

        console.print(
            f"\n[bold]Target:[/bold] {target['name']}"
        )

        console.print(
            f"[bold]Task:[/bold] {target['task']}"
        )

        console.print(
            f"[bold]Missing:[/bold] "
            f"{target['missing']} "
            f"({target['missing_percent']:.2f}%)"
        )

        if target["task"] == "classification":
            table = Table(title="Target distribution")

            table.add_column("Class")
            table.add_column("Count", justify="right")
            table.add_column("Percent", justify="right")

            for item in target["distribution"]:
                table.add_row(
                    item["class"],
                    str(item["count"]),
                    f'{item["percent"]:.2f}%',
                )

            console.print(table)
        else:
            stats = target["statistics"]

            table = Table(title="Target statistics")

            table.add_column("Min", justify="right")
            table.add_column("Max", justify="right")
            table.add_column("Mean", justify="right")
            table.add_column("Median", justify="right")
            table.add_column("Std", justify="right")

            table.add_row(
                f'{stats["min"]:.2f}',
                f'{stats["max"]:.2f}',
                f'{stats["mean"]:.2f}',
                f'{stats["median"]:.2f}',
                f'{stats["std"]:.2f}',
            )

            console.print(table)

    def print_warnings(self):
        warnings = self.data["warnings"]

        if not warnings:
            console.print(
                "\n[green]✓ No obvious dataset problems found.[/green]"
            )
            return

        table = Table(title="Problems & Warnings")

        table.add_column("Type")
        table.add_column("Column")
        table.add_column("Description")

        for warning in warnings:
            table.add_row(
                warning["type"],
                warning["column"] or "-",
                warning["message"],
            )

        console.print(table)

    def print_pipeline(self):
        self.print_common_info()
        self.print_columns_summary_table()
        self.print_numeric_summary_table()
        self.print_categorical_summary_table()
        self.print_target_table()
        self.print_warnings()

    