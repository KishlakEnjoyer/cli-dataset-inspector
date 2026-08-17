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
        dup_table = Table(title="Dataset Duplicates")
        dup_table.add_column("Count", justify="right")

        dup_table.add_row(
            str(self.data['duplicates'])
        )

        console.print(dup_table)

    def print_missing_table(self):
        missing_table = Table(title="Missing data")

        missing_table.add_column("Column")
        missing_table.add_column("Missing data", justify="right")

        missing = self.data["missing"]

        for column, value in missing.items():
            missing_table.add_row(
                str(column),
                str(value),
            )

        console.print(missing_table)

    def print_memory_table(self):
            memory_table = Table(title="Memory usage")
            memory_table.add_column("", justify="right")
    
            memory_table.add_row(
                str(self.data['memory'])
            )
    
            console.print(memory_table)