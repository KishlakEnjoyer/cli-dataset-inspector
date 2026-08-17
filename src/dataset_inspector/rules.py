def check_missing(column_stats):
    if column_stats.missing_percent > 50:
        return Warning(
            column=column_stats.name,
            message="More than 50% of values are missing",
        )