def validate_data(data):
    REQUIRED_COLUMNS = data.columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Required columns are present.")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    return True
