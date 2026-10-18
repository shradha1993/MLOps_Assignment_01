import os
from ucimlrepo import fetch_ucirepo

def download_uci_dataset(dataset_id: int, save_path: str):
    """
    Fetches a UCI dataset by ID and saves features, targets, metadata,
    and variable info to the specified directory.

    Parameters
    ----------
    dataset_id : int
        The UCI ML Repository dataset ID.
    save_path : str
        Directory path where the dataset files should be saved.

    Returns
    -------
    X, y : pandas.DataFrame
        Features and target dataframes.
    """

    # Fetch dataset
    dataset = fetch_ucirepo(id=dataset_id)

    # Extract components
    X = dataset.data.features
    y = dataset.data.targets
    metadata = dataset.metadata
    variables = dataset.variables

    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save dataframes
    X.to_csv(os.path.join(save_path, "features.csv"), index=False)
    y.to_csv(os.path.join(save_path, "targets.csv"), index=False)

    # Save metadata
    with open(os.path.join(save_path, "metadata.txt"), "w") as f:
        f.write(str(metadata))

    # Save variable info
    with open(os.path.join(save_path, "variables.txt"), "w") as f:
        f.write(str(variables))

    print(f"Dataset {dataset_id} downloaded successfully to: {save_path}")
    return X, y
