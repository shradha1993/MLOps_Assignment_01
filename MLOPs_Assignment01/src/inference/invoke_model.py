import requests


def invoke_model_server(
    data,
    columns=None,
    url="http://127.0.0.1:5001/predict",
):
    """
    Invokes the deployed FastAPI model prediction endpoint.
    Accepts data as a list of feature lists: [[val1, val2, ...]]
    """
    payload = {"inputs": data}

    response = requests.post(
        url,
        json=payload,
    )
    response.raise_for_status()
    return response.json()
