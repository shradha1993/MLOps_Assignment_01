import requests


def invoke_model_server(
    data,
    columns,
    url="http://127.0.0.1:5001/invocations",
):
    payload = {
        "dataframe_split": {
            "columns": columns,
            "data": data,
        }
    }

    response = requests.post(
        url,
        json=payload,
    )

    response.raise_for_status()

    return response.json()
