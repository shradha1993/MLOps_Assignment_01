import requests


def check_model_health(
    url="http://127.0.0.1:5001/ping",
):
    response = requests.get(url)
    response.raise_for_status()

    print("Model server is healthy.")
    return True
