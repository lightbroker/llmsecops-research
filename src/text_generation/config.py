import os


def get_api_url():
    host = os.environ.get("API_HOST", "localhost")
    port = 9999 if host == "localhost" else 80
    return f"http://{host}:{port}"
