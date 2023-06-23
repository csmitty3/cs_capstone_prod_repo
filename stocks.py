import json
import requests

def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = requests.get(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

api_key= 'dee9e143b1d0b3ce72ab2bf088fbfab9'
url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/AAPL?apikey={api_key}")
print(get_jsonparsed_data(url))

