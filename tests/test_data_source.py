from mosaic.constants import BASE_URL
import requests

def test_base_url_availability():
    response = requests.get(BASE_URL)
    assert response.status_code == 200, (
        f"Failed to reach Base URL {BASE_URL}. "
        f"Received status code: {response.status_code}. "
        f"Response text: {response.text[:200]}"
    )

    