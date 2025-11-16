import requests
import xml.etree.ElementTree as ET

def list_s3_folder(base_url: str, prefix: str):
    """
    List all objects under a public S3 prefix by parsing its XML index.
    Works for S3 static website endpoints like mosaicfmri.s3.amazonaws.com.
    """
    # S3 bucket listing URL
    list_url = f"{base_url}?prefix={prefix}"

    resp = requests.get(list_url)
    resp.raise_for_status()

    # Parse the XML list
    root = ET.fromstring(resp.text)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    keys = []
    for contents in root.findall("s3:Contents", namespace):
        key = contents.find("s3:Key", namespace).text
        # Skip directory placeholders
        if not key.endswith("/"):
            keys.append(key)

    return keys