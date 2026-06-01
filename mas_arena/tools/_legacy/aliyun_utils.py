import configparser
import os
import json
import time
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

# Constants for token caching
TOKEN_FILE = ".aliyun_token_cache.json"
TOKEN_VALID_DURATION = 3500  # Aliyun tokens are valid for 1 hour (3600s)

def get_config():
    """Reads Aliyun credentials from config.ini."""
    config = configparser.ConfigParser()
    # Ensure the path is relative to this file's location
    config_path = os.path.join(os.path.dirname(__file__), '../../../config.ini')
    if not os.path.exists(config_path):
        raise FileNotFoundError("Error: config.ini not found. Please create it from config.ini.template.")
    config.read(config_path)
    return config['alibabacloud']

def get_aliyun_token():
    """
    Retrieves a valid Aliyun NLS token.
    It uses a local cache to avoid requesting a new token on every call.
    """
    # Check for a cached token
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            try:
                token_data = json.load(f)
                if time.time() - token_data.get('timestamp', 0) < TOKEN_VALID_DURATION:
                    return token_data.get('token')
            except json.JSONDecodeError:
                pass  # Invalid JSON in cache, will fetch a new one

    # If no valid cached token, fetch a new one
    config = get_config()
    client = AcsClient(
        config['access_key_id'],
        config['access_key_secret'],
        "cn-shanghai"  # Region does not matter for this token request
    )

    request = CommonRequest()
    request.set_method('POST')
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('CreateToken')

    try:
        response = client.do_action_with_exception(request)
        response_data = json.loads(response)
        if response_data.get('Token') and response_data['Token'].get('Id'):
            token = response_data['Token']['Id']
            # Cache the new token and timestamp
            with open(TOKEN_FILE, 'w') as f:
                json.dump({'token': token, 'timestamp': time.time()}, f)
            return token
        else:
            error_msg = response_data.get('Message', 'Unknown error from Aliyun CreateToken API')
            raise ConnectionError(f"Failed to get Aliyun token: {error_msg}")
    except Exception as e:
        raise ConnectionError(f"Exception when getting Aliyun token: {e}")

if __name__ == '__main__':
    # For testing purposes
    try:
        print("Attempting to get Aliyun token...")
        token = get_aliyun_token()
        print(f"Successfully retrieved token: {token[:10]}...")
    except (FileNotFoundError, ConnectionError) as e:
        print(e) 