from enum import Enum
import copy
import requests
import json
import time
import base64
import io
from PIL import Image
from multiprocessing.pool import ThreadPool
import itertools

import modules.shared
import modules.scripts

ModelType = Enum('ModelType', ['CHECKPOINT','LORA','EMBEDDING','HYPERNET','VAE','SAMPLER','UPSCALER','CONTROLNET'])

class RemoteService(Enum):
    Local = ()
    SDNext = (True, 'http://127.0.0.1:7860', False)
    StableHorde = (True, 'https://stablehorde.net/api', True, 'https://stablehorde.net/register', 'Kudos', '')
    OmniInfer = (True, 'https://api.omniinfer.io', True, 'https://www.omniinfer.io/dashboard/key', 'Credits', '$')

    def __init__(self, has_endpoint=False, default_endpoint=None, has_key=False, url=None, credits_name=None, credits_symbol=None):
        self.has_endpoint = has_endpoint
        self.default_endpoint = default_endpoint
        self.has_key = has_key
        self.url = url
        self.credits_name = credits_name
        self.credits_symbol = credits_symbol

def get_remote_endpoint(remote_service):
    return modules.shared.opts.data.get(f'remote_{remote_service.name.lower()}_api_endpoint', remote_service.default_endpoint)

def get_api_key(remote_service):
    return modules.shared.opts.data.get(f'remote_{remote_service.name.lower()}_api_key')

def get_current_api_service():
    return RemoteService[modules.shared.opts.data.get('remote_inference_service', RemoteService.Local.name)]  

def safeget(dct, *keys):
    for key in keys:
        try:
            dct = dct[key]
        except (KeyError,TypeError,IndexError):
            return None
    return dct

def make_conditional_hook(func, replace_func):
    func_copy = copy.deepcopy(func)
    def wrap(*args, **kwargs):
        if get_current_api_service() == RemoteService.Local:
            return func_copy(*args, **kwargs)
        else:
            return replace_func(*args, **kwargs)
    return wrap

imported_scripts = {}
def import_script_data(dict):
    def get_script_data(path):
        script = next((data for data in itertools.chain(modules.scripts.scripts_data, modules.scripts.postprocessing_scripts_data) if data.path.endswith(path)), None)
        if not script:
            raise ImportError(f"Script module not found: {path}")
        return script

    imported_scripts.update({name: get_script_data(path) for name,path in dict.items()})

class RemoteInferenceAPIError(Exception):
    def __init__(self, service, error):
        super().__init__(f'RI: error with {service} api call: {error}')

class RemoteInferenceProcessError(Exception):
    def __init__(self, service, error):
        super().__init__(f'RI: error with process task for {service}: {error}')

class RemoteInferencePostprocessError(Exception):
    def __init__(self, service, error):
        super().__init__(f'RI: error with postprocess task for {service}: {error}')

def get_payload_str(payload):
    def truncate(value, max_length=50):
        if isinstance(value, str):
            return value[:max_length]+'...' if len(value) > max_length else value
        elif isinstance(value, list):
            return [truncate(val) for val in value]
        elif isinstance(value, dict):
            return {key: truncate(val) for key,val in value.items()}
        else:
            return value
    return truncate(payload)

def clean_payload_dict(payload):
    def clean(value):
        if isinstance(value, list):
            return [clean(val) for val in value if val is not None]
        elif isinstance(value, dict):
            return {key: clean(val) for key, val in value.items() if val is not None}
        else:
            return value
    return clean(payload)

def request_or_error(service, path, headers=None, method='GET', data=None):
    try:
        data = clean_payload_dict(data)
        modules.shared.log.debug(f'RI: payload: {get_payload_str(data)}')
        response = requests.request(method=method, url=get_remote_endpoint(service)+path, headers=headers, json=data)
        modules.shared.log.debug(f'RI: response: {get_payload_str(json.loads(response.content))}')
    except Exception as e:
        raise RemoteInferenceAPIError(service, e)
    if response.status_code not in (200, 202):
        raise RemoteInferenceAPIError(service, f"{response.status_code}: {response.content}")
    
    return json.loads(response.content)

cache = {}
def get_or_error_with_cache(service, path, headers=None, cache_time=None):
    global cache
    cache_key = (service, path)
    if cache_key in cache:
        result, timestamp = cache[cache_key]
        if time.time() - timestamp <= (cache_time or modules.shared.opts.remote_extra_networks_cache_time):
            if isinstance(result, RemoteInferenceAPIError):
                raise result
            else:
                return result

    try:
        result = request_or_error(service, path, headers)
        cache[cache_key] = (result, time.time())
        return result
    except RemoteInferenceAPIError as e:
        cache[cache_key] = (e, time.time())
        raise e

def clear_cache(service, path):
    global cache
    cache.pop((service, path))

def download_image(img_url):
    attempts = 5
    while attempts > 0:
        try:
            response = requests.get(img_url, timeout=5)
            response.raise_for_status()
            with io.BytesIO(response.content) as fp:
                return Image.open(fp).copy()
        except (requests.RequestException, Image.UnidentifiedImageError):
            modules.shared.log.warning(f"RI: Failed to download {img_url}, retrying...")
            attempts -= 1
    return None

def download_images(img_urls, num_threads=10):
    with ThreadPool(num_threads) as pool:
        images = pool.map(download_image, img_urls)

    return list(filter(lambda img: img is not None, images))

def decode_image(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))

def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP")
    return base64.b64encode(buffer.getvalue()).decode()

def get_image(img):
    if img.startswith('http'):
        return download_image(img)
    else:
        return decode_image(img)
    
stable_horde_client = "SD.Next Remote Inference:rolling:QuantumSoul"

stable_horde_controlnets = ["canny", "hed", "depth", "normal", "openpose", "seg", "scribble", "fakescribbles", "hough"]

stable_horde_samplers =  {
    "LMS": "k_lms",
    "Heun": "k_heun",
    "Euler": "k_euler",
    "Euler a": "k_euler_a",
    "DPM2": "k_dpm_2",
    "DPM2 a": "k_dpm_2_a",
    "DPM fast": "k_dpm_fast",
    "DPM adaptive": "k_dpm_adaptive",
    "DPM++ 2S a": "k_dpmpp_2s_a",
    "DPM++ 2M": "k_dpmpp_2m",
    "DPM solver": "dpmsolver",
    "DPM++ SDE": "k_dpmpp_sde",
    "DDIM": "DDIM",
}