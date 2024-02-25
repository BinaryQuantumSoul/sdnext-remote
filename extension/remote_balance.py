import modules.shared

from extension.utils_remote import get_current_api_service, RemoteService, get_or_error_with_cache, RemoteInferenceAPIError, get_api_key, stable_horde_client, clear_cache

def get_remote_balance_html():
    service = get_current_api_service()
    if not service.has_key:
        return ''     

    balance = get_remote_balance(service)
    if balance is None:
        return f'<p>{service.name}</p><p>Get an API key <a href="{service.url}">here</a></p>'
    else: 
        return f'<p>{service.credits_name}:</p><p class="remote_inference_balance_count">{service.credits_symbol}{balance}</p>'    

def refresh_balance():
    service = get_current_api_service()
    if service == RemoteService.StableHorde:
        clear_cache(service, '/v2/find_user')
    elif service == RemoteService.OmniInfer:
        clear_cache(service, '/v3/user')

def get_remote_balance(service):
    cache_time = modules.shared.opts.remote_balance_cache_time

    try:
        if service == RemoteService.StableHorde:
            response = get_or_error_with_cache(service, '/v2/find_user', {"apikey": get_api_key(service), "Client-Agent": stable_horde_client}, cache_time)
            return int(response['kudos'])
        elif service == RemoteService.OmniInfer:
            response = get_or_error_with_cache(service, '/v3/user', {"X-Omni-Key": get_api_key(service)}, cache_time)
            return response['credit_balance']/10000.
    except RemoteInferenceAPIError:
        return None