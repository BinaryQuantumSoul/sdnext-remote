import time

from modules import shared
from modules.scripts_postprocessing import PostprocessedImage

from extension.utils_remote import RemoteInferencePostprocessError, get_current_api_service, RemoteService, stable_horde_client, get_api_key, encode_image, get_image, request_or_error, imported_scripts

def remote_run(self, pp: PostprocessedImage, args):
    service = get_current_api_service()

    #================================== Stable Horde ==================================
    if service == RemoteService.StableHorde:      
        for script in self.scripts_in_preferred_order():
            process_args = {}
            for (name, _component), value in zip(script.controls.items(),  args[script.args_from:script.args_to]):
                process_args[name] = value

            if isinstance(script, imported_scripts['codeformer'].script_class):
                if(process_args["codeformer_visibility"] == 0):
                    continue
                form = 'CodeFormers'
            elif isinstance(script, imported_scripts['gfpgan'].script_class):
                if(process_args["gfpgan_visibility"] == 0):
                    continue
                form = 'GFPGAN'
            elif isinstance(script, imported_scripts['rembg'].script_class):
                if(process_args["model"] == 'None'):
                    continue
                form = 'strip_background'
            elif isinstance(script, imported_scripts['upscale'].script_class):
                if(process_args["upscaler_1_name"] == 'None'):
                    continue
                form = process_args["upscaler_1_name"]
            else:
                script_type = type(script).split('\'')[1]
                shared.log.warning(f"RI: {service} unable to do script of type {script_type}")
                continue
        
            headers = {
                "apikey": get_api_key(service),
                "Client-Agent": stable_horde_client,
                "Content-Type": "application/json"
            }
            payload = {
                "forms": [{"name": form}],
                "source_image": encode_image(pp.image),
                "slow_workers": shared.opts.horde_slow_workers
            }

            shared.state.job = service.name

            response = request_or_error(service, '/v2/interrogate/async', headers, method='POST', data=payload)
            uuid = response['id']

            while True:
                status = request_or_error(service, f'/v2/interrogate/status/{uuid}', headers)
                state = status['state']
                
                if state == 'done':
                    pp.image = get_image(status['forms'][0]['result'][form])
                    break
                time.sleep(7.5)