import json
import time
import math
from PIL import Image
import re

import modules.processing
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, Processed
import modules.shared
from modules.shared import state, log, opts
import modules.images

from extension.utils_remote import encode_image, decode_image, download_images, get_current_api_service, get_image, request_or_error, RemoteService, RemoteInferenceProcessError, imported_scripts, ModelType
from extension.remote_extra_networks import get_models

class RemoteModel:
    def __init__(self, checkpoint_info):
        self.sd_checkpoint_info = checkpoint_info
        self.sd_model_hash = ''
        self.is_sdxl = False

def fake_reload_model_weights(sd_model=None, info=None, reuse_dict=False, op='model'):
    try:
        checkpoint_info = info or modules.sd_models.select_checkpoint(op=op)
        model = RemoteModel(checkpoint_info)

        opts.sd_model_checkpoint = checkpoint_info.title
        modules.shared.sd_model = model
        return model
    except (StopIteration, AttributeError):
        log.warning("RI: Unable to load model, try refreshing model list")
        return None

def build_payload(service: RemoteService, p: StableDiffusionProcessing):
    txt2img = isinstance(p, StableDiffusionProcessingTxt2Img)
    img2img = isinstance(p, StableDiffusionProcessingImg2Img)
    if not txt2img and not img2img:
        raise TypeError("Neither txt2img nor img2img")

    if imported_scripts['controlnet']:
        control_units = [u for u in p.script_args if isinstance(u, imported_scripts['controlnet'].module.UiControlNetUnit) and u.enabled]
    else:
        control_units = []

    model = modules.shared.sd_model.sd_checkpoint_info.filename

    def get_loras_tis(prompt, negative_prompt, substract_lora, substract_ti):
        pattern_lora = r'<lora:([^:\s]+):([^:\s]+)>'
        pattern_embedding = r'embedding:([^\s]+)'

        pos_loras = re.findall(pattern_lora, prompt)
        neg_loras = re.findall(pattern_lora, negative_prompt)
        loras = [(i,float(j)) for i,j in pos_loras] + [(i,-float(j)) for i,j in neg_loras]
        tis = re.findall(r'embedding:(\w+)', prompt+' '+negative_prompt)

        if substract_lora:
            prompt = re.sub(pattern_lora, '', prompt)
            negative_prompt = re.sub(pattern_lora, '', negative_prompt)

        if substract_ti:
            prompt = re.sub(pattern_embedding, '', prompt)
            negative_prompt = re.sub(pattern_embedding, '', negative_prompt)

        return prompt, negative_prompt, loras, tis

    #================================== SD.Next ==================================
    if service == RemoteService.SDNext:
        p.prompt, p.negative_prompt, loras, tis = get_loras_tis(p.prompt, p.negative_prompt, False, True)

        txt2img_keys = ["enable_hr", "denoising_strength", "firstphase_width", "firstphase_height", "hr_scale", "hr_force", "hr_upscaler", "hr_second_pass_steps", "hr_resize_x", "hr_resize_y", "refiner_steps", "refiner_start", "refiner_prompt", "refiner_negative", "prompt", "styles", "seed", "subseed", "subseed_strength", "seed_resize_from_h", "seed_resize_from_w", "sampler_name", "latent_sampler", "batch_size", "n_iter", "steps", "cfg_scale", "image_cfg_scale", "clip_skip", "width", "height", "full_quality", "restore_faces", "tiling", "do_not_save_samples", "do_not_save_grid", "negative_prompt", "eta", "diffusers_guidance_rescale", "override_settings", "override_settings_restore_afterwards", "sampler_index", "script_name", "send_images", "save_images", "alwayson_scripts"] #"script_args", "sampler_index"
        img2img_keys = ["init_images", "resize_mode", "denoising_strength", "image_cfg_scale", "mask", "mask_blur", "inpainting_fill", "inpaint_full_res", "inpaint_full_res_padding", "inpainting_mask_invert", "initial_noise_multiplier", "refiner_steps", "refiner_start", "refiner_prompt", "refiner_negative", "prompt", "styles", "seed", "subseed", "subseed_strength", "seed_resize_from_h", "seed_resize_from_w", "sampler_name", "latent_sampler", "batch_size", "n_iter", "steps", "cfg_scale", "clip_skip", "width", "height", "full_quality", "restore_faces", "tiling", "do_not_save_samples", "do_not_save_grid", "negative_prompt", "eta", "diffusers_guidance_rescale", "override_settings", "override_settings_restore_afterwards", "sampler_index", "include_init_images", "script_name", "send_images", "save_images", "alwayson_scripts"] #"script_args", "sampler_index"

        data = vars(p)
        payload = {key: data[key] for key in (txt2img_keys if txt2img else img2img_keys) if key in data.keys() and data[key] is not None}
        if img2img:
            payload['init_images'] = [encode_image(img) for img in payload['init_images']]
        
        return payload
    

    #================================== ComfyUI & ComfyICU ==================================
    elif service in [RemoteService.ComfyUI, RemoteService.ComfyICU]:
        p.prompt, p.negative_prompt, loras, tis = get_loras_tis(p.prompt, p.negative_prompt, True, False)

        with open('extensions/sdnext-remote/workflows/txt2img.json') as f:
            prompt = json.loads("\n".join(f.readlines()))

        prompt["4"]["inputs"]["ckpt_name"] = model
        prompt["6"]["inputs"]["text"] = p.prompt
        payload = {"prompt": prompt}

        return payload


    #================================== StableHorde ==================================
    elif service ==  RemoteService.StableHorde:
        p.prompt, p.negative_prompt, loras, tis = get_loras_tis(p.prompt, p.negative_prompt, True, False)

        stable_horde_controlnets = get_models(ModelType.CONTROLNET, service)
        stable_horde_samplers = get_models(ModelType.SAMPLER, service)

        n = p.n_iter*p.batch_size

        if txt2img and len(control_units) > 1:
            raise RemoteInferenceProcessError(service, 'Max 1 controlnet')
        elif img2img and len(control_units) > 0:
            raise RemoteInferenceProcessError(service, 'No controlnet for img2img')
        if len(loras) > 5:
            raise RemoteInferenceProcessError(service, 'Max 5 loras')

        payload = {
            "prompt": f"{p.prompt}###{p.negative_prompt}" if p.negative_prompt else p.prompt,
            "params": {
                "sampler_name": stable_horde_samplers[p.sampler_name],
                "cfg_scale": p.cfg_scale,
                "denoising_strength": p.denoising_strength,
                "seed": str(p.seed),
                "height": p.height,
                "width": p.width,
                "seed_variation": 1000,
                # post_processing
                "karras": opts.schedulers_sigma == 'karras',
                "tiling": p.tiling,
                "hires_fix": p.enable_hr,
                "clip_skip": p.clip_skip,
                # control_type (later)
                # image_is_control
                # return_control_map
                # facefixer_strength
                "loras": [{"name": i, "model": j} for i,j in loras],
                "tis": [{"name": i} for i in tis],
                # special
                "steps": p.steps,
                "n": n
            },
            "nsfw": opts.remote_stablehorde_nsfw,
            "trusted_workers": opts.remote_stablehorde_trusted_workers,
            "slow_workers": opts.remote_stablehorde_slow_workers,
            "censor_nsfw": opts.remote_stablehorde_censor_nsfw,
            "workers": opts.remote_stablehorde_workers.split(',') if len(opts.remote_stablehorde_workers) > 0 else [],
            "worker_blacklist": opts.remote_stablehorde_worker_blacklist,
            "models": [model],
            # source_image (later)
            # source_processing
            # source_mask
            # r2
            "shared": opts.remote_stablehorde_share_laion
            # replacement_filter
            # dry_run
            # proxied_account
            # disable_batching
            # webhook
        }

        if txt2img:
            if control_units: 
                unit = control_units[0]
                control_type = next((i for i in unit.module.split('_') if i in stable_horde_controlnets), None)
                if not control_type:
                    raise RemoteInferenceProcessError(service, f'ControlNet type should be in {stable_horde_controlnets}')
                payload["params"]["control_type"] = control_type
                payload["params"]["control_strength"] = unit.weight
                if not p.enable_hr:
                    payload["params"]["denoising_strength"] = unit.weight
                payload["source_image"] = encode_image(Image.fromarray(unit.image['image']))
        elif img2img:
            payload["source_image"] = encode_image(p.init_images[0])
            if p.image_mask:
                payload["source_processing"] = "inpainting"
                payload["source_mask"] = encode_image(p.image_mask)

        return payload


    #================================== NovitaAI ==================================
    elif service == RemoteService.NovitaAI:
        p.prompt, p.negative_prompt, loras, tis = get_loras_tis(p.prompt, p.negative_prompt, True, True)

        n = p.n_iter*p.batch_size

        payload = {
            "request": {
                "model_name": model,
                #image_base64 (later)
                "prompt": p.prompt,
                "negative_prompt": p.negative_prompt,
                #sd_vae
                "loras": [{"model_name": i, "strength": j} for (i,j) in loras],
                "embeddings": [{"model_name": i} for i in tis],
                #controlnet (later)
                #hires_fix (later)
                #refiner (later)
                "width": p.width,
                "height": p.height,
                "image_num": n,
                "steps": p.steps,
                "seed": p.seed,
                "clip_skip": p.clip_skip,
                "guidance_scale": p.cfg_scale,
                "sampler_name": p.sampler_name + ' Karras' if opts.schedulers_sigma == 'karras' else p.sampler_name
                #strength (later)
            }
        }

        if txt2img:
            if p.enable_hr:
                if p.hr_scale:
                    p.hr_resize_x = p.width*p.hr_scale
                    p.hr_resize_y = p.height*p.hr_scale

                payload["request"].update({
                    "hires_fix": {
                        "target_width": p.hr_resize_x,
                        "target_height": p.hr_resize_y,
                        "strength": p.denoising_strength,
                        "upscaler": p.hr_upscaler
                    },
                    "refiner": {
                        "switch_at": p.refiner_start
                    }
                })
        elif img2img:
            payload["request"].update({
                "image_base64": encode_image(p.init_images[0], format="PNG"),
                "strength": p.denoising_strength,
                "controlnet": {
                    "units": [{
                        "model_name": unit.model,
                        "image_base64": encode_image(unit.image, format="PNG"),
                        "strength": unit.weight,
                        "preprocessor": unit.module,
                        "guidance_start": unit.guidance_start,
                        "guidance_end": unit.guidance_end
                    } for unit in control_units]
                }
            })

        return payload

def generate_images(service: RemoteService, p: StableDiffusionProcessing) -> Processed:
    p.seed = int(modules.processing.get_fixed_seed(p.seed))
    p.subseed = int(modules.processing.get_fixed_seed(p.subseed))
    p.prompt = modules.shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = modules.shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)

    payload = build_payload(service, p)
    txt2img = isinstance(p, StableDiffusionProcessingTxt2Img)

    #================================== SD.Next ==================================
    if service == RemoteService.SDNext:
        processed_keys = ["seed", "info", "subseed", "all_prompts", "all_negative_prompts", "all_seeds", "all_subseeds", "index_of_first_image", "infotexts", "comments"]

        request_or_error(service, '/sdapi/v1/options', method='POST', data={'sd_model_checkpoint': opts.sd_model_checkpoint})
        request_or_error(service, '/sdapi/v1/reload-checkpoint', method='POST')
        response = request_or_error(service, ('/sdapi/v1/txt2img' if txt2img else '/sdapi/v1/img2img'), method='POST', data=payload)

        info = json.loads(response['info'])
        info = {key: info[key] for key in processed_keys if key in info.keys()}
        return Processed(p=p, images_list=[decode_image(img) for img in response['images']], **info)


    #================================== ComfyUI ==================================
    elif service == RemoteService.ComfyUI:
        response = request_or_error(service, '/prompt', method='POST', data=payload)
        print(vars(response))
    

    #================================== ComfyICU ==================================
    elif service == RemoteService.ComfyICU:
        workflow_id = opts.remote_comfyicu_workflow_id
        response = request_or_error(service, f'/v1/workflows/{workflow_id}/runs', method='POST', data=payload)
        uuid = response['id']
        
        state.sampling_steps = 100
        state.sampling_step = 0
        start = time.time()

        while True:
            response = request_or_error(service, f'/v1/workflows/{workflow_id}/runs/{uuid}')

            elapsed = int(time.time() - start)
            state.sampling_steps = elapsed + 120
            state.sampling_step = elapsed
            
            if response['status'] == 'COMPLETED':
                state.sampling_step = state.sampling_steps
                state.textinfo = "Downloading images..."
                images = download_images(img['url'] for img in response['output'])
                if len(images) == 0:
                    raise RemoteInferenceProcessError(service, 'Generation failed, no output image')
                return processed_from_images(p, images)
            
            elif response['status'] == 'ERROR':
                error_message = response['output']['error']['exception_message']
                raise RemoteInferenceProcessError(service, f'Generation failed with error {error_message}')
            
            time.sleep(5)


    #================================== StableHorde ==================================
    elif service ==  RemoteService.StableHorde:
        response = request_or_error(service, '/v2/generate/async', method='POST', data=payload)
        uuid = response['id']
        
        state.sampling_steps = 100
        state.sampling_step = 0
        start = time.time()

        while True:
            status = request_or_error(service, f'/v2/generate/check/{uuid}')

            elapsed = int(time.time() - start)
            state.sampling_steps = elapsed + status["wait_time"]
            state.sampling_step = elapsed
            
            if status['done']:
                state.sampling_step = state.sampling_steps
                state.textinfo = "Downloading images..."
                response = request_or_error(service, f'/v2/generate/status/{uuid}')
                images = [get_image(generation['img']) for generation in response['generations']]
                return processed_from_images(p, images)

            elif status['faulted']:
                raise RemoteInferenceProcessError(service, 'Generation failed')
            elif not status['is_possible']:
                raise RemoteInferenceProcessError(service, 'Generation not possible with current worker pool')
            
            time.sleep(5)


    #================================== NovitaAI ==================================
    elif service == RemoteService.NovitaAI:
        response = request_or_error(service, ('/v3/async/txt2img' if txt2img else '/v3/async/img2img'), method='POST', data=payload)
        uuid = response['task_id']

        while True:
            response = request_or_error(service, f'/v3/async/task-result?task_id={uuid}')

            if response['task']['status'] == 'TASK_STATUS_PROCESSING':
                state.sampling_steps = 100
                state.sampling_step = response['task']['progress_percent']

            elif response['task']['status'] == 'TASK_STATUS_SUCCEED':
                state.sampling_step = 100
                state.textinfo = "Downloading images..."
                images = download_images(img['image_url'] for img in response['images'])
                return processed_from_images(p, images)

            elif response['task']['status'] == 'TASK_STATUS_FAILED':
                reason = response['task']['reason']
                raise RemoteInferenceProcessError(service, f'Generation failed: {reason}')
            
            time.sleep(5)

def processed_from_images(p, images):
    n = len(images)
    all_seeds=[p.seed + i*1000 for i in range(n)]
    all_subseeds=n*[1000]
    all_prompts=n*[p.prompt]
    all_negative_prompts=n*[p.negative_prompt]
    infotexts = n*[modules.processing.create_infotext(p, all_prompts, all_seeds, all_subseeds, all_negative_prompts=all_negative_prompts)]
    
    return Processed(
        p=p, 
        images_list=images,
        seed=p.seed,
        subseed=p.subseed,
        all_seeds=all_seeds,
        all_subseeds=all_subseeds,
        all_prompts=all_prompts,
        all_negative_prompts=all_negative_prompts,
        infotexts=infotexts
    )

def save_images_and_add_grid(proc: Processed, p:StableDiffusionProcessing):
    if opts.save and not p.do_not_save_samples:
        for i,img in enumerate(proc.images):
            modules.images.save_image(img, path=p.outpath_samples, basename="", seed=proc.all_seeds[i], prompt=proc.all_prompts[i], extension=opts.samples_format, info=proc.infotexts[i], p=p)

    if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and len(proc.images) >= 2:
        grid = modules.images.image_grid(proc.images, rows=math.ceil(math.sqrt(len(proc.images))))
        info = '\n'.join(proc.infotexts)

        if opts.return_grid:
            proc.infotexts.insert(0, info)
            proc.images.insert(0, grid)
            proc.index_of_first_image = 1

        if opts.grid_save:
            modules.images.save_image(grid, p.outpath_grids, "grid", proc.all_seeds[0], proc.all_prompts[0], opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, p=p, grid=True)

    return proc

def process_images(p: StableDiffusionProcessing) -> Processed:
    remote_service = get_current_api_service()

    state.begin()
    state.textinfo = f"Remote inference from {remote_service}"
    proc = generate_images(remote_service, p)
    proc = save_images_and_add_grid(proc, p)
    state.end()
    return proc