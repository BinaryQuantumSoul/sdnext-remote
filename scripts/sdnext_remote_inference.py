import gradio as gr

import modules.sd_models
import modules.ui_extra_networks_checkpoints
import modules.textual_inversion.textual_inversion
import modules.ui_extra_networks_textual_inversion
import modules.processing
import modules.scripts_postprocessing
import modules.scripts
import modules.script_callbacks
import modules.shared
from modules.shared import OptionInfo, options_section 

from extension.utils_remote import make_conditional_hook, RemoteService, import_script_data
import extension.remote_extra_networks
import extension.remote_process
import extension.remote_balance
import extension.remote_postprocess
import extension.ui_bindings

import ui_extra_networks_lora
import networks

def on_app_started(blocks, _app):
    # SCRIPT IMPORTS
    import_script_data({
        'controlnet': 'extensions-builtin/sd-webui-controlnet/scripts/controlnet.py',
        'rembg': 'extensions-builtin/stable-diffusion-webui-rembg/scripts/postprocessing_rembg.py',
        'codeformer': 'scripts/postprocessing_codeformer.py',
        'gfpgan': 'scripts/postprocessing_gfpgan.py',
        'upscale': 'scripts/postprocessing_upscale.py'
    })

    # EXTRA NETWORKS
    modules.sd_models.list_models = make_conditional_hook(modules.sd_models.list_models, extension.remote_extra_networks.list_remote_models)
    modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items = make_conditional_hook(modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items, extension.remote_extra_networks.extra_networks_checkpoints_list_items)

    networks.list_available_networks = make_conditional_hook(networks.list_available_networks, extension.remote_extra_networks.list_remote_loras)
    ui_extra_networks_lora.ExtraNetworksPageLora.list_items = make_conditional_hook(ui_extra_networks_lora.ExtraNetworksPageLora.list_items, extension.remote_extra_networks.extra_networks_loras_list_items)

    modules.textual_inversion.textual_inversion.EmbeddingDatabase.load_textual_inversion_embeddings = make_conditional_hook(modules.textual_inversion.textual_inversion.EmbeddingDatabase.load_textual_inversion_embeddings, extension.remote_extra_networks.list_remote_embeddings)
    modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.refresh = make_conditional_hook(modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.refresh, extension.remote_extra_networks.extra_networks_textual_inversions_refresh)
    modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.list_items = make_conditional_hook(modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.list_items, extension.remote_extra_networks.extra_networks_textual_inversions_list_items)

    # GENERATION
    modules.sd_models.reload_model_weights = make_conditional_hook(modules.sd_models.reload_model_weights, extension.remote_process.fake_reload_model_weights)
    modules.processing.process_images = make_conditional_hook(modules.processing.process_images, extension.remote_process.process_images)
    modules.scripts_postprocessing.ScriptPostprocessingRunner.run = make_conditional_hook(modules.scripts_postprocessing.ScriptPostprocessingRunner.run, extension.remote_postprocess.remote_run) 

    # UI
    with blocks:
        gr.HTML(value='', visible=False, elem_id='remote_inference_balance', show_progress=False)
        gr.Button(value='', visible=False, elem_id='remote_inference_balance_click')

modules.script_callbacks.on_app_started(on_app_started)
modules.script_callbacks.after_process_callback(lambda p: extension.remote_balance.refresh_balance())
modules.script_callbacks.on_after_component(lambda component, **kwargs: extension.ui_bindings.bind_component(component))

# SETTINGS
def on_ui_settings():
    additional_settings = {
        RemoteService.ComfyICU : {
            'remote_comfyicu_workflow_id': OptionInfo('', f'ComfyICU workflow id')
        },
        RemoteService.StableHorde : {
            'remote_stablehorde_nsfw': OptionInfo(False, "Enable NSFW generation (will skip anti-nsfw workers)"),
            'remote_stablehorde_censor_nsfw': OptionInfo(False, "Censor NSFW generations"),
            'remote_stablehorde_trusted_workers': OptionInfo(False, "Only trusted workers (slower but less risk)"),
            'remote_stablehorde_slow_workers': OptionInfo(True, "Allow slow workers (extra kudos cost if disabled)"),
            'remote_stablehorde_workers': OptionInfo('', "Comma-separated list of allowed/disallowed workers (max 5)"),
            'remote_stablehorde_worker_blacklist': OptionInfo(False, "Above list is a blacklist instead of a whitelist"),
            'remote_stablehorde_share_laion': OptionInfo(False, 'Share images with LAION for improving their dataset, reduce your kudos consumption by 2 (always True for anonymous users)')
        }
    }

    settings = {}
    for service in RemoteService:
        name = service.name.lower()

        if not service.has_endpoint:
            continue

        settings[f'remote_{name}_sep'] = OptionInfo(f"<h2>{service.name}</h2>", "", gr.HTML)
        settings[f'remote_{name}_api_endpoint'] =  OptionInfo(service.default_endpoint, f'{service.name} API endpoint')

        if not service.has_key:
            continue

        settings[f'remote_{name}_api_key'] = OptionInfo('', f'{service.name} API Key', gr.Textbox, {"type": "password"})
        settings[f'remote_{name}_api_key_url'] = OptionInfo(f'<p>Get an API key <a href="{service.url}" target="_blank">here</a></p>', "", gr.HTML)

        if not service in additional_settings:
            continue

        settings.update(additional_settings[service])

    settings.update({
        'remote_general_sep': OptionInfo("<h2>Other Settings</h2>", "", gr.HTML),
        'remote_balance_cache_time': OptionInfo(300, 'Cache time (in seconds) for remote balance api calls', gr.Slider, {"minimum": 300, "maximum": 3600, "step": 60}),
        'remote_extra_networks_cache_time': OptionInfo(600, 'Cache time (in seconds) for remote extra networks api calls', gr.Slider, {"minimum": 60, "maximum": 3600, "step": 60}),
        'remote_show_balance_box': OptionInfo(True, "Show top right available balance box"),
        'remote_show_balance_quick': OptionInfo(True, "Show quicksettings available balance"),
        'remote_show_nsfw_models': OptionInfo(False, "Show NSFW networks (StableHorde/NovitaAI)"),

        'remote_inference_service': OptionInfo(RemoteService.Local.name, "Remote inference service", gr.Dropdown, {"choices": [e.name for e in RemoteService]}),
        'remote_balance': OptionInfo("", "", gr.HTML)
    })

    modules.shared.options_templates.update(options_section(('sdnext_remote_inference', "Remote Inference"), settings))

    if modules.shared.opts.quicksettings_list[0] != 'remote_inference_service':
        modules.shared.opts.quicksettings_list.insert(0, 'remote_inference_service')
    if modules.shared.opts.quicksettings_list[1] != 'remote_balance':
        modules.shared.opts.quicksettings_list.insert(1, 'remote_balance') 

modules.script_callbacks.on_ui_settings(on_ui_settings)