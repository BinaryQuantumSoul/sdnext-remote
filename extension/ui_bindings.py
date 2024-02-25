import gradio as gr

import modules.shared

from extension.utils_remote import RemoteService, ModelType
from extension.remote_extra_networks import get_models
from extension.remote_balance import get_remote_balance_html

class UIBinding:
    COMPONENTS_IDS = [
        'remote_inference_balance_click', 'remote_inference_balance', 'setting_remote_balance',
        'setting_remote_inference_service', 'txt2img_sampling', 'img2img_sampling', 'extras_upscaler_1', 'txt2img_generate'
    ]

    def __init__(self):
        self.initialized = False
        self.components = {key: None for key in UIBinding.COMPONENTS_IDS}
        self.components_default = {}

    def __getattribute__(self, attr):
        if attr in UIBinding.COMPONENTS_IDS:
            return self.components[attr]
        return super().__getattribute__(attr)
    
    def add_component(self, component):
        if component.elem_id in self.components:
            self.components[component.elem_id] = component
            self.components_default[component.elem_id] = {attr: getattr(component, attr) for attr in ['value', 'choices'] if hasattr(component, attr)}

    def ready_for_binding(self):
        if not self.initialized and all(self.components.values()):
            self.initialized = True
            return True
        return False
    
    def back_to_default(self, components):
        return tuple(gr.update(**self.components_default[component.elem_id]) for component in components)
    
uibindings = UIBinding()

def bind_component(component):
    uibindings.add_component(component)

    if uibindings.ready_for_binding():
        modules.shared.log.debug('RI: Binding gradio components')
        uibindings.setting_remote_inference_service.change(fn=change_model_dropdowns, inputs=[uibindings.setting_remote_inference_service], outputs=[uibindings.txt2img_sampling, uibindings.img2img_sampling, uibindings.extras_upscaler_1])
        uibindings.remote_inference_balance_click.click(fn=update_balances, inputs=[], outputs=[uibindings.remote_inference_balance, uibindings.setting_remote_balance])
        uibindings.setting_remote_balance.show_progress = False
 
def change_model_dropdowns(setting_remote_inference_service_value):
    service = RemoteService[setting_remote_inference_service_value]

    if service == RemoteService.StableHorde:
        samplers = get_models(ModelType.SAMPLER, service)
        upscalers = get_models(ModelType.UPSCALER, service)

        sampler_update = gr.Dropdown.update(choices=samplers, value=samplers[0])
        upscaler_update = gr.Dropdown.update(choices=upscalers, value=upscalers[0])
        return (sampler_update, sampler_update, upscaler_update)
    
    return uibindings.back_to_default([uibindings.txt2img_sampling, uibindings.img2img_sampling, uibindings.extras_upscaler_1])

def update_balances():
    value = get_remote_balance_html()
    values = ['' if not show else value for show in [modules.shared.opts.remote_show_balance_box, modules.shared.opts.remote_show_balance_quick]]
    return tuple(gr.HTML.update(visible=False) if not value else gr.HTML.update(visible=True, value=value) for value in values)