import json
import html
import requests

import modules.shared
import modules.sd_models
import modules.textual_inversion
import modules.sd_hijack
import modules.ui_extra_networks
from modules.shared import log
import network
import networks

from extension.utils_remote import ModelType, RemoteService, get_current_api_service, get_remote_endpoint, safeget, get_or_error_with_cache, stable_horde_samplers

def log_debug_model_list(model_type, api_service):
    log.debug(f'RI: Listing {model_type.name.lower()}s from {api_service}')

def log_info_model_count(model_type, api_service, count):
    log.info(f'Available {model_type.name.lower()}s: {api_service} items={count}')

def sdnext_preview_url(url):
    return get_remote_endpoint(RemoteService.SDNext) + url[1:]

def get_models(model_type: ModelType, service: RemoteService):
    #================================== SD.Next ==================================
    if service == RemoteService.SDNext:
        model_list = get_or_error_with_cache(service, "/sdapi/v1/extra-networks")
        for model in model_list:
            model.update({'name': model['name'].split('\\')[-1]})
        model_list = sorted(model_list, key=lambda model: str.lower(model['name']))

        if model_type == ModelType.CHECKPOINT:
            for model in filter(lambda model: model['type'] == 'model', model_list):
                RemoteCheckpointInfo(model['name'], sdnext_preview_url(model['preview']), filename=model['filename'])

        elif model_type == ModelType.LORA:
            for model in filter(lambda model: model['type'] == 'lora', model_list):
                RemoteLora(model['name'], sdnext_preview_url(model['preview']), filename=model['filename'])

        elif model_type == ModelType.EMBEDDING:
            for model in filter(lambda model: model['type'] == 'embedding', model_list):
                RemoteEmbedding(model['name'], sdnext_preview_url(model['preview']), filename=model['filename'])

    #================================== Stable Horde ==================================
    elif service == RemoteService.StableHorde:
        if model_type == ModelType.CHECKPOINT:          
            model_list = get_or_error_with_cache(service, "/v2/status/models")

            model_list = filter(lambda model: model['type'] == 'image', model_list)
            
            data = json.loads(requests.get('https://github.com/Haidra-Org/AI-Horde-image-model-reference/blob/main/stable_diffusion.json').content)
            data_models = json.loads(''.join(data['payload']['blob']['rawLines']))
            
            for model in sorted(model_list, key=lambda model: (-model['count'], model['name'])):
                model_data = safeget(data_models, model['name'])
                if not safeget(model_data, 'nsfw') or modules.shared.opts.remote_show_nsfw_models: 
                    RemoteCheckpointInfo(f"{model['name']} ({model['count']})", safeget(model_data,'showcases',0), safeget(model_data,'description'), filename=model['name'])
    
        elif model_type == ModelType.LORA:
            pass
        elif model_type == ModelType.EMBEDDING:
            pass
        elif model_type == ModelType.SAMPLER:
            return list(stable_horde_samplers.keys())
        elif model_type == ModelType.UPSCALER:
            return ['RealESRGAN_x2plus', 'RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B', 'NMKD_Siax', '4x_AnimeSharp']
        elif model_type == ModelType.CONTROLNET:
            return ["canny", "hed", "depth", "normal", "openpose", "seg", "scribble", "fakescribbles", "hough"]

    #================================== OmniInfer ==================================
    elif service == RemoteService.OmniInfer:
        if not model_type in [ModelType.CHECKPOINT, ModelType.LORA, ModelType.EMBEDDING]:
            return

        model_list = get_or_error_with_cache(service, "/v2/models")
        model_list = model_list['data']['models']
        for model in model_list:
            model.update({'name': model['name'].lstrip()})
        model_list = sorted(model_list, key=lambda model: str.lower(model['name']))
        if not modules.shared.opts.remote_show_nsfw_models:
            model_list = list(filter(lambda x: not x['civitai_nsfw'], model_list))
            for model in model_list:
                if 'civitai_images' in model:
                    model['civitai_images'] = list(filter(lambda img: img['nsfw'] == 'None', model['civitai_images']))

        if model_type == ModelType.CHECKPOINT:
            for model in filter(lambda model: model['type'] == 'checkpoint', model_list):
                tags = {tag:0 for tag in model['civitai_tags'].split(',')} if 'civitai_tags' in model else {}
                RemoteCheckpointInfo(model['name'], safeget(model, 'civitai_images', 0, 'url'), filename=model['sd_name'], tags=tags)

        elif model_type == ModelType.LORA:          
            for model in filter(lambda model: model['type'] == 'lora', model_list):
                tags = {tag:0 for tag in model['civitai_tags'].split(',')} if 'civitai_tags' in model else {}
                RemoteLora(model['name'], safeget(model, 'civitai_images', 0, 'url'), tags=tags, filename=model['sd_name'])

        elif model_type == ModelType.EMBEDDING:
            for model in filter(lambda model: model['type'] == 'textualinversion', model_list):
                tags = {tag:0 for tag in model['civitai_tags'].split(',')} if 'civitai_tags' in model else {}
                RemoteEmbedding(model['name'], safeget(model, 'civitai_images', 0, 'url'), tags=tags, filename=model['sd_name'])
        
class PreviewDescriptionInfo():
    no_preview = modules.ui_extra_networks.ExtraNetworksPage.link_preview(None, 'html/card-no-preview.png')

    def __init__(self, preview_url=None, description=None, info=None):
        self.preview = preview_url or PreviewDescriptionInfo.no_preview
        self.description = description
        self.info = info

#============================================= CHECKPOINTS =============================================
class RemoteCheckpointInfo(modules.sd_models.CheckpointInfo, PreviewDescriptionInfo):
    def __init__(self, name, preview_url=None, description=None, info=None, filename='', tags={}):
        PreviewDescriptionInfo.__init__(self, preview_url, description, info)

        self.name = self.name_for_extra = self.model_name = self.title = name
        self.type = f"remote"
        self.ids = [self.name]

        self.model_info = None
        self.metadata = {}
        self.sha256 = self.hash = self.shorthash = None
        self.filename = self.path = filename

        self.tags = tags

        self.register()

def list_remote_models():
    api_service = get_current_api_service()

    modules.sd_models.checkpoints_list.clear()
    modules.sd_models.checkpoint_aliases.clear()

    log_debug_model_list(ModelType.CHECKPOINT, api_service)
    get_models(ModelType.CHECKPOINT, api_service)
    log_info_model_count(ModelType.CHECKPOINT, api_service, len(modules.sd_models.checkpoints_list))

def extra_networks_checkpoints_list_items(self):
    for name, checkpoint in modules.sd_models.checkpoints_list.items():
        yield {
            "type": 'Model',
            "name": name,
            "title": checkpoint.title,
            "filename": checkpoint.filename,
            "hash": None,
            "search_term": name + ' '.join(checkpoint.tags),
            "preview": checkpoint.preview,
            "local_preview": None,
            "description": checkpoint.description,
            "info": checkpoint.info,
            "metadata": checkpoint.metadata,
            "onclick": '"' + html.escape(f"""return selectCheckpoint({json.dumps(name)})""") + '"',
        }

#============================================= LORAS =============================================       
class RemoteLora(network.NetworkOnDisk, PreviewDescriptionInfo):
    def __init__(self, name, preview_url=None, description=None, info=None, tags={}, filename=''):
        PreviewDescriptionInfo.__init__(self, preview_url, description, info)

        self.name = name
        self.filename = self.alias = filename

        self.tags = tags

        self.ssmd_cover_images = None
        self.metadata = {}
        self.hash = self.shorthash = None

        self.register()

    def register(self):
        networks.available_networks[self.name] = self
        if self.alias in networks.available_network_aliases:
            networks.forbidden_network_aliases[self.alias.lower()] = 1
        networks.available_network_aliases[self.name] = self
        networks.available_network_aliases[self.alias] = self

def list_remote_loras():
    api_service = get_current_api_service()

    networks.available_networks.clear()
    networks.available_network_aliases.clear()
    networks.forbidden_network_aliases.clear()
    networks.available_network_hash_lookup.clear()
    networks.forbidden_network_aliases.update({"none": 1, "Addams": 1})

    log_debug_model_list(ModelType.LORA, api_service)
    get_models(ModelType.LORA, api_service)
    log_info_model_count(ModelType.LORA, api_service, len(networks.available_networks))

def extra_networks_loras_list_items(self):
    for name, remote_lora in networks.available_networks.items():
        prompt = f" <lora:{remote_lora.filename}:{modules.shared.opts.extra_networks_default_multiplier}>"
        prompt = json.dumps(prompt)

        yield {
            "type": 'Lora',
            "name": name,
            "filename": remote_lora.filename,
            "hash": None,
            "search_term": name + ' '.join(remote_lora.tags),
            "preview": remote_lora.preview,
            "description": remote_lora.description,
            "info": remote_lora.info,
            "prompt": prompt,
            "local_preview": None,
            "metadata": remote_lora.metadata,
            "tags": remote_lora.tags,
        }

#============================================= EMBEDDINGS =============================================
class RemoteEmbedding(modules.textual_inversion.textual_inversion.Embedding, PreviewDescriptionInfo):
    def __init__(self, name, preview_url=None, description=None, info=None, filename='', tags={}):
        super().__init__(None, name)
        PreviewDescriptionInfo.__init__(self, preview_url, description, info)

        self.filename = filename
        self.tags = tags

        self.register()

    def register(self):
        modules.sd_hijack.model_hijack.embedding_db.word_embeddings[self.name] = self

def extra_networks_textual_inversions_refresh(self):
    modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

def list_remote_embeddings(self, force_reload=False):
    api_service = get_current_api_service()

    self.ids_lookup.clear()
    self.word_embeddings.clear()
    self.skipped_embeddings.clear()
    self.embeddings_used.clear()
    self.expected_shape = None
    self.embedding_dirs.clear()

    log_debug_model_list(ModelType.EMBEDDING, api_service)
    get_models(ModelType.EMBEDDING, api_service)
    log_info_model_count(ModelType.EMBEDDING, api_service, len(self.word_embeddings))

def extra_networks_textual_inversions_list_items(self):
    for name, embedding in modules.sd_hijack.model_hijack.embedding_db.word_embeddings.items():
        prompt = json.dumps(f" {embedding.filename}")

        yield {
            "type": 'Embedding',
            "name": name,
            "filename": embedding.filename,
            "preview": embedding.preview,
            "description": embedding.description,
            "info": embedding.info,
            "search_term": name + ' '.join(embedding.tags),
            "prompt": prompt,
            "local_preview": None,
            "tags": embedding.tags,
        }
