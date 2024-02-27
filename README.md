# sdnext-remote
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

SD.Next extension to send compute tasks to remote inference servers.
Aimed to be universal for all providers, feel free to request other providers.

> [!NOTE]
> This project is still a Work In Progress, please report issues.

# Providers
- (WIP) [SD.Next](https://github.com/vladmandic/automatic) (someone else running SD.Next API)
- (WIP) [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (someone else running ComfyUI API)
- [StableHorde](https://stablehorde.net/) (free)
- (WIP) [OmniInfer](https://www.omniinfer.io/) (paid)
- (WIP) [ComfyICU](https://comfy.icu/) (paid)
- (WIP) Others :D

# Features
|                            | SD.Next API | ComfyUI API | StableHorde | OmniInfer | ComfyICU |
|----------------------------|-------------|-------------|-------------|-----------|----------|
| ***Model browsing***       |             |             |             |           |          |
| Checkpoints browser        | ✅          | 🆗          | ✅          | ✅        | ❌       |
| Loras browser              | ✅          | 🆗          | ⭕          | ✅        | ❌       |
| Embeddings browser         | ✅          | 🆗          | ⭕          | ✅        | ❌       |
| ***Generation***           |             |             |             |           |          |
| From Text                  | ✅          | 🆗+         | ✅          | ✅        | 🆗+      |
| From Image                 | ✅          | 🆗          | ✅          | 🆗+       | 🆗       |
| Inpainting                 | 🆗+         | 🆗          | ✅          | 🆗+       | 🆗       |
| Second pass (hires)        | 🆗+         | 🆗          | ✅          | 🆗+       | 🆗       |
| Second pass (refiner)      | 🆗          | 🆗          | ❌          | 🆗        | 🆗       |
| Loras and TIs              | 🆗          | 🆗          | ✅          | ✅        | 🆗       |
| ControlNet                 | 🆗          | 🆗          | ✅          | 🆗        | 🆗       |
| Upscale & postprocess      | 🆗          | 🆗          | ✅          | 🆗        | 🆗       |
| ***User***                 |             |             |             |           |          |
| Balance (credits/kudos)    | ⭕          | ⭕          | ✅          | ✅        | ❌       |
| Generation cost estimation | ⭕          | ⭕          | 🆗          | 🆗        | ❌       |

- ✅ functional
- 🆗+ work in progress
- 🆗 planned
- ⭕ not needed
- ❌ not supported

## Additional features
- Stable Horde worker settings
- Dynamic samplers/upscalers lists
- API calls caching
- Hide NSFW networks option

## Why yet another extension ?
There already are plenty of integrations of AI Horde. The point of this extension is to bring all remote providers into the same familiar UI instead of relying on other websites.
Eventually I'd also like to add support for other SD.Next extensions like dynamic prompts, deforum, tiled diffusion, adetailer and regional prompter (UI extensions like aspect ratio, image browser, canvas zoom or openpose editor should already be supported).


# Installation & usage
1. Launch SD.Next with backend set to `original`
2. Installation
    1. Go to extensions > manual install > paste `https://github.com/BinaryQuantumSoul/sdnext-remote` > install
    2. Go to extensions > manage extensions > apply changes & restart server
    3. Go to system > settings > remote inference > set right api endpoints & keys
3. Usage
    1. Select desired remote inference service in dropdown, **refresh model list** and **select model**
    2. Set generations parameters as usual and click generate
    
# License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
