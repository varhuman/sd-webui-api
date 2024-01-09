from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Union, Any
import re
import scripts.utils.utils as utils
max_string_length = 2000


def to_serializable(obj: Any):
    if isinstance(obj, BaseModel):
        return obj.dict()
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj.__dict__

class resize_mode(Enum):
    img2img = 1
    inpaint = 2
    inpaint_sketch = 3
    inpaint_upload_mask = 4

class CheckpointModel(BaseModel):
    title: str
    model_name: str

class Txt2ImgModel(BaseModel):
    prompt: str = "1girl"
    negative_prompt: str = "nsfw"
    override_settings: Dict[str, Union[str, int]] = {}
    seed: int = -1
    batch_size: int = 1 #每次张数
    n_iter: int = 1 # 生成批次
    steps: int = 20
    cfg_scale: float = 7
    width: int = 512
    height: int = 512
    restore_faces: bool = False
    clip_skip: int = 2
    enable_hr: bool = False
    hr_scale: float = 2
    hr_upscaler: str = "R-ESRGAN 4x+"
    denoising_strength: float = 0.3
    tiling: bool = False
    eta: int = 31337
    script_args: List[str] = []
    sampler_index: str = "Euler a"
    alwayson_scripts: Dict[str, Dict[str, Any]] = {}

    def shorten_strings(self,data):
        for key, value in data.items():
            if isinstance(value, dict):
                self.shorten_strings(value)
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str) and len(item) > max_string_length:
                        data[key][i] = f"...{len(item)}..."
                    elif isinstance(item, dict):
                        self.shorten_strings(item)
            elif isinstance(value, str) and len(value) > max_string_length:
                data[key] = f"...too long({len(value)})..."
        return data

    def __str__(self):
        data = self.dict()
        data = self.shorten_strings(data)
        return str(data)
    
    def create(self, prompt="", negative_prompt="", checkpoint_model="", seed=-1, batch_size=1, n_iter=1, steps=20, cfg_scale=7, width=512, height=512, restore_faces=False, tiling=False, eta=31337, sampler_index="Euler a"):
        super().__init__(prompt=prompt, negative_prompt=negative_prompt, seed=seed, batch_size=batch_size, n_iter=n_iter, steps=steps, cfg_scale=cfg_scale, width=width, height=height, restore_faces=restore_faces, tiling=tiling, eta=eta, sampler_index=sampler_index)
        self.set_override_settings(checkpoint_model)
        return self

    def set_override_settings(self, model):
        self.override_settings={}
        # title = utils.get_title_from_model_name(model)
        title = model
        self.override_settings['sd_model_checkpoint'] = title if title else None

    def get_checkpoint_model(self):
        return self.override_settings['sd_model_checkpoint'] if 'sd_model_checkpoint' in self.override_settings else None


inpainting_fill_choices = ['fill', 'original', 'latent noise', 'latent nothing']
inpainting_mask_invert_choices = ['Inpaint masked', 'Inpaint not masked']
resize_mode_choices = ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]
inpaint_full_res_choices = ['Whole picture', 'Only masked']
controlnet_resize_mode = ["Just Resize", "Inner Fit (Scale to Fit)", "Outer Fit (Shrink to Fit)"]
 # 0 img2img 1 img2img 2 inpaint 3 inpaint sketch 4 inpaint upload mask 暂时我们只需要4和1
class Img2ImgModel(Txt2ImgModel):
    init_images: List[str] = None #img2img 基础的图都在里面： 文件地址
    mask:str = None # 文件地址
    resize_mode: int = 1#["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]
    denoising_strength: float = 0.72
    mask_blur:int = 0 #蒙版模糊 4
    inpainting_fill:int = 0# 蒙版遮住的内容， 0填充， 1原图 2潜空间噪声 3潜空间数值零
    inpaint_full_res:int = 0 # inpaint area 0 whole picture 1：only masked
    inpaint_full_res_padding:int = 32 # Only masked padding, pixels 32
    inpainting_mask_invert:int = 0 # 蒙版模式 0重绘蒙版内容 1 重绘非蒙版内容

    def __str__(self):
        data = self.dict()
        data = self.shorten_strings(data)
        return str(data)

    def get_attribute_value(self, attribute_name):
        if attribute_name == "init_images":
            return self.init_images[0] if self.init_images else None
        elif attribute_name == "inpainting_fill":
            return inpainting_fill_choices[self.inpainting_fill]
        elif attribute_name == "inpainting_mask_invert":
            return inpainting_mask_invert_choices[self.inpainting_mask_invert]
        elif attribute_name == "inpaint_full_res":
            return inpaint_full_res_choices[self.inpaint_full_res]
        elif attribute_name == "resize_mode":
            return resize_mode_choices[self.resize_mode]
        #attribute_name包含control_的话，就是控制模块的参数
        elif attribute_name.startswith("control_"):
            key = attribute_name.replace("control_", "")
            args = self.get_controlnet_params()

            if key == "resize_mode":
                return controlnet_resize_mode[args[key]]
            return args[key] if key in args else None
        return super().get_attribute_value(attribute_name)
    
    def get_init_image(self):
        return self.init_images[0] if self.init_images else None
    
    def create(self, prompt="", negative_prompt="", checkpoint_model="", seed=0, batch_size=1, n_iter=1, steps=20, cfg_scale=0.72, width=512, height=512, restore_faces=False, tiling=False, eta=31337, sampler_index=0, inpaint_full_res=0, inpaint_full_res_padding=32, init_image="", init_mask="", mask_blur=4, inpainting_fill=0, inpainting_mask_invert=0,resize_mode=1, denoising_strength=0.72,
                control_enabled=False, control_module="", control_model="", control_weight=0.0, control_image="", control_mask="", control_invert_image=False, control_resize_mode=0, control_rgbbgr_mode=0, control_lowvram=False, control_processor_res=0, control_threshold_a=0.0, control_threshold_b=0.0, control_guidance_start=0, control_guidance_end=0, control_guessmode=0):
        super().__init__(prompt=prompt, negative_prompt=negative_prompt, seed=seed, batch_size=batch_size, n_iter=n_iter, steps=steps, cfg_scale=cfg_scale, width=width, height=height, restore_faces=restore_faces, tiling=tiling, eta=eta, sampler_index=sampler_index,resize_mode=resize_mode, denoising_strength=denoising_strength)
        self.set_override_settings(checkpoint_model)
        self.setup_img2img_params(init_image, init_mask, mask_blur, inpainting_fill, inpainting_mask_invert, inpaint_full_res, inpaint_full_res_padding)
        self.setup_controlnet_params(control_enabled, control_module, control_model, control_weight, control_image, control_mask, control_invert_image, control_resize_mode, control_rgbbgr_mode, control_lowvram, control_processor_res, control_threshold_a, control_threshold_b, control_guidance_start, control_guidance_end, control_guessmode)
        return self
    
    def setup_img2img_params(self, init_image, init_mask, mask_blur, inpainting_fill, inpainting_mask_invert, inpaint_full_res, inpaint_full_res_padding):
        self.init_images = init_image
        self.mask = init_mask
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert

    def set_override_settings(self, model):
        self.override_settings={}
        self.override_settings['sd_model_checkpoint'] = model

    def get_checkpoint_model(self):
        return self.override_settings['sd_model_checkpoint'] if 'sd_model_checkpoint' in self.override_settings else None

    def setup_controlnet_params(self, enabled, module, model, weight, image, mask, invert_image, resize_mode, rgbbgr_mode, lowvram, processor_res, threshold_a, threshold_b, guidance_start, guidance_end, guessmode):
        controlnet_args = {
            "enabled": enabled,
            "module": module,
            "model": model,
            "weight": weight,
            "image": image,
            "mask": mask,
            "invert_image": invert_image,
            "resize_mode": resize_mode,
            "rgbbgr_mode": rgbbgr_mode,
            "lowvram": lowvram,
            "processor_res": processor_res,
            "threshold_a": threshold_a,
            "threshold_b": threshold_b,
            "guidance_start": guidance_start,
            "guidance_end": guidance_end,
            "guessmode": guessmode
        }
        self.alwayson_scripts["ControlNet"] = {"args": controlnet_args}

    def get_controlnet_params(self):
        return self.alwayson_scripts["ControlNet"]["args"] if "ControlNet" in self.alwayson_scripts else ControlNetModel()

    def custom_to_dict(self):
        res_dict = self.dict()
        init_image = utils.image_path_to_base64(self.init_images[0])
        res_dict["init_images"][0] = init_image
        res_dict["mask"] = utils.image_path_to_base64(self.mask)
        if self.alwayson_scripts.get("ControlNet"):
            controlnet_args:ControlNetModel = self.get_controlnet_params()
            if controlnet_args:
                res_dict["alwayson_scripts"]["ControlNet"]["args"]["image"] = utils.image_path_to_base64(controlnet_args["image"])
                res_dict["alwayson_scripts"]["ControlNet"]["args"]["mask"] = utils.image_path_to_base64(controlnet_args["mask"])
        return res_dict

class ControlNetModel(BaseModel):
    enabled: bool = True #启用
    module: str = str #模式 openpose、canny等
    model: str = "" # 模型 control_openpose-fp16 [9ca67cc5]
    weight: float = 1.0 #权重
    image: str = None #图片
    mask: str = None #图片遮罩，一般不用
    invert_image: bool = False #反转图片
    resize_mode: int = 1 #0:Just Resize 1: Inner Fit 2: Outer Fit
    rgbbgr_mode: bool = False
    lowvram: bool = False #低显存需要开启
    processor_res: int = 512
    threshold_a: int = 64
    threshold_b: int = 64
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    guessmode: bool = False

class FaceEditorModel(BaseModel):
    enabled: bool = True
    affected_areas: List[str] = ["Face"] #Face,Neck,Hat,Hair
    mask_size: int = 0
    mask_blur: int = 12
