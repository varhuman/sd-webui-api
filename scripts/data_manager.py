import os
from gradio import Blocks
import PIL.Image as Image
import scripts.utils.utils as utils
import gradio as gr
import scripts.utils.file_util as file_util
from scripts.models.sd_models import Img2ImgModel, Txt2ImgModel, CheckpointModel
from scripts.models.api_models import TemplateBaseModel, ApiType
from scripts.utils.log_util import logger
import modules.scripts as scripts
from scripts.models.sd_models import to_serializable
import json
from datetime import datetime
from scripts.utils.parse_utils import parse_string_to_txt2img_model

demo:Blocks = None
txt_img_data: Txt2ImgModel = Txt2ImgModel()
img_img_data: Img2ImgModel = Img2ImgModel()
base_data: TemplateBaseModel = TemplateBaseModel()
templates_folders = []
templates = []
choose_template = None
choose_folder = None

submited_folder = ""

pre_choose_template:TemplateBaseModel = None

checkpoints_models:list[CheckpointModel] = []

class DataManager:
    def __init__(self):
        self.basedir = scripts.basedir()
        self.default_save_dir = "outputs/apimaker"
        os.makedirs(self.default_save_dir, exist_ok=True)
        self.default_save_name = "template"
        self.refresh_checkpoints()
        

    def refresh_checkpoints(self):
        #TODO
        return checkpoints_models != []

    def get_txt2img_model(self, txt2img_prompt, txt2img_negative_prompt, steps, sampler_index, restore_faces, tiling, batch_count, batch_size, cfg_scale, seed, height, width, eta, checkpoint_model):
        return Txt2ImgModel().create(prompt=txt2img_prompt, negative_prompt=txt2img_negative_prompt, 
                            steps=steps, sampler_index=sampler_index, restore_faces=restore_faces, 
                            tiling=tiling, n_iter=batch_count, batch_size=batch_size, cfg_scale=cfg_scale, 
                            seed=seed, height=height, width=width, checkpoint_model=checkpoint_model, eta=eta)

    def get_img2img_model(self, save_path, img2img_prompt, img2img_negative_prompt, restore_faces, tiling, seed, sampler_index, steps, cfg_scale, width, height, batch_size, batch_count, eta, inpaint_full_res, inpaint_full_res_padding, checkpoint_model, img_inpaint:Image, mask_inpaint:Image, mask_blur, inpainting_fill, inpainting_mask_invert, resize_mode, denoising_strength,control_enabled,
                        control_module, control_model, control_weight, control_image, control_mask, control_invert_image, control_resize_mode, control_rgbbgr_mode, 
                        control_lowvram, control_processor_res, control_threshold_a, control_threshold_b, control_guidance_start, control_guidance_end, control_guessmode):
        if img_inpaint is not None:
            if save_path is not None:
                img_save_path = os.path.join(save_path, "init_image.png")
                img_inpaint.save(img_save_path)
            init_image = [img_save_path]
        if mask_inpaint is not None:
            if save_path is not None:
                mask_path = os.path.join(save_path, "init_mask.png")
                mask_inpaint.save(mask_path)
            init_mask = mask_path
        else:
            init_mask = ""
        if control_image is not None:
            if save_path is not None:
                control_image_path = os.path.join(save_path, "control_image.png")
                control_image.save(control_image_path)
            control_image = control_image_path
        if control_mask is not None:
            if save_path is not None:
                control_mask_path = os.path.join(save_path, "control_mask.png")
                control_mask.save(control_mask_path)
            control_mask = control_mask_path
        return Img2ImgModel().create(prompt=img2img_prompt, negative_prompt=img2img_negative_prompt, 
                            restore_faces=restore_faces, tiling=tiling, seed=seed, sampler_index=sampler_index, 
                            steps=steps, cfg_scale=cfg_scale, width=width, height=height, batch_size=batch_size, 
                            n_iter=batch_count, eta=eta, inpaint_full_res=inpaint_full_res, 
                            inpaint_full_res_padding=inpaint_full_res_padding, checkpoint_model=checkpoint_model, 
                            init_image=init_image, init_mask=init_mask, mask_blur=mask_blur, 
                            inpainting_fill=inpainting_fill, inpainting_mask_invert=inpainting_mask_invert,
                            resize_mode=resize_mode, denoising_strength=denoising_strength,
                            control_enabled=control_enabled,control_module=control_module, control_model=control_model, control_weight=control_weight, control_image=control_image, 
                            control_mask=control_mask, control_invert_image=control_invert_image, control_resize_mode=control_resize_mode, control_rgbbgr_mode=control_rgbbgr_mode,
                            control_lowvram=control_lowvram, control_processor_res=control_processor_res, control_threshold_a=control_threshold_a, control_threshold_b=control_threshold_b,
                            control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, control_guessmode=control_guessmode)
    
        
    def save_template_from_infotext(self, infotext, save_dir = "default", template_name = "default", add_time = True):
        template_model = TemplateBaseModel()
        try:
            template_model.api_model = parse_string_to_txt2img_model(infotext)
        except Exception as e:
            logger.error(f"parse_string_to_txt2img_model error: {e}")
            return "parse error, please check your generation parameters"
        template_model.template_name = self.get_new_template_name(save_dir, template_name, add_time)

        data_manager.save_template_model(save_dir, template_model)
        return f"save success: {template_model.template_name} in {save_dir}"


    def get_info_in_template_path(self, template_path, name):
        if not template_path:
            return "错误的文件夹路径"
        if not name:
            return "错误的文件名"
        global pre_choose_template, choose_folder, choose_template
        choose_folder = template_path
        temp_data = self.get_model_from_folder(choose_folder, name)
        choose_template = name
        pre_choose_template = temp_data
        res = "读取成功！\n"+ str(pre_choose_template.api_model)
        return utils.get_ellipsis_string(res, 200)

    def save_template(self, template_path, name, options, template_type_label, *args):
        global choose_folder
        if not template_path:
            template_path = self.get_new_template_folder_name()
        #name none or empty
        if not name:
            name = self.get_new_template_name(template_path, self.default_save_name)
        else:
            name = self.get_new_template_name(template_path, name)

        choose_folder = template_path
        base_data.template_name = name
        base_data.options = options
        base_data.template_type = template_type_label['label']

        if base_data.template_type == ApiType.txt2img.value:
            base_data.api_model = self.get_txt2img_model(*args)
        elif base_data.template_type == ApiType.img2img.value:
            pass
            #TODO
            # save_path = self.get_input_images_save_path(choose_folder)
            # save_path = os.path.join(save_path, name)
            # file_util.check_folder(save_path)
            # base_data.api_model = get_img2img_model(save_path, *args)
        self.save_template_model(choose_folder, base_data)
        logger.info(f"save template ({base_data.template_name}) in folder({choose_folder}) success")

    #利用file_util中的方法，获取work_dir下所有文件夹
    def get_all_templates_folders(self):
        folders = file_util.get_dirs(self.default_save_dir)
        return folders

    def get_new_template_folder_name(self):
        folders = self.get_all_templates_folders(self.default_save_dir)
        i = 1
        while True:
            folder = "template_folder" + str(i)
            if folder not in folders:
                return folder
            i += 1

    def get_new_template_name(self,dir, template_name, add_time = True):
        templates = self.get_templates_from_folder(dir)
        if add_time:
            template_name = template_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        new_template_name = template_name
        i = 1
        while True:
            if new_template_name not in templates:
                return new_template_name
            new_template_name = template_name + "_" + str(i)
            i += 1

    #利用file_util中的方法，获取某个template文件夹下所有json文件
    def get_templates_from_folder(self,dir):
        json_files = file_util.get_json_files(dir)
        return json_files

    def get_model_from_folder(self,folder, template_name):
        all_templates = self.get_templates_from_folder(folder)
        for template in all_templates:
            if template_name in template:
                return self.get_model_from_template_path(os.path.join(self.default_save_dir, folder, template_name + ".json"))
        return None

    #将json先解析成apiTypeModel，根据apiTypeModel中得type再决定解析成哪个model
    def get_model_from_template_path(self,json_file):
        content = file_util.read_json_file(json_file)
        #json to TemplateBaseModel
        data = json.loads(content)
        apiTypeModel = TemplateBaseModel(**data)
        return apiTypeModel

    def get_model_from_template(self,json):
        #json to TemplateBaseModel
        data = json.loads(json)
        apiTypeModel = TemplateBaseModel(**data)
        return apiTypeModel

    #将apiTypeModel转换成json并存储到template得指定文件夹下
    def save_template_model(self,save_dir: str, apiTypeModel:TemplateBaseModel):
        save_dir = save_dir.strip()
        os.makedirs(save_dir, exist_ok=True)
        json_file = os.path.join(save_dir, apiTypeModel.template_name + ".json")
        content = json.dumps(apiTypeModel, default= to_serializable, indent=4)
        file_util.write_json_file(json_file, content)
        
    def check_templates_folder_is_exist(self,folder):
        return folder in self.get_all_templates_folders()

    def check_templates_folder_is_exist(self,folder, template):
        return template in self.get_templates_from_folder(folder)

    #将apiTypeModel转换成json并存储到template得指定文件夹下
    def get_image_save_path(self,folder,template_name):
        time_folder = datetime.now().strftime("%Y%m%d")
        save_path = os.path.join(self.default_save_dir, f"{folder}/{template_name}", time_folder)
        return save_path

data_manager = DataManager()