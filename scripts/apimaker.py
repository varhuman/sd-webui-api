import os
import gradio as gr
import modules.scripts as scripts
from modules import scripts, shared, images, scripts_postprocessing
from modules.processing import StableDiffusionProcessing,StableDiffusionProcessingImg2Img, Processed
from modules.shared import cmd_opts, opts, state
from PIL import Image
import os
from modules.ui_components import ToolButton, ResizeHandleRow, FormRow, FormColumn, FormGroup, FormHTML
import modules.scripts as scripts
from scripts.models.sd_models import Txt2ImgModel, Img2ImgModel
from scripts.models.api_models import TemplateBaseModel
from scripts.data_manager import data_manager
from pydantic import BaseModel
from typing import Any
from gradio import Textbox, Label
from scripts.utils.log_util import logger

class UIData(BaseModel):
    name: str = ""
    value: Any = None
    index: int = 0

class UIDataList(BaseModel):
    data: list[UIData] = []

    def add_ui(self, name, value):
        self.data.append(UIData(name=name, value=value, index=len(self.data)))

    def value_to_list(self):
        return [data.value for data in self.data]
    
    def get_index(self, name):
        for data in self.data:
            if data.name == name:
                return data.index
        return -1
    
    def get_data(self, name):
        for data in self.data:
            if data.name == name:
                return data.value
        return None

class ApiMakerScript(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.ui_data_list = UIDataList()
        self.generation_parameters_content = ""


    def title(self):
        return f"apimaker"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        if(self.ui_data_list.data != []):
            logger.info("ui_data_list is not empty, clear")
            self.ui_data_list.data.clear()
        with gr.Accordion(f"API Maker", open=False):
            with gr.Column():
                enable = gr.Checkbox(False, placeholder="enable", label="Enable")
                self.ui_data_list.add_ui("enable", enable)
            with gr.Row():
                auto_generate = gr.Checkbox(False, placeholder="auto_generate", label="Auto Generate")
                generate_all_or_first = gr.Checkbox(False, placeholder="generate_all_or_first", label="Generate All", visible=False)

                def on_auto_generate_change(new_value):
                    return gr.update(**{"visible": new_value})
                auto_generate.change(
                    fn=on_auto_generate_change,
                    inputs=[auto_generate],
                    outputs=[generate_all_or_first]
                )
                self.ui_data_list.add_ui("auto_generate", auto_generate)
                self.ui_data_list.add_ui("generate_all_or_first", generate_all_or_first)

            generation_parameters = gr.Textbox(
                show_label=True,
                placeholder="parameters from generation image",
                value=self.generation_parameters_content,
                label="generation parameters",
                lines=2)
            self.ui_data_list.add_ui("generation_parameters", generation_parameters)
            with gr.Column():
                save_dir = gr.Textbox(label="Save Directory", elem_id="save_directory", value=data_manager.default_save_dir)
                with gr.Row():
                    #保存名字是否添加时间后缀
                    is_add_time_suffix = gr.Checkbox(False, placeholder="is_add_time_suffix", label="Add Time Suffix")
                    save_file_name = gr.Textbox(label="Save Filename", elem_id="save_file_name", value=data_manager.default_save_name)
                    self.ui_data_list.add_ui("is_add_time_suffix", is_add_time_suffix)
                    self.ui_data_list.add_ui("save_file_name", save_file_name)

                
                self.ui_data_list.add_ui("save_dir", save_dir)

            mention_text = gr.Label(
                show_label=True,
                label="log",
                value="",
                lines=1)
            generate_btn = gr.Button("Generate")
            self.ui_data_list.add_ui("mention_text", mention_text)
            self.ui_data_list.add_ui("generate_btn", generate_btn)
            generate_btn.click(
                fn=data_manager.save_template_from_infotext,
                inputs=[
                    generation_parameters,
                    save_dir,
                    save_file_name,
                    is_add_time_suffix,
                ],
                outputs=[
                    mention_text
                ]
            )

        ui_list = self.ui_data_list.value_to_list()
        return ui_list

    def postprocess(self, p:StableDiffusionProcessing, processed: Processed, *args):
        infotexts = processed.infotexts
        enable = args[self.ui_data_list.get_index("enable")]
        if not enable:
            return
        generate_all = args[self.ui_data_list.get_index("generate_all_or_first")]
        auto_generate = args[self.ui_data_list.get_index("auto_generate")]

        # Determine the range of infotexts to process based on the checkbox states
        if auto_generate:
            if generate_all:
                range_to_process = range(len(infotexts))
            else:
                range_to_process = range(1)
                
            save_dir = args[self.ui_data_list.get_index("save_dir")]
            save_file_name = args[self.ui_data_list.get_index("save_file_name")]
            is_add_time_suffix = args[self.ui_data_list.get_index("is_add_time_suffix")]
            for i in range_to_process:
                infotext = infotexts[i]
                msg = data_manager.save_template_from_infotext(infotext, save_dir, save_file_name,is_add_time_suffix)
                logger.info(msg)
        else:
            range_to_process = 0
            generation_parameters: Textbox= self.ui_data_list.get_data("generation_parameters")
            self.generation_parameters_content = infotexts[0]
            generation_parameters.value = infotexts[0]

       
