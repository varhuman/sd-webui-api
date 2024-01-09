from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Union, Any
from scripts.models.sd_models import Txt2ImgModel, Img2ImgModel
#give a enum for txt and img
class ApiType(Enum):
    txt2img = "txt2img"
    img2img = "img2img"
    
class TemplateBaseModel(BaseModel):
    template_name: str = ""
    api_model: Txt2ImgModel = None # txt2img or img2img
    options: str = "default"
    template_type: str = ApiType.txt2img.value
    def __init__(self, **data: Any):
        super().__init__(**data)
        # if data == {} return
        if data == {}:
            return
        if self.template_type == ApiType.txt2img.value:
            self.api_model = Txt2ImgModel(**data["api_model"])
        elif self.template_type == ApiType.img2img.value:
            self.api_model = Img2ImgModel(**data["api_model"])

    def __str__(self) -> str:
        api_model_str = str(self.api_model) if self.api_model else "None"
        return f"template_name: {self.template_name}, api_model: {api_model_str}, options: {self.options}, template_type: {self.template_type}"