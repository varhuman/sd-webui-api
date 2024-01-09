from scripts.models.sd_models import Img2ImgModel, Txt2ImgModel
import re
import scripts.utils.utils as utils
    
re_param_code = r'\s*([\w ]+):\s*("(?:\\"[^,]|\\"|\\|[^\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
def parse_string_to_img2img_model(s: str) -> Img2ImgModel:
    def get_int_value(pattern: str, default: int = 0):
        match = re.search(pattern, s)
        return int(match.group(1)) if match else default

    def get_float_value(pattern: str, default: float = 0.0):
        match = re.search(pattern, s)
        return float(match.group(1)) if match else default

    def get_string_value(pattern: str, default: str = ''):
        match = re.search(pattern, s)
        return match.group(1) if match else default
    *lines, lastline = s.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ''
    done_with_prompt = False
    prompt = ""
    negative_prompt = ""
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()

        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line
    steps = get_int_value(r'Steps: (\d+),')
    sampler = get_string_value(r'Sampler: ([^,]+),', '')
    cfg_scale = get_int_value(r'CFG scale: (\d+),')
    seed = get_int_value(r'Seed: (\d+),')
    restoration = get_string_value(r'Face restoration: ([^,]+),', '')
    size = re.search(r'Size: (\d+)x(\d+),', s)
    width, height = int(size.group(1)), int(size.group(2)) if size else (512, 512)
    model_hash = get_string_value(r'Model hash: ([^,]+),', '')
    denoising_strength = get_float_value(r'Denoising strength: ([^,]+),')
    mask_blur = get_int_value(r'Mask blur: (\d+),')

    checkpoint_model = utils.get_model_name_from_hash(model_hash) 

    return Img2ImgModel().create(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        sampler_index=sampler,
        restore_faces=True if restoration else False,
        checkpoint_model=checkpoint_model,
        denoising_strength=denoising_strength,
        mask_blur=mask_blur
    )
def parse_string_to_txt2img_model(s: str) -> Txt2ImgModel:
    def get_int_value(pattern: str, default: int = 0):
        match = re.search(pattern, s)
        return int(match.group(1)) if match else default

    def get_float_value(pattern: str, default: float = 0.0):
        match = re.search(pattern, s)
        return float(match.group(1)) if match else default

    def get_string_value(pattern: str, default: str = ''):
        match = re.search(pattern, s)
        return match.group(1) if match else default

    *lines, lastline = s.strip().split("\n")
    prompt = ""
    negative_prompt = ""
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ''
    done_with_prompt = False
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()

        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    steps = get_int_value(r'Steps: (\d+),')
    sampler = get_string_value(r'Sampler: ([^,]+),', '')
    cfg_scale = get_int_value(r'CFG scale: (\d+),')
    seed = get_int_value(r'Seed: (\d+),')
    restoration = get_string_value(r'Face restoration: ([^,]+),', '')
    size = re.search(r'Size: (\d+)x(\d+),', s)
    width, height = int(size.group(1)), int(size.group(2)) if size else (512, 512)
    model_hash = get_string_value(r'Model hash: ([^,]+),', '')

    if model_hash == '':
        raise ValueError('Model hash is empty')
    if cfg_scale == 0 or steps == 0:
        raise ValueError('CFG scale is 0 or steps is 0')
    if sampler == '':
        raise ValueError('Sampler is empty')

    # checkpoint_model = utils.get_model_name_from_hash(model_hash) 
    checkpoint_model = model_hash
    return Txt2ImgModel().create(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        sampler_index=sampler,
        restore_faces=True if restoration else False,
        checkpoint_model=checkpoint_model,
    )

