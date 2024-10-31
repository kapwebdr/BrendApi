from PIL import Image
from diffusers import AutoPipelineForText2Image,AutoPipelineForImage2Image #StableDiffusionUpscalePipeline
from transformers import CLIPProcessor, CLIPModel
import torch
from os import path
from slugify import slugify
import pytesseract
from PIL import Image
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
import json
import os
import io
import base64

repo_id             = "stabilityai/stable-diffusion-xl-base-1.0"
repo_id_refiner     = "stabilityai/stable-diffusion-xl-refiner-1.0"
repo_id_turbo       = "stabilityai/sdxl-turbo"
repo_from_image     = "runwayml/stable-diffusion-v1-5"

current_file = path.realpath(__file__)

def SDXL(prompt,turbo=True,num_images_per_prompt=1,output_dir="Outputs",num_inference_steps=25):
    repo = repo_id
    if turbo == True:
        repo = repo_id_turbo
    #, requires_safety_checker=False
    #, use_safetensors=True
    pipe = AutoPipelineForText2Image.from_pretrained(repo, torch_dtype=torch.float16, variant="fp16", cache_dir=path.join(path.dirname(current_file),"..", "Cache"))
    pipe.to("mps") #,strength=0.5
    images = pipe(prompt=prompt,num_images_per_prompt=1,num_inference_steps=1, guidance_scale=0.0).images
    savedimages = SaveImage(prompt,images,output_dir)
    return savedimages

def SDXLImage2Image(prompt,image_base,turbo=True,num_images_per_prompt=1,num_inference_steps=25):
    repo = repo_id
    if turbo == True:
        repo = repo_id_turbo
    pipe = AutoPipelineForImage2Image.from_pretrained(repo, requires_safety_checker=False, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir=path.join(path.dirname(current_file),"..", "Cache"))
    pipe.to("mps")
    #init_image = load_image(path.join(path.dirname(current_file),"..",image_base)).resize((512, 512))
    init_image = Image.open(path.join(path.dirname(current_file),"..",image_base)).resize((512, 512))
    images = pipe(prompt=prompt,image=init_image,num_images_per_prompt=num_images_per_prompt,num_inference_steps=num_inference_steps, strength=0.5, guidance_scale=0.0).images
    savedimages = SaveImage(prompt,images)
    return savedimages

def SD15FromIMAGE(prompt,image_base,num_images_per_prompt=1,num_inference_steps=25):
    pipe = AutoPipelineForImage2Image.from_pretrained(repo_from_image, requires_safety_checker=False , torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir=path.join(path.dirname(current_file),"..", "Cache"))       
    pipe.to("mps")
    images = pipe(prompt=prompt,image=image_base,num_images_per_prompt=num_images_per_prompt,num_inference_steps=num_inference_steps).images
    savedimages = SaveImage(prompt,images)
    return savedimages

def SaveImage(prompt,images,output_dir="Outputs"):
    savedimages=[]
    dirpath = path.join(path.dirname(current_file),"..",output_dir)
    for idx, image in enumerate(images):
        imagename = f'{slugify(prompt)}-{idx}.png'
        savedimages.append(imagename)
        image_name = imagename
        image.save(path.join(dirpath,image_name))
    return savedimages

def Ocr(image,input_dir="static"):
    file = path.join(path.dirname(current_file),"..",input_dir,image)
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

#https://github.com/mlfoundations/open_clip
#https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/discussions
#https://github.com/KerryHalupka/intro_to_clip_blog/blob/master/load_clip.py
def Labelize(image_base, labels=[],modelname="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",device="mps"):
    image = Image.open(path.join(path.dirname(current_file),"..",image_base)).resize((512, 512))
    model = CLIPModel.from_pretrained(modelname)
    processor = CLIPProcessor.from_pretrained(modelname)
    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
    return logits_per_image,probs

def load_image_models(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

image_config = {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "stream": True
}

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def loadImageModel(model):
    model_path = path.join(path.dirname(current_file), "..", "Cache", "ImageModel", 
                          model['model_name'].replace('/', path.sep), model['model_file'])
    if not path.exists(model_path):
        return None, None

    n_gpu_layers = 1
    n_batch = 4096
    
    model = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=n_batch,
        f16_kv=True,
        config=image_config,
        callback_manager=callback_manager,
        streaming=True
    )
    
    return model, model

async def generate_image(prompt, model, negative_prompt="", width=1024, height=1024):
    try:
        # Simulation de la génération progressive
        steps = 20
        for step in range(steps):
            # Ici nous simulerons les étapes de la génération
            progress = (step + 1) / steps
            yield f"data: {{'progress': {progress:.2f}}}\n\n"
            
        # Une fois l'image générée, la convertir en base64
        # Note: ceci est un placeholder, à remplacer par la vraie génération
        image_path = "generated_image.png"
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            yield f"data: {{'status': 'completed', 'image': '{encoded_string}'}}\n\n"
            
    except Exception as e:
        yield f"data: {{'error': 'Erreur lors de la génération : {str(e)}'}}\n\n"