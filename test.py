import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import argparse
import torch
from diffusers import AutoPipelineForInpainting, AutoencoderKL, AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_model_type", "-m",
        type=str,
        choices=["sd21", "sdxl"],  
        help="Specify the model type to load. Available options: sd21, sdxl"
    )
    parser.add_argument("--strength", "-s", type=float, default=1.0, help="Strength of the diffusion model")
    args = parser.parse_args()
    desired_size = (1024, 768)
    args.load_model_type = "sdxl"
    if args.load_model_type == "sd21":
        pipe_id = "stabilityai/stable-diffusion-2-1"
        cached_folder = "./lora_v2_new"
    elif args.load_model_type == "sdxl":
        pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
        cached_folder = "./weights"
    else:
        raise ValueError("Invalid model type specified. Please choose from the available options.")
    
    pipeline = AutoPipelineForInpainting.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")

    # change vae
    # vae_path="madebyollin/sdxl-vae-fp16-fix"
    # pipeline.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16).to("cuda")

    pipeline.load_lora_weights(cached_folder)
    generator = torch.Generator("cuda").manual_seed(0)
    prompt = "a healthy pig liver"
    lora_scale = 0.9

    # we randomly get 4 masks from the folder and used for inference
    mask_path = "/data_hdd3/users/yangchen/Ruijing_medical_projects/new_data/masks"
    mask_files = os.listdir(mask_path)
    masked_images = []
    number_of_masks = 8
    for mask_file in mask_files[:number_of_masks]:
        mask_image = load_image(os.path.join(mask_path, mask_file))
        masked_images.append(mask_image)

    # for simple input image, we generate a pure white image with desired size to serve as the input image
    init_image = Image.new("RGB", desired_size, (255, 255, 255))

    grids = []
    for i, mask_image in enumerate(masked_images):
        image = pipeline(prompt=prompt, image=mask_image, mask_image=mask_image, generator=generator, 
                    num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, strength=args.strength).images[0]
        # Resize the output image to match the dimensions of init_image
        image = image.resize(mask_image.size)
        grid = make_image_grid([mask_image, image], rows=1, cols=2)
        # grid.save(f"liver_inpaint_{args.load_model_type}_{i}.png")
        grids.append(grid)

    # convert the list of grids to a single image
    final_grid = make_image_grid(grids, rows=number_of_masks//2, cols=2)
    final_grid.save(f"grid_liver_inpaint_{args.load_model_type}_strength_{args.strength}_all_data.png")


    # # pipe_id = "stabilityai/stable-diffusion-2-1"
    # pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
    # pipeline = AutoPipelineForInpainting.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
    # pipeline.load_lora_weights("./lora_xl")

    # # load base and mask image
    # # init_image = load_image("/data_hdd3/users/yangchen/Ruijing_medical_projects/data/1014/images/0001.png")
    # # mask_image = load_image("/data_hdd3/users/yangchen/Ruijing_medical_projects/data/1014/output_masks/0001.png")

    # # Load and resize the image
    # desk_image_path = "desk.jpg"
    # desired_size = (1024, 768)
    # desk_image = Image.open(desk_image_path)
    # resized_desk_image = desk_image.resize(desired_size)

    # # Save the resized image if needed
    # resized_desk_image.save("resized_desk.jpg")

    # # Use the resized image as init_image
    # init_image = resized_desk_image
    # mask_image = load_image("/data_hdd3/users/yangchen/Ruijing_medical_projects/data/1014/output_masks/0001.png")

    # generator = torch.Generator("cuda").manual_seed(0)
    # prompt = "a healthy human liver"
    # lora_scale = 0.9
    # image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}).images[0]
    # # Resize the output image to match the dimensions of init_image
    # image = image.resize(init_image.size)
    # grid = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
    # grid.save("liver_inpaint_xl.png")