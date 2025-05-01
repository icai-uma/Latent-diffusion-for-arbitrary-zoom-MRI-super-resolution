from setuptools import setup, find_packages
import os
import shutil
import nibabel as nib
from PIL import Image
import PIL
import SimpleITK as sitk
import gzip
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
import argparse, os, sys, glob,shutil
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange, repeat
import cv2
from edsr import edsr 


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    
    parser.add_argument(
        "--input_size",
        type=int,
        default=256,
        help="Size to which each MRI slice will be resized (square shape, e.g., 256x256)",
    )
    parser.add_argument(
        "-n",
        "--n_slides_generated",
        type=int,
        default=2,
        help="Number of slices generated between two consecutive slices (temporal resolution)",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="Eta parameter for DDIM sampling",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="Number of steps for DDIM sampling",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config of the LDM model",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--input_mri_path",
        type=str,
        required=True,
        help="Path to the input MRI file in .nii.gz format",
    )
    parser.add_argument(
        "--edsr_path",
        type=str,
        required=True,
        help="Path to the wights of the EDSR Net",
    )

    return parser


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def get_number(element):
        return int(element.split('_')[1])

if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()


    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(opt.input_size // 4),
        ])

    t_enc = int(opt.ddim_steps)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = OmegaConf.load(f"{opt.config_path}")



################### LOADING LDM

    model = load_model_from_config(config, f"{opt.ckpt_path}")
    sampler = DDIMSampler(model)



################### LOADING EDSR


    edsr_model= edsr.edsr(scale=1, num_res_blocks=16)

    edsr_model.load_weights(opt.edsr_path)


################### Preparing the execution

    file_path = opt.input_mri_path

    input_file_name = file_path.split("/")[-1].split(".")[0]

    outpath="execution"

    try:
      shutil.rmtree(os.path.join(outpath, "inputs_z"))
    except:
      pass
      
    try:
      shutil.rmtree(os.path.join(outpath, "superresolution_z"))
    except:
      pass
      
    inputs_z = os.path.join(outpath, "inputs_z")
    os.makedirs(inputs_z, exist_ok=True)

    output_directory = os.path.join(outpath, "superresolution_z")
    os.makedirs(output_directory, exist_ok=True)

    nii_img = nib.load(file_path)

    image_data = nii_img.get_fdata()

    num_images_z = image_data.shape[-1]

    target_height = opt.input_size
    target_width = opt.input_size

#################### Generating slices from the volume

    for i in range(num_images_z):
    
        slice_data = image_data[:,:,i]

        slice_data = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype('uint8')
        slice_img = Image.fromarray(slice_data, mode='L')
        image=np.array(slice_img)
    
        if image.shape[0] != target_height or image.shape[1] != target_width:
            top_padding = (target_height - image.shape[0]) // 2
            bottom_padding = target_height - image.shape[0] - top_padding
            left_padding = (target_width - image.shape[1]) // 2
            right_padding = target_width - image.shape[1] - left_padding
            image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            image= Image.fromarray(image, mode='L')
        else:
            image= Image.fromarray(image, mode='L')
      
        output_file = os.path.join(inputs_z, f"{input_file_name}_{i+1}.png")
        image.save(output_file)

    images_z= sorted(os.listdir(inputs_z),key=get_number)
    init_image_list = []
    for item in images_z:
                        cur_image = load_img(os.path.join(inputs_z, item)).to(device)
                        cur_image = transform(cur_image)
                        init_image_list.append(cur_image)
    init_image = torch.cat(init_image_list, dim=0)



    samples, _ = sampler.sample(
    t_enc, init_image.size(0), (3, opt.input_size // 4, opt.input_size // 4),
    init_image, unconditional_guidance_scale=1.0, verbose=False, x_T=None)

#################### Interpolation in the latent space

    for j in range(len(samples)-1):
        value_0 = samples[j].to(device)
        value_1 = samples[j+1].to(device)

        interpolated_points = torch.linspace(0, 1, opt.n).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(device)
        new_values= value_0 + interpolated_points * (value_1 - value_0)

        new_tensor = new_values.reshape(opt.n, 3, 64, 64)    
    
        x_samples = model.decode_first_stage(new_tensor)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        file_name= images_z[j].split(".")[0]

        for i in range(0, x_samples.size(0)-1):
            x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
            image = (x_sample.astype(np.uint8))

            image = edsr.edsr_f(image, edsr_model)
            final_image = np.clip(image, 0, 255).astype(np.uint8)

            Image.fromarray(final_image).save(
                os.path.join(output_directory, f"{file_name}_{i}.png"))
            
#################### Upper limit, no interpolation

    j = len(samples)-1
    x_sample = model.decode_first_stage(samples[j].unsqueeze(0))
    x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    file_name = images_z[j].split(".")[0]

    x_sample= 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
    image = (x_sample.astype(np.uint8))
    final_image = np.clip(image, 0, 255).astype(np.uint8)

    Image.fromarray(final_image).save(
        os.path.join(output_directory, f"{file_name}.png"))                          



