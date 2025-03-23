# FYP
Extracting Scene Intrinsics with LoRA Experimentation on Stable Diffusion Models

Data preprocessing notebook contains script for generating the diode_meta.json file for parsing data in training phase. 
In our case we have the ground truth with accompanying depth and normal maps from Diode-dataset.

train_ files are finetuned for specific tasks and Stable diffusion models, like-wise for validate_ files
There are very miniscule changes between the files, providing us with a reproducible framework for other intrinsics such as albedo and shading

Training Outputs at each epoch are saved in outputs folder. The training loss can be seen to drop each epoch/timestep from the respective .png files 
