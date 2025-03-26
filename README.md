# Liver Texture Generation Project

This project focuses on generating liver textures using advanced diffusion models. Below is a detailed explanation of the environment setup, data preprocessing, training, testing, and inference processes.

---

## 0. Environment Setup

Before starting, set up the required environment using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```


## 1. Data Preprocessing

The preprocessing step involves preparing the dataset for training and inference. This includes generating masked images and inverting masks.

### Key Script:
- **`preprocess_images.py`**


## 2. Training

The training process fine-tunes a pre-trained diffusion model on the prepared dataset.

### Key Script:
- **`train.sh`**

### Steps:
1. **Model Setup**:  
    Load the base model (`stabilityai/stable-diffusion-xl-base-1.0`) and VAE model.  

2. **Training Configuration**:  
    - Training data: Masked images.  
    - Batch size: 1 (with gradient accumulation).  
    - Learning rate: `1e-04`.  
    - Training steps: 5000.  

3. **Validation**:  
    Validate the model periodically using a prompt like "a healthy pig liver".  

4. **Output**:  
    The fine-tuned model is saved to the specified output directory.


## 3. Testing

The testing process evaluates the fine-tuned model using specific prompts and masks.

### Key Script:
- **`test.py`**

### Steps:
1. **Load Model**:  
    Load the fine-tuned model and its LoRA weights.  

2. **Generate Outputs**:  
    Use random masks and a prompt (e.g., "a healthy pig liver") to generate liver textures.  

3. **Save Results**:  
    Save the generated images and grids for evaluation.


## 4. Inference

The inference process uses advanced techniques like ControlNet to generate liver textures with additional control signals.

### Key Notebook:
- **`infer_controlnet.ipynb`**

### Steps:
1. **Edge Detection (Canny)**:  
    Generate edge maps from masked images using the Canny algorithm.  

2. **Depth Estimation**:  
    Generate depth maps using a pre-trained depth estimation model (`Intel/dpt-hybrid-midas`).  

3. **ControlNet Integration**:  
    Use ControlNet models (`controlnet-depth-sdxl-1.0` and `controlnet-canny-sdxl-1.0`) to condition the generation process.  

4. **Generate Images**:  
    Generate liver textures using prompts and control signals (e.g., depth or edge maps).  

