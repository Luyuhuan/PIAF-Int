# SKGC: Semantic Knowledge-Guided Classification for Congenital Heart Disease Diagnosis

## Overview

🚀 **Introduction:** Congenital heart disease (CHD) is a critical health concern affecting healthy development and growth. SKGC, our proposed framework, leverages deep learning to provide accurate diagnosis, overcoming limitations in existing methods.

💡 **Key Features:**
- **Semantic-Level Knowledge Extraction (SKEM) 🧠:** Extracts crucial semantic knowledge from 4CH ultrasound images.
- **Multi-Knowledge Fusion (MFM) 🔄:** Integrates knowledge from semantic and original images.
- **Classification Module (CM) 🤖:** Enables flexible classification using advanced classifiers.

🎯 **Loss Function Enhancement:** SKGC introduces a novel loss function, strengthening the constraint between foreground and background predictions for improved semantic-level knowledge.

## Experimental Environment

🔧 **Platform:** Arch Linux  
💻 **Hardware:** Intel Xeon Gold 5320 CPU, NVIDIA GeForce RTX 3090 GPU  
🛠️ **Implementation:** PyTorch with MMPreTrain, MMSegmentation

## Results

📊 **Performance:** SKGC demonstrates remarkable accuracy, achieving peak accuracies of 99.1% and 96.5% on real-world NA-4CH and CAMUS datasets, respectively.

🌟 **Noteworthy Improvement:** Even with only 10 labeled masks, SKGC enhances accuracy from 75.9% to 88.5%.

## Getting Started

### 1. Environment Installation

Make sure to install the necessary packages in your environment:
- PyTorch
- mmcv

```bash
pip install torch
pip install mmcv
###  2. Code Compilation

Compile the `mmseg` and `mmpretrain` code provided in this repository. Note that these codes are modified from the official ones. Use the provided compilation script or follow the instructions in the repository.

```bash
# Example compilation command
./compile_script.sh

### 3. Data Preparation for Training

Organize training data in COCO format. Ensure that images are stored in different folders based on their categories.

### 4. Commands (Using XCiT as an Example)

#### Training

Use the following command for training:

```bash
nohup ./dist_train.sh mmpretrain/configs/xcit/xcit-nano-12-p16_8xb128_in1k.py gpunum --work-dir yourworkpath >yourlogpath  2>&1 &

Replace mmpretrain/configs/xcit/xcit-nano-12-p16_8xb128_in1k.py with your chosen configuration file, gpunum with the number of GPUs you want to use, yourworkpath with the path where you want to save the training results, and yourlogpath with the path to save the training logs.

Please note: If you need to modify parameters, you can do so in the configuration files.
#### Testing

Run the following command for testing:

```bash
python run_dist_test.py

The test results will be automatically saved in log and pkl files.

Please note: If you need to modify parameters, you can do so in the configuration files.

### 5. Tool Usage
The relevant statistical tool code is located in the tool path. Use the tools as needed.

## Feedback

Feel free to contribute, report issues, or give feedback. Together, let's advance CHD diagnosis using state-of-the-art technology!

