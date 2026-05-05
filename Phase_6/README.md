# 🌍 Siamese U-Net for Deforestation Change Detection

This project implements a deep learning–based change detection framework using a **Siamese U-Net architecture** to identify land-cover changes from bi-temporal satellite imagery.

The model combines:
- Siamese feature extraction (shared weights)
- U-Net encoder–decoder architecture
- BCE + Dice loss for segmentation

---

## 📁 Project Structure

PHASE_6/
│
├── data/
│   └── processed_dataset/
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── siamese_unet.py
│   ├── unet_blocks.py
│   ├── train.py
│   ├── evaluate.py
│   └── verify_phase6.py
│
├── requirements.txt
├── siamese_unet_dice.pth
├── siamese_unet.pth

---

## ⚙️ System Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- GPU (optional but recommended)

---

## 🚀 Setup Instructions (New System)

### 1. Clone the Repository
git clone <your-repo-url>  
cd PHASE_6

---

### 2. Create Virtual Environment

python -m venv venv

Activate it:

Windows:
venv\Scripts\activate

Linux / Mac:
source venv/bin/activate

---

### 3. Install Dependencies

pip install -r requirements.txt

---

### ⚠️ GPU Setup (Important)

If using GPU, install PyTorch separately:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

(Change cu118 based on your CUDA version)

---

## 📊 Dataset Structure

Make sure your dataset is organized like this:

processed_dataset/
├── train/
│   ├── t1/
│   ├── t2/
│   └── label/
│
├── val/
│   ├── t1/
│   ├── t2/
│   └── label/
│
├── test/
│   ├── t1/
│   ├── t2/
│   └── label/

---

## 🧠 Training the Model

cd src  
python train.py

✔ Shows progress bar  
✔ Displays loss and ETA  
✔ Saves model as: siamese_unet_dice.pth  

---

## 📈 Evaluate the Model

python evaluate.py

Outputs:
- Precision
- Recall
- F1-score
- IoU

---

## 🔍 Verify Predictions

python verify_phase6.py

Used for:
- Visual inspection
- Checking prediction quality

---

## ⚙️ Configuration

Edit hyperparameters in:

src/config.py

Important parameters:
- BATCH_SIZE
- LR (learning rate)
- EPOCHS
- DEVICE ("cpu" or "cuda")

---

## 🧩 Model Architecture

- Siamese encoder (shared weights)
- Feature differencing: |F1 - F2|
- U-Net decoder with skip connections
- Output: binary change mask

---

## 📉 Loss Function

Loss = BCE + Dice

This helps:
- Handle class imbalance
- Improve segmentation accuracy

---

## 📌 Key Features

✔ Robust to illumination changes  
✔ Handles spatial misalignment  
✔ High precision change detection  
✔ Efficient deep learning pipeline  

---

## 🧪 Example Workflow

pip install -r requirements.txt  
cd src  
python train.py  
python evaluate.py  
python verify_phase6.py  

---

## 🚨 Common Issues

### CUDA not detected
Set in config.py:
DEVICE = "cpu"

---

### Dataset not loading
- Check folder structure
- Verify paths in config.py

---

### Training is slow
- Reduce batch size
- Use GPU if available

---

## 📦 Pretrained Models

- siamese_unet.pth
- siamese_unet_dice.pth
- siamese_unet_phase6_v1.pth

---

## 👨‍💻 Author

Anuraag S Sarav  
SRM Institute of Science and Technology  

---

## 📝 License

For academic and research use only.