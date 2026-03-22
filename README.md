# ViPGaRe: Video-informed Physical Gaussian Splatting for Dynamic Scene Reconstruction
<img width="1911" height="811" alt="pipline4" src="https://github.com/user-attachments/assets/7111b503-0e38-4675-a3ba-25f898f42972" />

## ⚙️ Installation
```shell script
git clone https://github.com/DustSettled/VeloGauss.git --recursive
cd VeloGauss

### CUDA 12.4
conda env create -f env.yml
conda activate veloGauss

# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# install gaussian requirements
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/simple-knn
```

## 💾 Datasets
All the datasets will be uploaded soon. We organize the dataset following [D-NeRF](https://github.com/albertpumarola/D-NeRF) convention.
We split the dataset as:
- **train**: contains the frames within observed time interval, used for training the model.
- **val**: contains the frames within observed time interval but for novel views, used for evaluating *novel-view interpolation*.
- **test**: contains the frames in unobserved **future** time for both observed and novel views, used for evaluating *future extrapolation*.

Datasets can be downloaded from HuggingFace: 
- [Dynamic Objects](https://huggingface.co/datasets/scintigimcki/DynamicObjects)
- [Dynamic Indoor Scenes](https://huggingface.co/datasets/scintigimcki/DynamicIndoorScenes)

## 🔑 Train
```
bash train_eval.sh
```

