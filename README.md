***Still in progress***


# GLIP-LOC : Grounded Image-Langugage Alignment for Geo-Localization
![main_fig](https://github.com/user-attachments/assets/d5bbff8a-f203-40a7-a637-a9930de93d8e)

This repository implements multimodal contrastive learning methods for visual geo-localization, focusing on learning representations from satellite and ground images. The approach leverages image-text  embeddings to improve spatial and semantic understanding for better retrieval performance.

## 📂 Repository Structure
```

Directory Structure:

/glip-loc
├── configs
│   ├── default.yaml
│   └── eval.yaml 
├── notebooks # Jupyter notebooks for data analysis and visualization
├── src
│   ├── data # Data processing and dataloaders
│   ├── models # Model architectures
│   ├── training
│   │   ├── __init__.py
│   │   ├── eval.py
│   │   ├── hooks.py
│   │   ├── optimizers.py
│   │   ├── schedulers.py
│   │   ├── trainer.py # Main training class
│   │   └── utils.py
│   ├── utils
│   ├── evaluate_retrieval.py # Evaluation script
│   └── main.py 
├── .gitignore
├── README.md
└── requirements.txt
```

## 🛠 Training

```
python src/main.py --config configs/default.yaml
```
## 📊 Evaluation

```
python src/evaluate_retrieval.py --config configs/eval.yaml --checkpoint path/to/checkpoint.pth
```
