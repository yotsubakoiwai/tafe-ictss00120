# tafe-ictss00120
#### This repo consists of all projects related to TAFE work on ICTSS00120 - Artificial Intelligence Skill Set
* Project 1 - Credit Card Detection Group Project
* Project 2 - SVHN & AG News Classification

The following is the folder structure of this repo:
##
- ### **CC Fraud**
  - Contains all the files, powerpoint and report for the Credit Card Fraud Detection Group Project.
  - Group 4:
    - Harrison
    - Alice
    - Jackie
    - Yali
      

- ### **SVHN**
  - Contains all files, notebook, plots, saved_models and report for the Street View House Number (SVHN) project part of AT4.
  - The following are the file hierarchy:
      ```text
      src/
        │
        └── svhn/
            │
            ├── data/
            │   ├── *.csv
            │   └── README.md (contains instructions on how & where to download needed datasets)
            │
            ├── plots/
            │   ├── *.png
            │   └── *.jpg
            │
            ├── saved_models/
            │   ├── *.pth
            │   ├── *.csv
            │   └── README.md (contains instructions on how & where to download saved trained models)  
            │
            ├── *.ipynb
            └── *.pdf
        │
        └── news/            
      ```


- ### **AG News Classification**
  - Contains all files, notebook, plots, saved_models and report for the AG News Classification project part of AT4.
  - The following are the file hierarchy:
      ```text
      src/
        │
        ├── svhn/
        │
        └── news/
            │
            ├── data/
            │   ├── *.csv
            │   └── README.md (contains instructions on how & where to download needed datasets)
            │
            ├── plots/
            │   ├── *.png
            │   └── *.jpg
            │
            ├── saved_models/
            │   ├── *.pth
            │   ├── *.csv
            │   └── README.md (contains instructions on how & where to download saved trained models)  
            │
            ├── *.ipynb
            └── *.pdf
      ```





## 
### For the SVHN & AG News Classification Project, there are 2 options how to run the notebook:
### 💻 Option 1: Running the Notebook on a Local Machine

### Prerequisites
* Python 3.11+ (or the version used for development)
* Jupyter Notebook or JupyterLab
* Sufficient available disk space, as trained model checkpoints may be large

### Setup
1. Clone the repository:
  ```bash
  git clone https://github.com/yotsubakoiwai/tafe-ictss00120.git
  cd YOUR_REPOSITORY
  ```

2. Launch Jupyter Notebook:
  ```bash
  jupyter notebook
  ```
  
  or
  
  ```bash
  jupyter lab
  ```

3. For respective dataset,
    - For SVHN, follow directions from this [link to download the SVHN dataset](https://github.com/yotsubakoiwai/tafe-ictss00120/blob/main/src/svhn/data/README.md)
    - For AG News, follow directions from this [link to download the AG News Classification dataset](https://github.com/yotsubakoiwai/tafe-ictss00120/blob/main/src/news/data/README.md)
     
5. Saved Models
   
     Pre-trained model checkpoints should be available in the `saved_models/` directory and can be loaded directly for evaluation **<ins>WITHOUT RETRAINING</ins>**.
   
    ⚠️⚠️ Note: If the saved checkpoints are not downloaded locally, the model will train from scratch!
     - For SVHN, follow directions from this [link to download the model checkpoints](https://github.com/yotsubakoiwai/tafe-ictss00120/blob/main/src/svhn/saved_models/README.md)
     - For AG News Classification, follow directions from this [link to download the model checkpoints](https://github.com/yotsubakoiwai/tafe-ictss00120/blob/main/src/news/saved_models/README.md)
   
7. Open `*.ipynb` and run all cells sequentially.


## 
### 🚀 Option 2: Running the Notebook in Google Colab

1. SVHN
    - All the necessary files are in this [Google Drive link](https://drive.google.com/drive/folders/1psm97y5tEz35gR19-cLFp-rTjhWC2Ngl?usp=sharing)
        - You should see a similar directory hierarchy as above.
    - Right-click on `20111144_Jackie Custodio_SVHN_MLPvsCNN - using MixerMLP & ResNet50 v3.ipynb` file and open with Google Colaboratory.
    - The notebook has already been run so all cell outputs are going to be displayed.
    - Else, if wanting to run again, just `Restart session and run all`

2. AG News
    - All the necessary files are in this [Google Drive link](https://drive.google.com/drive/folders/1C-SMsOG5GaxstJ6j0-82rF6SEGWfj9me?usp=sharing)
        - You should see a similar directory hierarchy as above.
    - Right-click on `20111144_Jackie Custodio_NewsClassification_RNNvsTransformer - using BiLSTM & Ensemble v12.ipynb` file and open with Google Colaboratory.
    - The notebook has already been run so all cell outputs are going to be displayed.
    - Else, if wanting to run again, just `Restart session and run all`






