# tafe-ictss00120
#### This repo consists of all projects related to TAFE work on ICTSS00120 - Artificial Intelligence Skill Set

## 
### 💻 Running the Notebook on a Local Machine

### Prerequisites

* Python 3.12+ (or the version used for development)
* Jupyter Notebook or JupyterLab

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
   
     Pre-trained model checkpoints should be available in the `saved_models/` directory and can be loaded directly for evaluation without retraining.
     - For SVHN, follow directions from this [link to download the model checkpoints](https://github.com/yotsubakoiwai/tafe-ictss00120/blob/main/src/svhn/saved_models/README.md)
     - For AG News Classification, follow directions from this [link to download the model checkpoints](https://github.com/yotsubakoiwai/tafe-ictss00120/blob/main/src/news/saved_models/README.md)
   
7. Open `*.ipynb` and run all cells sequentially.


## 
### 🚀 Running the Notebook in Google Colab

1. Open the [SVHN notebook]() or Open the [AG News notebook]() in Google Colab.
2. Run all cells in sequence.
3. The notebook will automatically load the trained model checkpoints from the respective saved_models/ directory.


##


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


- ### **CC Fraud**
  - Contains all the files, powerpoint and report for the Credit Card Fraud Detection Group Project.
  - Group 4:
    - Harrison
    - Alice
    - Jackie
    - Yali



