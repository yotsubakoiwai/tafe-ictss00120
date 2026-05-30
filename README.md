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
  a. For SVHN, follow this [link to download the SVHN dataset](https://github.com/yotsubakoiwai/tafe-ictss00120/blob/main/src/svhn/data/README.md)
  b. For AG News, follow this [link to download the AG News Classification dataset](https://github.com/yotsubakoiwai/tafe-ictss00120/blob/main/src/news/data/README.md)
     
4. Saved Models
   Pre-trained model checkpoints should be available in the `saved_models/` directory and can be loaded directly for evaluation without retraining.
   a. For SVHN, follow this [link to download the model checkpoints]()
   b. For AG News Classification, follow this [link to download the model checkpoints]()
   
5. Open `*.ipynb` and run all cells sequentially.


## 

### ⚠️both svhn & news saved_models can all be downloaded from this [link](https://drive.google.com/drive/folders/1p-nsHgzaz1jSINvlfJm1XxR7XSUgvgol?usp=sharing)

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



