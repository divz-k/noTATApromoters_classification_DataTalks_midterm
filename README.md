# TATA-less Promoter Classification

## Project Overview
Promoters are those regions of the DNA that regulate gene expression. They are often defined by specific sequence features, like the presence of a TATA box. Identifying promoters is an important cornerstone in molecular biology. We do not have an established method for recognising promoter sequences, especially for those promoters without a canonical TATA box. Traditional motif-based approaches rely on the TATA sequence, leaving many promoters unrecognized.  In this project, I developed a machine learning model capable of predicting promoter regions directly from DNA sequences, when the canonical TATA motif is absent. The project is designed for a DataTalks Machine Learning Zoomcamp midterm. 

Currently, there are far more advanced models to perform the same task, that use deep neural networks and achieve much higher rates of accuracy than what is possible by standard ML models. In comparison to the state of the art technologies, I have been limited by model simplicity and computational power: so this model isn't likely to identify something novel that hasn't been computationally identified beofre. Yet, I chose this project for the following reasons:

- we can set the baseline of how much is achieveable without deep neural networks
- we can identify those sequence features primarily resposible for the distinction between both categories
 
The workflow implements a full machine-learning pipeline:

- I use the dataset from the Nucleotide transformers study[https://www.nature.com/articles/s41592-024-02523-z]. I have selected TATA less promoters among all the DNA data options
- Deriving features from the DNA (GC content and all possible 5mers) to input into the ML model
- exploratory data analysis (EDA) on the features derived
- model training (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- hyperparameter tuning
- evaluation and interpretation

## Key Steps:
- Feature Extraction:
-- Counts of 5-mers (sequence + reverse complement) → 512 features
-- GC content → 1 feature
- Exploratory Data Analysis (EDA): Correlation, mutual information, PCA to understand feature relationships.
- Model Training & Evaluation:
-- Logistic Regression (best performer)
-- Decision Tree
-- Random Forest
-- XGBoost
- Model Persistence: Trained models saved with joblib.
- Deployment: FastAPI application, containerized using Docker, deployed on Render.

## Installation
1) Clone Repository
```
git clone https://github.com/divz-k/noTATApromoters_classification_DataTalks_midterm tataless_promClassification
cd tataless_promClassification

```

2) Create and activate Environment
```
python -m venv promoter_env
# Linux/Mac
source promoter_env/bin/activate
# Windows
promoter_env\Scripts\activate
#install dependencies
pip install -r requirements.txt
```

## Running locally by taking from Docker
1) Use the script convertDNASeqToX.ipynb to make the X_json from the input DNA sequence. Just paste the required DNA sequence (containing only A/T/G/C) into the script, and run all the code blocks.
2) At the end, you will see the X_json printed: this is the input X features in a json format that can be pasted as the input. It will look like this: {"AAAAA":1.0,"AAAAT":0.0, ... } (513 features). (The first part on making the X_json is demonstrated in ScreenRecording_testingDeployment.mov
3) Copy this and use in the following code.
```
#build docker image
docker build -t promoter-classifier .
# run docker containerised
docker run -p 8000:8000 promoter-classifier
#The API will be accessible at http://localhost:8000
#Test Locally
curl -X POST "http://localhost:8000/predict_regression" \
     -H "Content-Type: application/json" \
     -d '{"AAAAA":1.0,"AAAAT":0.0, ... }'
```

## Running the deployed API
1) Use the script convertDNASeqToX.ipynb to make the X_json as described before
2) Then go to script sendReq.ipynb. Paste the required X_json and run the blocks. You should see the output prediction (eg: {'prediction': 1}).
3) There is a video Screen Recording describing this process: ScreenRecording_testingDeployment.mov. Github doesn't allow you to play it, so you have to download and watch the video.

## Files in Repository
- code.ipynb – Feature extraction, model training, and evaluation
- models/ – Trained models:
-- regression_model.pkl
-- tree_model.pkl
-- forest_model.pkl
-- xgb_model.pkl
- Dockerfile – For building the container
- requirements.txt – Dependencies
- app.py – FastAPI application
- convertDNASeqToX.ipynb - make the X feature matrix for any DNA sequence
- sendReq.ipynb - takes the X feature matrix (from the json format) and sends request to the deployed API, and returns the prediction.

## Notes:
- The data analysis, EDA, model evaluations, performance metrics and the basis for the final model choice are all documented in the markdowns on the code.ipynb script. 
- How to use this model has been described in the README
