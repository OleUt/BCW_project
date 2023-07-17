### BCW PROJECT
Breast Cancer Wisconsin Diagnostic Dataset, classification Malignant (cancer) vs Benign (non cancer) Tumors.
569 samples, 30 features (morphology data on tumors: mean value (10), standard error of mean (10), the worst feature value (10)

#### input file:
BCW_dataset.csv

#### BCW_data_preprocesssing.ipynb:
- selection of the best features
- data normalization
- creates files: preprocessed_data(n).csv, max.norm.csv (contains max feature values for further new data MAX normalization)

#### BCW_model.ipynb:
- testing classification models (5 simple, 1 ensembly (Stacking))
- creates files with generated models: model_LGR.joblib, model_LDA.joblib, model_QDA.joblib, model_SCV.joblib, model_GNB.joblib, model_Stack.joblib

#### BCW_prediction.ipynb:
- makes prediction using the best models
- creates BC_prediction(n).csv file

#### BC_prediction(n).csv:
contains results from the best classification models (LDA, QDA, Stack):
- prediction (B or M), marked with * for P>0.99 and ** for P>0.999 
- probability for B
- probability for M
