# TUMOR TYPE PREDICTION

import pandas as pd
import numpy as np
import joblib
from joblib import load
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    

def model_loading_and_work(filename, model_name):
    try:
        model = joblib.load(filename)
        probability = prediction_function(model, X_array, model_name)
    except Exception as e:
        probability = pd.DataFrame(np.zeros(len(X_array[0:]), dtype=float, order='C'), columns=[model_name+'_predict']) 
    return probability


def prediction_function(model, X_test, name):
    # creates dataframe with 3 columns: probability_B, probability_M, predicted tumor type (B or M)
    # predicted types are marked according to probability (>=0.99*, >=0.999**) 
    
    #  predicted classes 
    tumor_type = model.predict(X_test)
    tumor_type = pd.DataFrame(tumor_type)
    tumor_type = tumor_type[0].replace(0, 'B').replace(1, 'M')

    # prediction probabilities
    probability = model.predict_proba(X_test)
    probability = pd.DataFrame(probability, columns=[f'{name}_prob_B', f'{name}_prob_M'] )
    probability[f'{name}_predict'] = tumor_type

    # stars
    for i in range(len(tumor_type)):
        for p in [0.99, 0.999]:
            if probability[f'{name}_prob_B'].iloc[i] >= p or probability[f'{name}_prob_M'].iloc[i] >= p:
                probability[f'{name}_predict'].iloc[i] = probability[f'{name}_predict'].iloc[i] + '*'

    return probability

def save_file_function(content, file_name, file_format='csv'):  
    # saves files filename(1).csv, filename(2).csv...
    
    import os    
    n = 1
    while True:  
        name = file_name+'('+str(n)+').'+file_format
        if os.path.isfile(name) == True:
            n += 1 
        else:
            try:
                content.to_csv(name)   
                print('object saved:', os.getcwd()+'\\'+name)
                break
            except Exception:
                print('sorry, object not saved')
                break

                
# DATASET LOADING and PREPARING 

data = pd.read_csv('BCW_dataset.csv')     # origilal format

# dataset preprocessing: the best features
features = ['x.radius_mean', 'x.texture_mean', 'x.smoothness_mean', 'x.compactness_mean', 
         'x.concavity_mean', 'x.concave_pts_mean', 'x.symmetry_worst', 'x.fractal_dim_worst']
y = data[['y']].replace('B', 0).replace('M', 1)
X = data[features]

# dataset preprocessing: normalization
max_norm = list(pd.read_csv('max_norm.csv').iloc[0, 1:9])
for i in range(len(features)):
    X[features[i]] = X[features[i]].apply(lambda x: x/max_norm[i])

X_array = np.array(X)
y_array = np.array(y)

                
# PREDICTION WITH THE BEST MODELS

# linear Discriminant Analysis
lda_probability = model_loading_and_work('model_LDA.joblib', 'LDA')

# Quadratic Discriminant Analysis
qda_probability = model_loading_and_work('model_QDA.joblib', 'QDA')

# Stacking
stacking_model_probability = model_loading_and_work('model_Stack.joblib', 'Stack')

# table containing predictions for all models
result_table = pd.concat([lda_probability, qda_probability, stacking_model_probability, y], axis=1)

save_file_function(result_table, 'BC_prediction')