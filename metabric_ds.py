

from sklearn.datasets import load_boston,load_breast_cancer
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import numpy as np
from sklearn.metrics import r2_score
np.random.seed(1)
import lime
import lime.lime_tabular
from lime import submodular_pick

import os, sys
import datetime
import re
start_time = str(datetime.datetime.now().time())

outfolder = "METABRIC_DS"
os.makedirs(outfolder, exist_ok=True)

#### prepare the data
import pandas as pd

inputfile= os.path.join('/home','marie','Documents','FREITAS_LAB',
                            'VAE_tutos','CancerAI-IntegrativeVAEs',
                            'data','MBdata_33CLINwMiss_1KfGE_1KfCNA.csv')


predictLab = "ER_Status"

# training data
df=pd.read_csv(inputfile)

n_samp = df.shape[0]
n_genes = sum(['GE_' in x for x in df.columns])

mrna_data = df.iloc[:,34:1034].copy().values 
gene_names = [re.sub('GE_', '', x) for x in df.columns[34:1034]]
# the values after are CNA, the values before are clinical data
# copy() for deep copy
# values to convert to multidim array
mrna_data2= df.filter(regex='GE_').copy().values
assert mrna_data2.shape == mrna_data.shape

mrna_data_scaled = (mrna_data - mrna_data.min(axis=1).reshape(-1,1))/ \
(mrna_data.max(axis=1)-mrna_data.min(axis=1)).reshape(-1,1)

# will use the 'ER_Status' as condition
tokeep = np.where( (df[covarLab].values == "pos") | (df[covarLab].values == "neg"))
mrna_data_filt = mrna_data_scaled[list(tokeep[0]),:]

sample_labels = df[covarLab].values[tokeep]

assert mrna_data_filt.shape[0] + df[df[covarLab] == "?"].shape[0] == mrna_data_scaled.shape[0]

input_data = mrna_data_filt
c_data = df[covarLab][list(tokeep[0])]
assert len(c_data) == mrna_data_filt.shape[0]
c_data_bin = np.vectorize(np.int)(c_data == "pos")
#c_data_bin = np.vectorize(np.float)(c_data == "pos")
assert np.all((c_data_bin == 1) | (c_data_bin==0))































#load example dataset
#boston = load_boston()
cancer= load_breast_cancer()

#print a description of the variables
#print(boston.DESCR)
print(cancer.DESCR)

#train a regressor
# rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)
# train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(boston.data, boston.target, train_size=0.80, test_size=0.20)
# rf.fit(train, labels_train);

#train a classifier
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(cancer.data, cancer.target, train_size=0.80, test_size=0.20)
rf.fit(train, labels_train);
# train.shape
# Out[4]: (455, 30)
# test.shape
# Out[5]: (114, 30)
# labels_train.shape
# Out[6]: (455,)
# labels_test.shape
# Out[7]: (114,)
# train
# Out[21]: 
# array([[1.799e+01, 2.066e+01, 1.178e+02, ..., 1.974e-01, 3.060e-01,
#         8.503e-02],
#        [2.029e+01, 1.434e+01, 1.351e+02, ..., 1.625e-01, 2.364e-01,
#         7.678e-02],


# generate an "explainer" object
# categorical_features  = np.argwhere(np.array([len(set(boston.data[:,x])) for x in range(boston.data.shape[1])]) <= 10).flatten()
# explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=boston.feature_names, class_names=['price'], categorical_features=categorical_features, 
#                                                    verbose=False, mode='regression',discretize_continuous=False,feature_selection='none')

categorical_features = np.argwhere(np.array([len(set(cancer.data[:,x]))\
                                             for x in range(cancer.data.shape[1])]) <= 10).flatten()
# cancer.data.shape
# Out[10]: (569, 30)

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=cancer.feature_names, 
                                                   categorical_features=categorical_features, 
                                                   class_names=cancer.target_names,
                                                   verbose=False, mode='classification',
                                                   discretize_continuous=False,
                                                   feature_selection='none')

# cancer.feature_names
# Out[11]: 
# array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#        'mean smoothness', 'mean compactness', 'mean concavity',....
# cancer.feature_names.shape
# Out[12]: (30,)
# cancer.target_names
# Out[13]: array(['malignant', 'benign'], dtype='<U9')



#generate an explanation for testing..
i = 39
exp = explainer.explain_instance(test[i], rf.predict_proba,#num_features=13,
                                 model_regressor='Bay_info_prior',
                                 num_samples=1000,
                                 #labels=labels_test[i],
                                 top_labels=2)#'non_Bay' 'Bay_non_info_prior' 'Bay_info_prior'
#exp.show_in_notebook(show_table=True)

fig = exp.as_pyplot_figure(label=0)
fig = exp.as_pyplot_figure(label=1)
# -


