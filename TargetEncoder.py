from sklearn import base
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import numpy.ma as ma

class KFoldTargetEncoderTrain(base.BaseEstimator,base.TransformerMixin):
    def __init__(self,colnames,targetName,
                  n_fold=5, verbosity=True,
                  discardOriginal_col=False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold,
                   shuffle = True, random_state=2019)
        # print(kf)
        col_mean_name = self.colnames + '_' + 'tar'
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            #train set and validation set
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)
                                     [self.targetName].mean())
            #test set you take the average of the target values of all samples that have a given category in the entire train set.
            # X[col_mean_name].fillna(mean_of_target, inplace = True)
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName,                    
                   ma.corrcoef(ma.masked_invalid(X[self.targetName].values), ma.masked_invalid(encoded_feature))[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X