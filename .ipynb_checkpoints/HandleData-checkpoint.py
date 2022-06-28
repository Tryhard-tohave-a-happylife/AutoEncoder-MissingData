import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def missing_method(raw_data, mechanism, method, thre, label='label', state=42) :
    
    label_col = raw_data[label]
    data = raw_data.copy()
    data = data.drop(columns=[label])
    rows, cols = data.shape
    columns, mask = data.columns.tolist(), None
    
    # missingness thresholds
    t = thre
    np.random.seed(state) 

    if mechanism == 'mcar' :
    
        if method == 'uniform' :
            # uniform random vector
            v = np.random.uniform(size=(rows, cols))
            # missing values where v<=t
            mask = (v<=t)
            data[mask] = np.nan

        elif method == 'random' :
            # only half of the attributes to have missing value
            missing_cols = np.random.choice(cols, cols//2)
            c = np.zeros(cols, dtype=bool)
            c[missing_cols] = True

            # uniform random vector
            v = np.random.uniform(size=(rows, cols))

            # missing values where v<=t
            mask = (v<=t)*c
            data[mask] = np.nan

        else :
            print("Error : There are no such method")
            raise
    
    elif mechanism == 'mnar' :
        cat_cols = [i for i in data.select_dtypes(include='object').columns]
        copy_data = data.copy()
        for each in cat_cols:
            copy_data[each] = LabelEncoder().fit_transform(copy_data[each])
        if method == 'uniform' :
            # randomly sample two attributes
            sample_cols = np.random.choice(cols, 2)
            
            # calculate ther median m1, m2
            m1, m2 = np.median(copy_data.iloc[:,sample_cols], axis=0)
            # uniform random vector
            v = np.random.uniform(size=(rows, cols))

            # missing values where (v<=t) and (x1 <= m1 or x2 >= m2)
            m1 = copy_data.iloc[:,sample_cols[0]] <= m1
            m2 = copy_data.iloc[:,sample_cols[1]] >= m2
            m = (m1*m2)[:, np.newaxis]

            mask = m*(v<=t)
            data[mask] = np.nan


        elif method == 'random' :
            # only half of the attributes to have missing value
            missing_cols = np.random.choice(cols, cols//2)
            c = np.zeros(cols, dtype=bool)
            c[missing_cols] = True

            # randomly sample two attributes
            sample_cols = np.random.choice(cols, 2)

            # calculate ther median m1, m2
            m1, m2 = np.median(copy_data.iloc[:,sample_cols], axis=0)
            # uniform random vector
            v = np.random.uniform(size=(rows, cols))

            # missing values where (v<=t) and (x1 <= m1 or x2 >= m2)
            m1 = copy_data.iloc[:,sample_cols[0]] <= m1
            m2 = copy_data.iloc[:,sample_cols[1]] >= m2
            m = (m1*m2)[:, np.newaxis]

            mask = m*(v<=t)*c
            data[mask] = np.nan

        else :
            print("Error : There is no such method")
            raise
    
    else :
        print("Error : There is no such mechanism")
        raise
    
    # if mask is not None:
    #     data = pd.DataFrame(
    #         data = data, columns=columns
    #     )
    #     print(data)
    data[label] = label_col
    return data, mask