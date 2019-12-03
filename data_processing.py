# Creator: Hoang-Dung Do

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing as preprocessing
import data_util
import re



# ===================================CLASSIFICATION============================================ #

def diabetic_retinopathy(test_size=0.2):
    data = np.loadtxt("data/classification/messidor_features.arff", delimiter=',', skiprows=24)
    x, y = data[:, :18], data[:, 18]
    return data_util.normalize_split(x, y, test_size)


def credit_card_client(test_size=0.2):
    data = np.loadtxt("data/classification/default_credit_card_clients.csv", delimiter=',', skiprows=2)
    x, y = data[:, 1:24], data[:, 24]
    return data_util.normalize_split(x, y, test_size)


def breast_cancer(test_size=0.2):
    data = np.loadtxt("data/classification/breast_cancer/wdbc.data",
                      delimiter=',', skiprows=0,
                      converters={1: cancer_type_num})
    x, y = data[:, 2:32], data[:, 1]

    return data_util.normalize_split(x, y, test_size)


def cancer_type_num(char):
    if char == b'M':
        return 1
    return 0

def adult():
    def clean_1(x) :
        return x.strip()

    names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex',
            'capital-gain','capital-loss','hours-per-week','native-country','income']
    data = pd.read_csv('data/classification/Adult/adult.data', delimiter=',', header = None, names = names, na_values=['?'])
    test = pd.read_csv('data/classification/Adult/adult.test', delimiter=',', header = None, names = names, skiprows = 1, na_values=['?'])

    for ser in data.select_dtypes(include = object):
        data[ser].map(clean_1)
    for ser in test.select_dtypes(include = object):
        test[ser].map(clean_1)

    pattern= r'([^?\s.]{1,20})' #eliminate spaces(again), ? and period
    re.compile(pattern)
    for ser in data.select_dtypes(include = object):
        data[ser]= data[ser].str.extract(pattern)

    for ser in test.select_dtypes(include = object):
        test[ser]= test[ser].str.extract(pattern)

    data.dropna(inplace = True)
    test.dropna(inplace = True)
    #encode categorical data in data and test
    enc = preprocessing.OrdinalEncoder()
    enc.fit(data.select_dtypes(include = object))
    x_train = enc.transform(data.select_dtypes(include = object))
    x_test  = enc.transform(test.select_dtypes(include = object))
    #eight column is the target variable
    y_train = x_train[:,8]
    y_test  = x_test[:,8]
    #concatenate encoded and scalar data
    x_train = np.concatenate((x_train[:,:8], data.select_dtypes(include = int)), axis =1)
    x_test  = np.concatenate((x_test[:,:8] , test.select_dtypes(include = int)), axis =1)


    return x_train, x_test, y_train, y_test

def plates(test_size=0.2):
    data = np.loadtxt('data/classification/Plates/Faults.NNA', delimiter='\t')
    x, y = data[:, :27], data[:, 27:]
    _y, y_class = np.nonzero(y) #simple coding of target data
    return data_util.normalize_split(x, y_class, test_size)


def yeast(test_size=0.2):

    data = pd.read_csv('data/classification/yeast/yeast.data', delimiter = '\s+', names = range(10))
    #encode target classes as numbers
    enc = preprocessing.OrdinalEncoder()
    enc.fit(data.select_dtypes(include = object))

    x = data.iloc[:,1:9]
    y = enc.transform(data.select_dtypes(include = object))

    return data_util.normalize_split(x, y[:,1], test_size)


def torax(test_size=0.2):

    data = pd.read_csv('data/classification/torax/ThoraricSurgery.arff', names = range(17),skiprows=21)
    x_train, x_test, y_train, y_test= sklearn.model_selection.train_test_split(data.iloc[:,:16], data.iloc[:,16], random_state =0, test_size= 0.2)

    enc = preprocessing.OrdinalEncoder()
    enc.fit(x_train.select_dtypes(include = object))

    en_train = enc.transform(x_train.select_dtypes(include = object))
    en_test  = enc.transform(x_test.select_dtypes(include = object))

    #change (F,T) to (0,1) for target variable
    enc = preprocessing.OrdinalEncoder()
    enc.fit(np.array(y_train).reshape(-1,1))

    y_train = enc.transform(np.array(y_train).reshape(-1,1))
    y_test  = enc.transform(np.array(y_test).reshape(-1,1))

    y_train = y_train.reshape(-1,)
    y_test = y_test.reshape(-1,)

    x_train = np.concatenate((en_train,  x_train.select_dtypes(include = [int, float])), axis =1)
    x_test  = np.concatenate((en_test,   x_test.select_dtypes(include = [int, float])), axis =1)

    return x_train, x_test, y_train, y_test


# =======================================REGRESSION============================================ #

def red_wine_quality(test_size=0.2):
    data = np.loadtxt("data/regression/wine_quality/winequality-red.csv", delimiter=';', skiprows=1)
    x, y = data[:, :11], data[:, 11]

    return data_util.normalize_split(x, y, test_size)

def bike(test_size=0.2):
    data = pd.read_csv('data/regression/bike/hour.csv')
    #instant column is not needed because is a serial number
    x_train, x_test, y_train, y_test= sklearn.model_selection.train_test_split(data.iloc[:,1:16], data.iloc[:,16], random_state =0, test_size= test_size)

    enc = preprocessing.OrdinalEncoder()
    enc.fit(x_train.select_dtypes(include = object))

    en_train = enc.transform(x_train.select_dtypes(include = object))
    en_test  = enc.transform(x_test.select_dtypes(include = object))

    x_train = np.concatenate((en_train, x_train.select_dtypes(include = [int, float])), axis =1)
    x_test = np.concatenate((en_test,   x_test.select_dtypes(include = [int, float])), axis =1)

    return x_train, x_test, y_train, y_test



def student(test_size=0.2):
    data = pd.read_csv('data/regression/student/student-por.csv')

    x_train, x_test, y_train, y_test= sklearn.model_selection.train_test_split(data.iloc[:,1:16], data.iloc[:,16], random_state =0, test_size= test_size)

    enc = preprocessing.OrdinalEncoder()
    enc.fit(x_train.select_dtypes(include = object))

    en_train = enc.transform(x_train.select_dtypes(include = object))
    en_test  = enc.transform(x_test.select_dtypes(include = object))

    x_train = np.concatenate((en_train, x_train.select_dtypes(include = [int, float])), axis =1)
    x_test  = np.concatenate((en_test,   x_test.select_dtypes(include = [int, float])), axis =1)

    return x_train, x_test, y_train, y_test


def concrete(test_size=0.2):

    data = pd.read_excel('data/regression/concrete/Concrete_Data.xls')
    x, y = data.iloc[:,:8], data.iloc[:,8]

    return data_util.normalize_split(x, y, test_size)



def gpu(test_size=0.2):

    data = pd.read_csv('data/regression/gpu/sgemm_product.csv')
    #from the 4 runs, use the average as target
    data['average']=data.iloc[:,14:18].median(axis=1)
    #take logarithm as suggested in data set readme
    data.average = data.average.apply(np.log10)

    x, y = data.iloc[:,:14], data.iloc[:,18]

    return data_util.normalize_split(x, y, test_size)


######################################################################################

def molecular(test_size=0.2):

    data = pd.read_csv('data/regression/molecular/ACT4_competition_training.csv')

    #from the 4 runs, use the average as target
    return data_util.normalize_split(x, y, test_size)


####################################
