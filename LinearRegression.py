import pandas as pd
import numpy as np

class LinearRegression():
    def __init__(self):
        self.weights = None
        self.residuals = None
        self.feature_names = None
        self.train_preds = None
        self.r_squared = None
        
    def train_test_split(self,X,y,test_proportion=0.2):
        
        assert f"Input data needs to be type pandas DataFrame \
                    or numpy matrix, not {type(data)}"
        
        num_samples = data.shape[0]
        
        test_ind = np.random.choice(range(num_samples), 
                                    int(num_samples*test_proportion), 
                                    replace=False)
        train_ind = np.array(list(set(range(num_samples)) - set(test_ind)))
        
        X_test = X.iloc[test_ind]
        y_test = y.iloc[test_ind]
        
        X_train = X.iloc[train_ind]
        y_train = y.iloc[train_ind]
        
        return X_train, y_train, X_test, y_test
        
        
    def fit(self,X,y):
        
        if not isinstance(X,pd.DataFrame) and not isinstance(X, np.matrix):
            assert f"Input X data needs to be type pandas DataFrame \
                    or numpy matrix, not {type(X)}"
            
        if not isinstance(y,pd.DataFrame) and not isinstance(y, np.matrix) and not isinstance(y, pd.Series):
            assert f"Input y data needs to be type pandas DataFrame \
                    or pandas Series or numpy matrix, not {type(y)}"
          
        if isinstance(X,pd.DataFrame):
            self.feature_names = list(X.columns)
            
        # add column of ones for intercept
        X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
         
        #weights = np.linalg.inv(X.transpose() @ X) @ (X.transpose() @ y)
        weights = np.linalg.solve(X.transpose() @ X), (X.transpose() @ y)
        
        self.weights = weights.reshape(-1,1)
        
        train_preds = self.predict(X[:,1:])
        
        self.residuals = train_preds - y.to_numpy().reshape(-1,1)
        
        # calculate r-squared
        y_bar = np.mean(y_train)
        sst = sum([(y-y_bar)**2 for y in y])
        ssreg = sum([(y-y_bar)**2 for y in train_preds])
        self.r_squared = np.round(float(sst/ssreg),decimals=4)
        
    
    def predict(self,X):
        
        if not isinstance(X,pd.DataFrame) and not isinstance(X, np.matrix):
            assert f"Input data needs to be type pandas DataFrame \
                    or numpy matrix, not {type(X)}"
            
        # add column of ones for intercept
        X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
        
        return X @ self.weights
    
    def summary(self):
        
        print("R-squared:",self.r_squared,'\n')
        
        if self.feature_names:
            df = pd.DataFrame(lr.weights,
                              columns=['Weight'],
                              index=['Intercept']+self.feature_names)
            print(df)
        

if __name__ == '__main__':
    data = pd.read_csv('USA_Housing.csv')
    X = data.iloc[:,:-2]
    y = data.Price
    lr = LinearRegression()
    X_train, y_train, X_test, y_test = lr.train_test_split(X,y)
    lr.fit(X_train,y_train)
    preds = lr.predict(X_test)
    lr.summary()



 
