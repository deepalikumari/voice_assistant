import datetime
from Speak import Say



def Time():
    time = datetime.datetime.now().strftime("%H:%M")
    Say(time)

def Date():
    date = datetime.date.today()
    Say(date)

def Day():
    day = datetime.datetime.now().strftime("%A")
    Say(day)


def Decisiontree():
    from Speak import Say
    from sklearn import datasets
    import pandas as pd
    import math 
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.metrics._plot.confusion_matrix import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split 
   
    data =datasets.load_iris()
# Extracting Attributes / Features
    X = data.data
# Extracting Target / Class Labels
    y = data.target

# splitting Train and Test datasets
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1, test_size = 0.3) # 70% training and 30% test

# Creating Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)

# Predict Accuracy Score
    y_pred = clf.predict(X_test)
    print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred=clf.predict(X_train)))
    Say('For decision tree classifier train data accuracy is ')
    Say(accuracy_score(y_true = y_train, y_pred=clf.predict(X_train)))
    print("Test data accuracy:",round(accuracy_score(y_true = y_test, y_pred=y_pred)))
    Say('For decision tree classifier test data accuracy is ')
    Say(round(accuracy_score(y_true = y_test, y_pred=y_pred)))



    def train_using_entropy(X_train,X_test,y_train):
  #decision tree with entropy
        clf_entropy = DecisionTreeClassifier(
            criterion = "entropy",random_state = 100,
            max_depth = 3, min_samples_leaf = 5
            )
  #performing training
        clf_entropy.fit(X_train,y_train)
        return clf_entropy

#function to perform training with giniindex
    def train_using_gini(X_train,X_test,y_train):
  #creating the classifier object
            clf_gini = DecisionTreeClassifier(criterion="gini",
                                    random_state = 100,max_depth=3,min_samples_leaf=5)
  
    #performing training
            clf_gini.fit(X_train,y_train)
            return clf_gini

#function to make predictions
    def prediction(X_test,clf_object):
  #prediction on test with giniindex
            y_pred = clf_object.predict(X_test)
            print("Predicted value: ")
            print(y_pred)
            return y_pred

#function to calculate accuracy
    def cal_accuracy(y_test,y_pred):
            print("Confusion Matrix",confusion_matrix(y_test,y_pred))
            Say('Confusion matrix using gini index and entropy are same that is')
            Say(confusion_matrix(y_test,y_pred))

            print("Accuracy",accuracy_score(y_test,y_pred)*100)
            Say('Accuracy using gini index and entropy are also same that is  ')
            Say(round(accuracy_score(y_test,y_pred)*100))
            print("Report",classification_report(y_test,y_pred))


    clf_gini=train_using_gini(X_train,X_test,y_train)
    y_pred_gini = prediction(X_test,clf_gini)
    cal_accuracy(y_test,y_pred_gini)

    Say(Decisiontree())


def PCA():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from Speak import Say
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    iris = datasets.load_iris()
    X = iris.data
    f=iris.feature_names
    Say('performing PCA on iris dataset')
    Say('features names in  iris dataset are')
    Say(f)
   
    # standardizing data
    std_x = StandardScaler().fit_transform(X)

# covariance matrix
    cov_mat_x = np.cov(std_x.T)
    print("Covariance Matrix of X: \n", cov_mat_x)
    

# eigen values and eigen vectors for covariance matrix of train data
    eig_values, eig_vectors = np.linalg.eig(cov_mat_x)
    Say('printing eigen value and vectors of ')
    print("\nEigen Values of X: \n", eig_values)
    print("\nEigen Vectors of X: \n", eig_vectors)
    
    Say(PCA())


    
def regression():
    from Speak import Say
    import numpy as np
    import matplotlib.pyplot as plt
    x=np.array([0,1,2,3,4,5,6,7,8,9])
    y=np.array([1,3,2,5,7,8,8,9,10,12])
    n=np.size(x);
    m_x,m_y=np.mean(x),np.mean(y);
    SS_xy=np.sum(y*x)-n*m_y*m_x;
    SS_xx=np.sum(x*x)-n*m_x*m_x;
    b1=SS_xy/SS_xx;
    b0=m_y-b1*m_x;
    Say('For linear regression result of given input data are')
    print(x)
    print(y)
    print('b0 value is',round(b0));
    Say('beta 0 value is' )
    Say(round(b0))
    print('b1 values is',round(b1));
    Say('beta 1 value is')
    Say(round(b1))
    plt.scatter(x,y,color="m",marker="o",s=30)
    y_pred=b0+b1*x;
    print(y_pred);
    plt.plot(x,y_pred,color="g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    Say('result for Regression Coefficients using analytical formulation')
    import numpy as np
    import matplotlib.pyplot as plt
    x=np.array([0,1,2,3,4,5,6,7,8,9])
    y=np.array([1,3,2,5,7,8,8,9,10,12])
    n=np.size(x);
    SSE=0;
    SSR=0;
    SST=0;
    m_x,m_y=np.mean(x),np.mean(y);
    SS_xy=np.sum(y*x)-n*m_y*m_x;
    SS_xx=np.sum(x*x)-n*m_x*m_x;
    b1=SS_xy/SS_xx;
    b0=m_y-b1*m_x;
    print('b0 value is',round(b0));
    print('b1 values is',round(b1));
    Say('beta 0 value is')
    Say(round(b0))
    Say('beta 1 value is');
    Say(round(b1))
    for i in range(n):
        y_pred[i]=b0+b1*x[i];
        SSE=SSE+pow((y[i]-y_pred[i]),2)
        SSR=SSR+pow((y_pred[i]-m_y),2)
        SST=SST+pow((y[i]-m_y),2)
    print('SSE value is',round(SSE))
    print('SSR value is',round(SSR))
    print('SST value is',round(SST))
    R_Square=SSR/SST;
    print('R_Square value is',round(R_Square));

    Say('Sum Square error value is')
    Say(round(SSE))
    Say('SSR value is')
    Say(round(SSR))
    Say('SST value is')
    Say(round(SST))
    Say('R Square value is')
    Say(round(R_Square))


    Say('RESULT OF Stochastic Gradient Descent to compute coefficients of regression')
    import matplotlib.pyplot as plt
    x=np.array([0,1,2,3,4,5,6,7,8,9])
    y=np.array([1,3,2,5,7,8,8,9,10,12])
    beta0=0;
    beta1=1;
    alpha=0.001;
    n=np.size(x);
    y_expected=np.zeros(n);
    b=[0,0];
    squared_error=10;
    while(squared_error>1):
        for i in range(n):
            y_expected[i]=beta0+beta1*x[i];
            beta0=beta0+alpha*(y[i]-y_expected[i])*1;
            beta1=beta1+alpha*(y[i]-y_expected[i])*x[i];
            b[0]=beta0
            b[1]=beta1
            error=y-y_expected;
            squared_error=np.sum((error**2))*(1/n);
            plt.plot(x,y_expected,color="g")
    print('Squared error',round(squared_error))
    Say('value of squared error is')
    Say(round(squared_error))
    plt.scatter(x,y,color="m",marker="o",s=30)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



    Say(regression())


	




def NonInputExecution(query):
    query = str(query)
    if "time" in query:
        Time()

    elif "date" in query:
        Date()

    elif "day" in query:
        Day()
    
    elif "Decision tree" in query:
        Decisiontree()

    elif "Principal component analysis" in query:
        PCA()
    
    elif "regression" in query:
        regression()
    
    
def InputExecution(tag,query):
    if "wikipedia" in tag:
        name = str(query).replace("who is","").replace("about","").replace("what is","").replace("wikipedia","")
        import wikipedia
        result = wikipedia.summary(name)
        Say(result)

    if "google" in tag:
        query = str(query).replace("google","")
        query = query.replace("search","")
        import pywhatkit
        pywhatkit.search(query)

    

