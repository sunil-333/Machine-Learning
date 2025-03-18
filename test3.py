from flask import Flask, request
from flask import render_template
import jyserver.Flask as jsf
from disease_predictor import *


from tkinter import *
import numpy as np
import pandas as pd

app = Flask(__name__)

@jsf.use(app)
class App:
    

    #def __init__(self):
            #self.js.document.getElementById("i1").innerHTML = " "
            #self.js.document.getElementById("2").innerHTML = " "
            #self.js.document.getElementById("3").innerHTML = " "

    

    def DecisionTree(self):

                    
                    from sklearn import tree

                    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
                    clf3 = clf3.fit(X,y)
                   
                    
                    s1 = self.js.document.getElementById("s1").value
                    s2 = self.js.document.getElementById("s2").value
                    s3 = self.js.document.getElementById("s3").value
                    s4 = self.js.document.getElementById("s4").value
                    s5 = self.js.document.getElementById("s5").value
                   
                    

                    psymptoms = [s1,s2,s3,s4,s5]
                   
                    for k in range(0,len(l1)):
                        # print (k,)
                        for z in psymptoms:
                            if(z==l1[k]):
                                l2[k]=1

                    inputtest = [l2]
                    predict = clf3.predict(inputtest)
                    predicted=predict[0]

                    h='no'
                    for a in range(0,len(disease)):
                        if(predicted == a):
                            h='yes'
                            break


                    if (h=='yes'):
                        self.js.document.getElementById("1").innerHTML = disease[a]
                        
                    else:
                        self.js.document.getElementById("1").innerHTML = "Not Found"


    def randomforest(self):
                   

                    from sklearn.ensemble import RandomForestClassifier
                    clf4 = RandomForestClassifier()
                    clf4 = clf4.fit(X,np.ravel(y))

                    # calculating accuracy-------------------------------------------------------------------
                    from sklearn.metrics import accuracy_score
                    y_pred=clf4.predict(X_test)
                    print(accuracy_score(y_test, y_pred))
                    print(accuracy_score(y_test, y_pred,normalize=False))
                    # -----------------------------------------------------
                    s1 = self.js.document.getElementById("s1").value
                    s2 = self.js.document.getElementById("s2").value
                    s3 = self.js.document.getElementById("s3").value
                    s4 = self.js.document.getElementById("s4").value
                    s5 = self.js.document.getElementById("s5").value
                   
                    

                    psymptoms = [s1,s2,s3,s4,s5]

                    for k in range(0,len(l1)):
                        for z in psymptoms:
                            if(z==l1[k]):
                                l2[k]=1

                    inputtest = [l2]
                    predict = clf4.predict(inputtest)
                    predicted=predict[0]

                    h='no'
                    for a in range(0,len(disease)):
                        if(predicted == a):
                            h='yes'
                            break

                    if (h=='yes'):
                        self.js.document.getElementById("2").innerHTML = disease[a]
                    else:
                        self.js.document.getElementById("2").innerHTML = "Not Found"


    def NaiveBayes(self):
                    from sklearn.naive_bayes import GaussianNB
                    gnb = GaussianNB()
                    gnb=gnb.fit(X,np.ravel(y))
                                
                    from sklearn.metrics import accuracy_score
                    y_pred=gnb.predict(X_test)
                    print(accuracy_score(y_test, y_pred))
                    print(accuracy_score(y_test, y_pred,normalize=False))
                    # -----------------------------------------------------

                    s1 = self.js.document.getElementById("s1").value
                    s2 = self.js.document.getElementById("s2").value
                    s3 = self.js.document.getElementById("s3").value
                    s4 = self.js.document.getElementById("s4").value
                    s5 = self.js.document.getElementById("s5").value
                   
                    

                    psymptoms = [s1,s2,s3,s4,s5]
                    for k in range(0,len(l1)):
                        for z in psymptoms:
                            if(z==l1[k]):
                                l2[k]=1

                    inputtest = [l2]
                    predict = gnb.predict(inputtest)
                    predicted=predict[0]

                    h='no'
                    for a in range(0,len(disease)):
                        if(predicted == a):
                            h='yes'
                            break

                    if (h=='yes'):
                         self.js.document.getElementById("3").innerHTML = disease[a]
                    else:
                        self.js.document.getElementById("3").innerHTML = "Not Found"


        


@app.route('/')
def index():
    return App.render(render_template('index.html'))

if __name__ == '__main__':
    app.run()