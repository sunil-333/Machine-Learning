from flask import Flask, request
from flask import render_template
import jyserver.Flask as jsf



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

                    l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
                    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
                    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
                    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
                    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
                    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
                    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
                    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
                    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
                    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
                    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
                    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
                    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
                    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
                    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
                    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
                    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
                    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
                    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
                    'yellow_crust_ooze']

                    disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
                    'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
                    ' Migraine','Cervical spondylosis',
                    'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
                    'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
                    'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
                    'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
                    'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
                    'Impetigo']

                    l2=[]
                    for x in range(0,len(l1)):
                        l2.append(0)

                    # TESTING DATA df -------------------------------------------------------------------------------------
                    df=pd.read_csv("Training.csv")

                    df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
                    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
                    'Migraine':11,'Cervical spondylosis':12,
                    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
                    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
                    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
                    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
                    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
                    'Impetigo':40}},inplace=True)

                    # print(df.head())

                    X= df[l1]

                    y = df[["prognosis"]]
                    np.ravel(y)
                    # print(y)

                    # TRAINING DATA tr --------------------------------------------------------------------------------
                    tr=pd.read_csv("Testing.csv")
                    tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
                    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
                    'Migraine':11,'Cervical spondylosis':12,
                    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
                    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
                    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
                    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
                    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
                    'Impetigo':40}},inplace=True)

                    X_test= tr[l1]
                    y_test = tr[["prognosis"]]
                    np.ravel(y_test)
                    # ------------------------------------------------------------------------------------------------------

                    from sklearn import tree

                    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
                    clf3 = clf3.fit(X,y)

                    # calculating accuracy-------------------------------------------------------------------
                    from sklearn.metrics import accuracy_score
                    y_pred=clf3.predict(X_test)
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
                    l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
                    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
                    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
                    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
                    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
                    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
                    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
                    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
                    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
                    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
                    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
                    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
                    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
                    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
                    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
                    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
                    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
                    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
                    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
                    'yellow_crust_ooze']

                    disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
                    'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
                    ' Migraine','Cervical spondylosis',
                    'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
                    'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
                    'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
                    'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
                    'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
                    'Impetigo']

                    l2=[]
                    for x in range(0,len(l1)):
                        l2.append(0)

                    # TESTING DATA df -------------------------------------------------------------------------------------
                    df=pd.read_csv("Training.csv")

                    df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
                    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
                    'Migraine':11,'Cervical spondylosis':12,
                    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
                    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
                    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
                    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
                    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
                    'Impetigo':40}},inplace=True)

                    # print(df.head())

                    X= df[l1]

                    y = df[["prognosis"]]
                    np.ravel(y)
                    # print(y)

                    # TRAINING DATA tr --------------------------------------------------------------------------------
                    tr=pd.read_csv("Testing.csv")
                    tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
                    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
                    'Migraine':11,'Cervical spondylosis':12,
                    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
                    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
                    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
                    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
                    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
                    'Impetigo':40}},inplace=True)

                    X_test= tr[l1]
                    y_test = tr[["prognosis"]]
                    np.ravel(y_test)
                    # ------------------------------------------------------------------------------------------------------

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
                    l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
                    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
                    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
                    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
                    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
                    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
                    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
                    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
                    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
                    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
                    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
                    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
                    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
                    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
                    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
                    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
                    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
                    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
                    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
                    'yellow_crust_ooze']

                    disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
                    'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
                    ' Migraine','Cervical spondylosis',
                    'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
                    'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
                    'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
                    'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
                    'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
                    'Impetigo']

                    l2=[]
                    for x in range(0,len(l1)):
                        l2.append(0)

                    # TESTING DATA df -------------------------------------------------------------------------------------
                    df=pd.read_csv("Training.csv")

                    df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
                    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
                    'Migraine':11,'Cervical spondylosis':12,
                    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
                    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
                    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
                    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
                    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
                    'Impetigo':40}},inplace=True)

                    # print(df.head())

                    X= df[l1]

                    y = df[["prognosis"]]
                    np.ravel(y)
                    # print(y)

                    # TRAINING DATA tr --------------------------------------------------------------------------------
                    tr=pd.read_csv("Testing.csv")
                    tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
                    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
                    'Migraine':11,'Cervical spondylosis':12,
                    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
                    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
                    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
                    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
                    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
                    'Impetigo':40}},inplace=True)

                    X_test= tr[l1]
                    y_test = tr[["prognosis"]]
                    np.ravel(y_test)
                    # ------------------------------------------------------------------------------------------------------
                    from sklearn.naive_bayes import GaussianNB
                    gnb = GaussianNB()
                    gnb=gnb.fit(X,np.ravel(y))

                    # calculating accuracy-------------------------------------------------------------------
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