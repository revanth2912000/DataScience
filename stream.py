import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def user_input_features():
    SEX = st.sidebar.selectbox('Sex',('Male','Female'))
    PC = st.sidebar.selectbox('Pclass',(1,2,3))
    Age = st.sidebar.number_input('Age')
    Fare=st.sidebar.number_input('Fare')
    Embarked=st.sidebar.selectbox('Embarked',('Q','C','S'))
    Family=st.sidebar.number_input('Siblings/Spouse')
    Parents=st.sidebar.number_input('Parents')
    data = {'Sex':SEX,
            'Pclass':PC,
            'Age':Age,
            'Fare':Fare,
            'Embarked':Embarked,
            'SibSp':Family,
            'Parch':Parents }
    features = pd.DataFrame(data,index = [0])
    return features 

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

TitanicTest = pd.read_csv("C:\\Users\\revan\\Sravani Mam\\DSA\\Assignments\\Titanic_test.csv")
TitanicTrain =pd.read_csv("C:\\Users\\revan\\Sravani Mam\\DSA\\Assignments\\Titanic_train.csv")
data = pd.concat([TitanicTest, TitanicTrain], axis=0)
data = data.reset_index(drop=True)

data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
train_data_len=len(TitanicTest)
train_loc=data.iloc[:train_data_len]
test_loc=data.iloc[train_data_len:]
x=train_loc.drop(columns=['PassengerId','Survived','Name','Sex','Ticket','Embarked','Cabin'],axis=1)
y=train_loc['Pclass']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
st.subheader('prediction')
st.write(y_pred)
accuracy=model.score(x_test,y_test)
st.subheader('accuracy')
st.write(accuracy)
report = classification_report(y_test, y_pred)
fig=sns.pairplot(data)
st.pyplot(fig)