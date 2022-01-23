
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


st.title('Loan predition')



def main():
        Gender=st.selectbox('Gender',(0,1))
        married=st.selectbox('married',(0,1))
        Dependent=st.selectbox('Dependent',(0,1))
        Education=st.selectbox('Education',(0,1))
        Self_Employed=st.selectbox('Self_Employed',(0,1))
        income=st.slider('income in Rs',150,81000)
        co_income=st.slider('Co income in Rs',0,45000)
        loan_amount=st.slider('loan_amount in thousand',9,700)
        loan_amount_term=st.slider('loan_amount_term in thousand',12,480)
        Credit_history=st.selectbox('Credit history',(0,1))
        property_area=st.selectbox('property where u live',(0,1,2))
            


    
        data={'Gender':Gender,
          'married':married,"Dependent":Dependent,
              'Education':Education,'Self_Employed':Self_Employed,'income':income,
          'co_income':co_income,'loan_amount':loan_amount,'loan_amount_term':loan_amount_term,'Credit_history':Credit_history,
          'property_area':property_area,}
        feature=pd.DataFrame(data,index=[1])
        st.write(feature)




        load_clf=pickle.load(open('Logistic.pkl','rb'))
        prediction=load_clf.predict(feature)

        st.write('0  tells us our application is failed  ')
        st.write(' 1 tell us our application is passed')

        df1=pd.DataFrame(prediction,index=['prediction by our model']
                       )
        st.write(df1)


main()
