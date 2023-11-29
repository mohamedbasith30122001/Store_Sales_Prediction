# Importing Libraries
import pandas as pd
import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setting up page configuration
icon = Image.open(r"C:\Users\God\Desktop\phonepe pulse Project folder\ML_final_sales_predict_project\sales_png.jfif")
st.set_page_config(page_title= "Store Sales Prediction| By Mohamedbasith A",
page_icon=icon,
layout= "wide",initial_sidebar_state= "expanded",
menu_items={'About': """# This dashboard app is created by *Mohamedbasith A*!Data has been gathered from CSV file"""})
st.markdown("<h1 style='text-align:center; color:CornflowerBlue;'>Store Sales Prediction</h1>", unsafe_allow_html=True)

#hide the streamlit main and footer
hide_default_format = """
       <style>
       MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# SETTING-UP BACKGROUND IMAGE
def setting_bg():
    st.markdown(f""" 
    <style>
        .stApp {{
            background-image: radial-gradient( circle farthest-corner at 10% 20%,  rgba(234,249,249,0.67) 0.1%, rgba(239,249,251,0.63) 90.1% );
            background-size: cover;
            transition: background 0.5s ease;
        }}
        .stButton>button {{
            color: #f3f3f3;
            background-color: #328EBD;
            transition: all 0.3s ease-in-out;
        }}
        .stButton>button:hover {{
            color: #4e4376;
            background-color:#8BD8FF ;
        }}
    </style>
    """,unsafe_allow_html=True) 
setting_bg()

#Creating option menu in the menu bar
selected = option_menu(None,["Home","Analysis & Prediction","About"],
                        icons=["house","bar-chart","toggles"],
                        default_index=0,
                        orientation="horizontal",
                        styles={"nav-link": {"font-size": "25px", "text-align": "centre", "margin": "0px", "--hover-color": "#8BD8FF", "transition": "color 0.3s ease, background-color 0.3s ease"},
                                "icon": {"font-size": "25px"},
                                "container" : {"max-width": "6000px", "padding": "6px", "border-radius": "5px"},
                                "nav-link-selected": {"background-color": "#328EBD", "color": "white"}})

# Load the CSV files into pandas dataframes
df1=pd.read_csv(r'C:\Users\God\Desktop\phonepe pulse Project folder\ML_final_sales_predict_project\stores_data_set.csv')
df2=pd.read_csv(r'C:\Users\God\Desktop\phonepe pulse Project folder\ML_final_sales_predict_project\sales_data_set - sales_data_set.csv')
df3=pd.read_csv(r'C:\Users\God\Desktop\phonepe pulse Project folder\ML_final_sales_predict_project\Features_data_set.csv')

df4=df2.merge(df3, on=['Store', 'Date','IsHoliday'], how='left')
df5=df1.merge(df4, on=['Store'], how='left')
df5.drop('Date', axis=1, inplace=True)
df5.fillna(0, inplace=True)
df=df5
 
eng=OrdinalEncoder()
df['Type']=eng.fit_transform(df[['Type']]) # Type---> 0 - A, 1 - B, 2 - C
df['IsHoliday']=eng.fit_transform(df[['IsHoliday']]) # IsHoliday---> 0 - False, 1 - True
 

x=df.drop('Weekly_Sales',axis=1)
y=df['Weekly_Sales']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
regressor=DecisionTreeRegressor()
model=regressor.fit(x_train,y_train)
y_pred=model.predict(x_test)

def type_func(type):
    if type == "A":
        return float(0)
    elif type == "B":
        return float(1)
    else:
        return float(2)
    
def hol_func(holiday):
    if holiday == False:
        return float(0)
    else:
        return float(1)

if selected == "Home":
    col1,col2, = st.columns(2)
    with col1:
        st.markdown("<h2 style='color:DodgerBlue;'>Store Sales Prediction</h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:DeepSkyBlue;'>Technologies:</h3>", unsafe_allow_html=True)
        st.markdown("#### Python, Pandas, Plotly, Streamlit, Python scripting,Data Preprocessing, Visualization, EDA,Machine Learning ")
        st.markdown("<h3 style='color:DeepSkyBlue;'>Overview:</h3>", unsafe_allow_html=True)
        st.markdown("#### To Create a Machine Learning Model and Predict the Sales in each dept and stores and then implemented the insights in business to improve the sales and customer interaction.")
    with col2:
        st.markdown("<h3 style='color:DeepSkyBlue;'>Domain:</h3>", unsafe_allow_html=True)
        st.markdown("### Sales Predict, Store Management and Sales information")
        col3,col4, = st.columns(2)
        with col3:
            sales1=Image.open(r'C:\Users\God\Desktop\phonepe pulse Project folder\ML_final_sales_predict_project\sales1.jfif')
            st.image(sales1)
        with col4:
            sales2=Image.open(r'C:\Users\God\Desktop\phonepe pulse Project folder\ML_final_sales_predict_project\sales2.jfif')
            st.image(sales2)

if selected ==  "Analysis & Prediction":
    select = option_menu(None,["MAE & MSE & Accuracy","Predicted_sales","Data Visualization","Insights"],
                    icons=["files","search","bar-chart","rocket"],
                    default_index=0,
                    orientation="horizontal",
                    styles={"nav-link": {"font-size": "20px", "text-align": "centre", "margin": "-4px", "--hover-color": "#8BD8FF", "transition": "color 0.3s ease, background-color 0.3s ease"},
                            "icon": {"font-size": "20px"},
                            "container" : {"max-width": "6000px", "padding": "6px", "border-radius": "5px"},
                            "nav-link-selected": {"background-color": "#328EBD", "color": "white"}})

    if select == "MAE & MSE & Accuracy":
        with st.spinner('Please wait...'):
            regressor=DecisionTreeRegressor()
            model=regressor.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)
            acc_rf= round(model.score(x_train,y_train) * 100, 2)
            # Calculate evaluation metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)        
            # Print the custom report
            st.markdown("<h3 style='color:DodgerBlue;'>Custom Regression Evaluation Report:</h3>", unsafe_allow_html=True)
            st.write("Mean Absolute Error (MAE):", mae)
            st.write("Mean Squared Error (MSE):", mse)
            st.write('R2_train:',r2_train)
            st.write('R2_test:',r2_test)
            st.write("Training Accuracy:")
            st.info("Accuracy: %i %% \n"%acc_rf)

    if select == "Predicted_sales":
        with st.spinner('Please wait...'):
            st.markdown("<h3 style='color:DodgerBlue;'>Sales Prediction:</h3>", unsafe_allow_html=True)
            with st.form("my_form"):
                col1,col2,col3=st.columns(3)
                with col1:
                    Store= st.selectbox("Select the Store",(df['Store'].unique()))
                    type=st.selectbox("Select the Type",(df1['Type'].unique()))
                    Type=type_func(type)
                    Size = st.number_input("Enter Size")
                    Dept=st.selectbox("Select the Dept",(df['Dept'].unique()))
                    holiday=st.selectbox("Select the IsHoliday",df2['IsHoliday'].unique())
                    Holiday=hol_func(holiday)
                with col2:
                    Temperature = st.number_input("Enter Temperature")
                    Fuel_Price = st.number_input("Enter Fuel Price")
                    MarkDown1 = st.number_input("Enter MarkDown1")
                    MarkDown2 = st.number_input("Enter MarkDown2")
                    MarkDown3 = st.number_input("Enter MarkDown3")
                with col3:
                    MarkDown4 = st.number_input("Enter MarkDown4")
                    MarkDown5 = st.number_input("Enter MarkDown5")
                    Cpi=st.number_input("Enter CPI")
                    Unemployment=st.number_input("Enter Unemployment")

                    new_data1=np.array([[Store,Type,Size,Dept,Holiday,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,Cpi,Unemployment]])

                # Handle the form submission
                if st.form_submit_button(label="PREDICT WEEKLY SALES"):
                    with st.spinner('Please wait...'):
                        predicted_values = model.predict(new_data1)
                        if predicted_values > 10000:
                            st.write("Predicted values:",predicted_values[0])
                            st.success("Good Sales")
                            sales_growth=Image.open(r'C:\Users\God\Desktop\phonepe pulse Project folder\ML_final_sales_predict_project\sales_growth.jfif')
                            st.image(sales_growth)
                        else:
                            st.write("Predicted values:",predicted_values[0])
                            st.error("Low Sales")
                            sales_loss=Image.open(r'C:\Users\God\Desktop\phonepe pulse Project folder\ML_final_sales_predict_project\sales_loss.jfif')
                            st.image(sales_loss)

    if select == "Data Visualization":
        with st.spinner('Please wait...'):
            st.markdown("<h3 style='color:DodgerBlue;'>Data Visualization:</h3>", unsafe_allow_html=True)
            tab1,tab2=st.tabs(["Features Importance & diff b/w weekly sales and markdowns","Display difference between all column"])
            with tab1:
                col1,col2=st.columns([5,5])
                with col1:
                    st.markdown("<h4 style='color:DodgerBlue;'>Features and Importance:</h4>", unsafe_allow_html=True)
                    regressor=DecisionTreeRegressor()
                    model=regressor.fit(x_train,y_train)

                    importance_df = pd.DataFrame({'feature': x.columns,'importance': regressor.feature_importances_ }).sort_values('importance', ascending=False)
                    fig=plt.figure(figsize=(10,6))
                    plt.title('Feature Importance')
                    sns.barplot(data=importance_df, x='importance', y='feature',color='#8BD8FF')
                    st.pyplot(fig)
                with col2:
                    st.markdown("<h4 style='color:DodgerBlue;'>Display difference between MarkDowns and Weekly_sales:</h4>", unsafe_allow_html=True)
                    markdown = st.selectbox('**Select MarkDown:**', ('1','2','3','4','5'),key='MarkDown')
                    fig=plt.figure(figsize=(10,6))
                    plt.title(f"Display difference between MarkDown{markdown} and Weekly_sales")
                    sns.barplot(data=df,x='IsHoliday',y=f'MarkDown{markdown}',color='#8BD8FF')
                    st.pyplot(fig)
            with tab2:
                col1,col2=st.columns([5,5])
                with col1:
                    st.markdown("<h4 style='color:DodgerBlue;'>Display difference between specific column in 3D:</h4>", unsafe_allow_html=True)
                    from mpl_toolkits.mplot3d import Axes3D
                    fig = plt.figure()
                    fig = plt.figure(figsize = (12, 8), dpi=80)
                    ax = fig.add_subplot(111, projection='3d')
                    pnt3d = ax.scatter3D(df5['Store'],df5['Type'], df5['Size'],c=df5['Weekly_Sales'])
                    cbar=plt.colorbar(pnt3d)
                    cbar.set_label("Weekly sales")
                    fig.set_facecolor('white')
                    ax.set_facecolor('white')
                    ax.set_xlabel('Store')
                    ax.set_ylabel('Type')
                    ax.set_zlabel('Size')
                    st.pyplot(fig)
                with col2:
                    st.markdown("<h4 style='color:DodgerBlue;'>Display difference between weekly sales and other columns:</h4>", unsafe_allow_html=True)
                    column=st.selectbox('**Select Brand**', ("Store","Type","Size","Dept","IsHoliday","Temperature","Fuel_Price","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","Cpi","Unemployment"))    
                    fig=plt.figure()
                    plt.scatter(df[f'{column}'] , df['Weekly_Sales'],c=df['Weekly_Sales'])
                    plt.title(f'Display difference between Weekly sales and {column}')
                    plt.ylabel('Weekly_Sales',fontsize=8)
                    plt.xlabel(f'{column}',fontsize=8)
                    st.pyplot(fig)

    if select  == "Insights":
        with st.spinner('Please wait...'):
            tab1, tab2,= st.tabs(["actions based on the insights drawn, with largest business impact."," Describing weekly_sales columns"])
            with tab1:
                with st.spinner('Please wait...'):
                    st.markdown("<h4 style='color:DodgerBlue;'>Insights for Prediction:</h4>", unsafe_allow_html=True)
                    st.write("##### Weekly sales are largely dependent on department,size,store and  CPI.")
                    st.write("##### First we select the specified department,size and store , then enter cpi , temp, markdown3, type, fuel price, unemployment, holiday and other markdowns.")
                    st.write("##### These inputs have little effect on weekly_sales, but the biggest effect on weekly sales is department, size, store and cpi.")
                    st.write("##### Markdowns have little impact on weekly sales. But there are no major vulnerabilities in markdowns.")
                    st.write('##### This process recommended actions based on the insights drawn, with prioritization placed on largest business impact.')
                    st.markdown("<h4 style='color:DodgerBlue;'>Insights for Data visualization:</h4>", unsafe_allow_html=True)
                    st.write("##### In the data visualization, a feature importance plot shows the largest effect on weekly sales using these columns Department, Size, Store and CPI.")
                    st.write("##### In this data visualization, to model the effects of markdowns on holiday weeks. Markdown1 and markdown5 had no high holidays for holiday weeks, whereas Markdown 2,3,4 had more holidays.")
                    st.markdown("<h4 style='color:DodgerBlue;'>Business Idea:</h4>", unsafe_allow_html=True)
                    st.write("##### Choose specific store and department to improve size,then check cpi and unemployment it makes small difference in business.")
                    st.write("##### Temperature is differ between store and each department,so improve the cooling surrounding store and dept.")
                    st.write("##### Each Store and department, then Size and CPI,Type these informations are makes more difference in business.")
            with tab2:
                with st.spinner('Please wait...'):
                    col1,col2=st.columns(2)
                    # check negative weekly_sales count
                    col1.markdown("<h4 style='color:DeepSkyBlue;'>Negative weekly_sales:</h4>", unsafe_allow_html=True)
                    qry=df[df['Weekly_Sales']<=0]
                    col1.write(qry)
                    col1.markdown("<h4 style='color:DeepSkyBlue;'>Negative weekly_sales Count:</h4>", unsafe_allow_html=True)
                    col1.write(len(qry))
                    # check positive weekly_sales 
                    col2.markdown("<h4 style='color:DeepSkyBlue;'>Positive Weekly_Sales:</h4>", unsafe_allow_html=True)
                    qry=df[df['Weekly_Sales']>0]
                    col2.write(qry)
                    col2.markdown("<h4 style='color:DeepSkyBlue;'>Positive Weekly_Sales Count:</h4>", unsafe_allow_html=True)
                    col2.write(len(qry))
                    col2.markdown("<h4 style='color:DeepSkyBlue;'>Describing the dataset:</h4>", unsafe_allow_html=True)
                    des=df.describe().T
                    col2.write(des)
if selected == "About":
    st.markdown("<h2 style='color:DodgerBlue;'>About this project:</h2>",unsafe_allow_html=True)
    st.markdown("<h3 style='color:DeepSkyBlue;'>Introduction:</h3>",unsafe_allow_html=True)
    st.write('##### Store Sales Prediction: Extracting useful data in CSV file to predict the weekly sales. Store Sales Prediction  web application which anlaysis and predict the  data using Machine Learning Algorithms to Predict following year or upcoming days sales information previously.predict the  Weekly sales  information using ,depending values store,dept,markdowns,temperature,fuel,cpi,unemployment and etc..To get  weekly sales value. then displayed a  Streamlit  for future reference and prediction.')
    st.write("")
    st.markdown("<h3 style='color:DeepSkyBlue;'>Problem statement:</h3>",unsafe_allow_html=True)
    st.write("##### This project aims to predict  store sales information using Machine Learnng algorithms, perform data cleaning and preparation, develop Machine Learning models, and create streamlit application to gain insights into sales prediction for following years.")
    st.write('##### Predict the department-wide sales for each store for the following year.')
    st.write('##### Model the effects of markdowns on holiday weeks.')
    st.write('##### Provide recommended actions based on the insights drawn, with prioritization placed on largest business impact.')
    st.write('##### To Clean and Preprocessing  the store,sales,sales feature dataset using Pandas & numpy and ensure efficient data retrieval for prediction.To clean and prepare the addressing missing values, duplicates, and data type conversions for accurate prediction.')
    st.write('##### Choosing dependent and independent columns to predict the sales.then choosing the suitable Machine Learning algorithm for prediction. ')
    st.write('##### Develop a streamlit web application with interactive prediction page showcasing the predicting value, allowing users to select store,department,fuel,markdowns and other relevant factors.')
    st.write("")
    st.markdown("<h3 style='color:DeepSkyBlue;'>Problem solution:</h3>",unsafe_allow_html=True)
    st.markdown("<h4 style='color:DeepSkyBlue;'>what i did for project solution:</h4>",unsafe_allow_html=True)
    st.markdown("<h4 style='color:LightSkyBlue;'>Data Availability:</h4>",unsafe_allow_html=True)
    st.markdown('#### stores_data_set.csv: ')
    st.markdown('##### This file contains anonymized information about the 45 stores, indicating the type and size of store.')
    st.markdown("#### sales_data_set.csv: ")
    st.markdown("##### This is the historical training data, which covers to 2010-02-05 to 2012-11- 01, Within this file you will find the following fields : Store,Dept,Date,Weekly_Sales,IsHoliday.")
    st.markdown('#### features_data_set.csv: ')
    st.markdown('##### This file contains additional data related to the store, department, and regional activity for the given dates. It contains the following fields : Store,Date,Temperature,Fuel_Price,MarkDown1-5,CPI,Unemployment.') 
    st.markdown("<h4 style='color:LightSkyBlue;'>Machine Learning Model:</h4>",unsafe_allow_html=True)
    st.write('#####  Given CSV files to convert useful CSV files to correct data format. ')
    st.write('#####  To Merge the  three csv files to useful csv  combine datasets')
    st.write('#####  Then to Cleaning and Preprocessing the datasets ex:-checking datatypes, checkin NULL or not.')
    st.write('#####  And drop unneccessary columns. then change the  Alphapetic value(object) to Numeric value using **OrdinalEncoder** for IsHoliday and Type columns.')
    st.write('#####  To choose Dependent(Y) and Independent(X) columns and importing train_test_split module to split Training data and Testing data randomly.')
    st.write('#####  To using SKlearn(Scikit-learn) to import DecisionTreeRegressor Machine Learning algorithm and  to using fit() func to insert Training data to implemented the algorithm.')
    st.write('#####  To check y_pred to using predict() func to predict Testing data.')
    st.write('#####  To check regressor custom model evaluation report: to using "MAE & MSE & Accuracy" method.')
    st.write('#####  Then Predict the sales using models.')
    st.write("")
    st.markdown("<h4 style='color:LightSkyBlue;'>Streamlit Dashboard about:</h4>",unsafe_allow_html=True)
    st.write('#####  Create a Streamlit web app and using technologies Pandas, PIL,plotly.express (px) libraries in imported  setup and running Features .')
    st.write('##### **"Home"** : Displays an overview of the app including technologies used and a brief description of the app.and Store sales predictions. ')
    st.write('##### **"Analysis & Prediction"** : This section allows the user to see evaluation report and prediction and visualization of data and Isights of business growth.')
    st.write('##### - "**MAE & MSE & Accuracy"** : This section its also like as checking accuracy in regressor Mean Absolute Error, Mean Square Error, R2-R Square.')
    st.write('##### - **"Prediction"** : This section to select the independent (or) x columns data then click the predict button to predict the sales.')
    st.write('##### - **"Data Visualization"** : This section   to visualize information Model the effects of markdowns on holiday weeks   also using  streamlit plots.')                  
    st.write('###### - **"Effect b/w weekly sales and markdowns"** : This section   to visualize information Model of features informations and Model the effects of markdowns on holiday weeks  also using  streamlit plots.')
    st.write('###### - **"Display difference between all column"** : This section   to visualize information Display difference between specific column in 3D and Display difference between weekly sales and other columns  also using  streamlit plots.')
    st.write('##### -**"Insights"** : This section to display the  business growth insights in datasets and during predictions.')
    st.write('##### **About** : This section to dsiplays about for this Project.')
    st.write("")
    st.markdown("<h3 style='color:DeepSkyBlue;'>Conclusion:</h3>",unsafe_allow_html=True)
    st.write("##### I Created the project to used to predicting the  weekly sales information to improve business idea and business perspective . ")
    st.write("##### It mostly  helped as client knows about the business and  sales informations to easly improve business with these columns and other works.")
    st.write("##### Weekly sales are largely dependent on department,size,store and  CPI.")
    st.write("##### First we select the specified department,size and store , then enter cpi , temp, markdown3, type, fuel price, unemployment, holiday and other markdowns.")
    st.write("##### These inputs have little effect on weekly_sales, but the biggest effect on weekly sales is department, size, store and cpi.")
    st.write("##### Markdowns have little impact on weekly sales. But there are no major vulnerabilities in markdowns.")
    st.write("##### In the data visualization, a feature importance plot shows the largest effect on weekly sales using these columns Department, Size, Store and CPI.")
    st.write("##### In this data visualization, to model the effects of markdowns on holiday weeks. Markdown1 and markdown5 had no high holidays for holiday weeks, whereas Markdown 2,3,4 had more holidays.")
    st.write("##### Choose specific store and department to improve size,then check cpi and unemployment it makes small difference in business.")
    st.write("##### Temperature is differ between store and each department,so improve the cooling surrounding store and dept.")
    st.write("##### Each Store and department, then Size and CPI,Type these informations are makes more difference in business.")