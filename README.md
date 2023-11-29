# Store_Sales_Prediction 💰✨💲
## Store sales prediction using Machine Learning  algorithms and predict the sales and found insights to improve business.
# Introduction:
- Store Sales Prediction: Extracting useful data in CSV file to predict the weekly sales.
- Store Sales Prediction  web application which anlaysis and predict the  data using Machine Learning Algorithms to Predict following year or upcoming days sales information previously.
- predict the  Weekly sales  information using ,depending values store,dept,markdowns,temperature,fuel,cpi,unemployment and etc..To get  weekly sales value.
- then displayed a  Streamlit  for future reference and prediction.
# Domain:
- Business Industry, sales growth and sales loss
# Problem statement:
- This project aims to predict  store sales information using Machine Learnng algorithms, perform data cleaning and preparation, develop Machine Learning models, and create streamlit application to gain insights into sales prediction for following years.
- Predict the department-wide sales for each store for the following year.
- Model the effects of markdowns on holiday weeks.
- Provide recommended actions based on the insights drawn, with prioritization placed on largest business impact.
- To Clean and Preprocessing  the store,sales,sales feature dataset using Pandas & numpy and ensure efficient data retrieval for prediction.To clean and prepare the addressing missing values, duplicates, and data type conversions for accurate prediction.
- Choosing dependent and independent columns to predict the sales.then choosing the suitable Machine Learning algorithm for prediction. 
- Develop a streamlit web application with interactive prediction page showcasing the predicting value, allowing users to select store,department,fuel,markdowns and other relevant factors.
## Libraries
### Libraries/Modules needed for the project!
- matplotlib, Seaborn - (To plot and visualize the data)
- Pandas - (To Clean and maipulate the data)
- Streamlit - (To Create Graphical user Interface)
- Streamlit_option_menu (To create option_menu)
- PIL (To insert and open  Image)
- Sklearn.preprocessing import OrdinalEncoder(To import all preprocessing method & OrdinalEncoder)
- Sklearn.model_selection train_test_split (To split the training data and testing data)
- Sklearn.tree import DecisionTreeRegressor (Machine Learning Algorithm)
- Sklearn.metrics import accuracy_score (To mfind Accuracy in the model)
- Sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score (To find regresor accuracy MAE,MSE and R2)
# Problem solution:
## what i did for project solution:
### Data Availability:
- stores_data_set.csv:
- This file contains anonymized information about the 45 stores, indicating the type and size of store.
- sales_data_set.csv:
- This is the historical training data, which covers to 2010-02-05 to 2012-11- 01, Within this file you will find the following fields: Store,Dept,Date,Weekly_Sales,IsHoliday.")
- features_data_set.csv:
- This file contains additional data related to the store, department, and regional activity for the given dates. It contains the following fields : Store,Date,Temperature,Fuel_Price,MarkDown1-5,CPI,Unemployment.
## workflow:
## Machine Learning Model:
#### Step 1 :
- Given CSV files to convert useful CSV files to correct data format.
- To Merge the  three csv files to useful csv  combine datasets.
#### Step 2 :
- Then to Cleaning and Preprocessing the datasets ex:-checking datatypes, checkin NULL or not.
- And drop unneccessary columns. then change the  Alphapetic value(object) to Numeric value using **OrdinalEncoder** for IsHoliday and Type columns.
#### Step 3 :
- To choose Dependent(Y) and Independent(X) columns and importing train_test_split module to split Training data and Testing data randomly.
#### Step 4 :
- To using SKlearn(Scikit-learn) to import DecisionTreeRegressor Machine Learning algorithm and  to using fit() func to insert Training data to implemented the algorithm.
#### Step 5 :
- To check y_pred to using predict() func to predict Testing data.
#### Step 6 :
- To check regressor custom model evaluation report: to using "MAE & MSE & Accuracy" method.
Then Predict the sales using models.
## Streamlit Dashboard workflow:
- Create a Streamlit web app and using technologies Pandas, PIL,plotly.express (px) libraries in imported  setup and running Features .
- **"Home"** : Displays an overview of the app including technologies used and a brief description of the app.and Store sales predictions.
- **"Analysis & Prediction"** : This section allows the user to see evaluation report and prediction and visualization of data and Isights of business growth.
- -"**MAE & MSE & Accuracy"** : This section its also like as checking accuracy in regressor Mean Absolute Error, Mean Square Error, R2-R Square.
- -**"Prediction"** : This section to select the independent (or) x columns data then click the predict button to predict the sales.
- -**"Data Visualization"** : This section   to visualize information Model the effects of markdowns on holiday weeks   also using  streamlit plots.                 
- -**"Effect b/w weekly sales and markdowns"** : This section   to visualize information Model of features informations and Model the effects of markdowns on holiday weeks  also using  streamlit plots.
- -**"Display difference between all column"** : This section   to visualize information Display difference between specific column in 3D and Display difference between weekly sales and other columns  also using  streamlit plots.
- -**"Insights"** : This section to display the  business growth insights in datasets and during predictions.
- **About** : This section to dsiplays about for this Project.
# Conclusion:
- I Created the project to used to predicting the  weekly sales information to improve business idea and business perspective .
- It mostly  helped as client knows about the business and  sales informations to easly improve business with these columns and other works.
- Weekly sales are largely dependent on department,size,store and  CPI.
- First we select the specified department,size and store , then enter cpi , temp, markdown3, type, fuel price, unemployment, holiday and other markdowns.
- These inputs have little effect on weekly_sales, but the biggest effect on weekly sales is department, size, store and cpi.
- Markdowns have little impact on weekly sales. But there are no major vulnerabilities in markdowns.
- In the data visualization, a feature importance plot shows the largest effect on weekly sales using these columns Department, Size, Store and CPI.
- In this data visualization, to model the effects of markdowns on holiday weeks. Markdown1 and markdown5 had no high holidays for holiday weeks, whereas Markdown 2,3,4 had more holidays.
- Choose specific store and department to improve size,then check cpi and unemployment it makes small difference in business.
- Temperature is differ between store and each department,so improve the cooling surrounding store and dept.
- Each Store and department, then Size and CPI,Type these informations are makes more difference in business.
