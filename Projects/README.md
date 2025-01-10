Problem Statement:
Predict the price of house in the United States based on the city, area and number of bedrooms and baths

Task Breakup:
1. Load the data in Pandas DataFrame
2. Perform Data Cleaning. It comprises of below sub-tasks-
     1. Drop the columns which will not be required for data transformation or ML training
     2. Delete the NULL and NA row elements
     3. Delete the rows which have inconclusive values like number of bedrooms/baths is zero, or price is zero
  
3. Perform Data transformation as part of Feature Engineering. It comprises of below tasks-
     1. Remove the rows as part of outliers where the houses are older than 50years and are not renovated within last 25years
     2. Remove the rows where the number of bedrooms are not in sync with the area of the house
     3. Decrease the number of locations by adding "Other" location which will be having count of houses less than 10
     4. Remove outliers where the price of the house is more than 1 standard deviation in order to have normal data distribution
     5. Since location is the only categorical data, perform one-hot encoding on city column for data standardisation
  
4. Train and test the Linear Regression model
5. Implement GridSearchCV to evaluate the best model for prediction
6. Train the best scorer model with the data prepared in above steps
7. Predict the price providing different inputs - city, area, number of bedrooms and bath


