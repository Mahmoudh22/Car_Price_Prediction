# Car Price Prediction Project

![car_rear](https://github.com/Mahmoudh22/Car_Price_Prediction/assets/131708611/87938fcf-f52b-4466-a361-ad3554240ba7)


# Introduction
Welcome to the Car Price Prediction project! As a data scientist student, I developed a comprehensive project to predict car prices using a dataset scraped from cars.com. The project encompasses data cleaning, NLP (Natural Language Processing) analysis, and the creation of an interactive web app to predict car prices based on user input.

# Dataset
The dataset used for this project comprises over 10,000 car listings from cars.com, containing various features related to each car, such as mileage, dealer information, price, ratings, accidents or damage history, and more.

# Features included in the dataset:
Car Name
Mileage
Dealer Name
Dealer Rating
Review Count
Price
Badge Label
Car Rating
Car Reviews
Accidents or damage history
1-owner vehicle status
Personal use only status
Open recall status
Exterior color
Interior color
Drivetrain
MPG (Miles Per Gallon)
Fuel type
Transmission
Engine
VIN (Vehicle Identification Number)
Year
Make
Model
Type

# Data Preprocessing and NLP
I performed data cleaning to handle missing values, outliers, and inconsistent data.
Utilizing NLP techniques, I analyzed the car reviews to gain valuable insights into the sentiment and customer perception associated with each car.


# Machine Learning Model
For car price prediction, I developed a Gradient Boost Regressor model that achieved an R-squared value of 90% and an RMSE (Root Mean Squared Error) of approximately $1,000. The model showcases a strong predictive capability in estimating car prices based on various features.


# NLP with NMF Model
I applied Non-Negative Matrix Factorization (NMF) to process the car reviews. This allowed me to uncover underlying topics and sentiments expressed by customers in the reviews.

<img width="987" alt="Screenshot 2023-08-03 at 11 48 44 AM" src="https://github.com/Mahmoudh22/Car_Price_Prediction/assets/131708611/544986e6-bef0-4658-bc8d-95cc69b43cfe">
we can see words such as:
- buy, good, love, reliable fall under driving experience; Of course we want reliable lovely cars!
- pay, fix, pointless under dealership experiences; Well, who wants to deal with a pointless dealer?
- fluid, engine, leak, transmission, shutting under car performance and issues; Want to go to the mechanic everyday? 
- struggle, awesome, design, all under the experience section; lets buy a struggling car! (Just joking, DON"T!) 
- coil, service, engine, repair all under maintenance + repair which is also essential.
So to bring it together, this NMF unsupervised model is to highlight key terms and topics for us when purchasing

# Interactive Web App
To enhance user experience, I created an interactive web app that enables users to input car details and receive a predicted price based on the trained model.

# Conclusion
The Car Price Prediction project provides valuable insights into car prices based on a dataset scraped from cars.com. Utilizing machine learning and NLP techniques, I developed a predictive model and conducted sentiment analysis on car reviews.

The interactive web app enables users to get estimated car prices, making it a valuable tool for both car buyers and sellers.

For further details or inquiries, feel free to reach out. Happy exploring!

Created by Mahmoud Hosny
