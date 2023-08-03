import streamlit as st
import pandas as pd
import pickle 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
#from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn import set_config
set_config(display="diagram")
from sklearn.naive_bayes import ComplementNB
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report
#from sklearn.metrics import roc_curve, auc,roc_curve,roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
#import xgboost as xgb
from sklearn.impute import SimpleImputer
import transformers


#### 
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pickle
import pandas as pd
import streamlit as st
import joblib
import time
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_norm = X.apply(self.process_doc)
        return X_norm

    def process_doc(self, doc):
        def pos_tagger(nltk_tag):
            if nltk_tag.startswith('J'):
                return wordnet.ADJ
            elif nltk_tag.startswith('V'):
                return wordnet.VERB
            elif nltk_tag.startswith('N'):
                return wordnet.NOUN
            elif nltk_tag.startswith('R'):
                return wordnet.ADV
            else:
                return None
            
        # remove stop words and punctuations, then lower case
        doc_norm = [tok.lower() for tok in word_tokenize(doc) if tok.isalpha() and tok.lower() not in self.stop_words]
        
        #  POS detection on the result will be important in telling Wordnet's lemmatizer how to lemmatize
        
        # creates list of tuples with tokens and POS tags in wordnet format
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tag(doc_norm)))
        doc_norm = [self.wnl.lemmatize(token, pos) for token, pos in wordnet_tagged if pos is not None]
        return " ".join(doc_norm)

st.image("https://images.hdqwalls.com/wallpapers/lamborghini-centenario-car-rear-8g.jpg")
#st.image(image, caption='Welcome!')


st.title("Predicting Car Prices")

st.markdown("Enter the following information for a car and get an estimated car value.")


df = pd.read_csv('/Users/mahmoud/Car_Price_Prediction/V3/working_dataset.csv')


# nlp_preprocessor = TextPreprocessor()
# df['Processed_Reviews'] = nlp_preprocessor.transform(df['Car Reviews'])


# df['Review_Length'] = df['Processed_Reviews'].apply(len)  
# df['Num_Exclamation_Marks'] = df['Processed_Reviews'].apply(lambda text: text.count('!')) 





df.drop(columns=['Car Reviews'], inplace=True)


X = df.drop(columns=['Price','Unnamed: 0', 'VIN', 'Engine'])  
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



b = X_train.iloc[:1,:]
z = b.copy()

z['Mileage'] = st.slider('Mileage', min_value=0, max_value=200000, step=1000)
 
z['Dealer Rating'] = st.slider('Dealer Rating', min_value=0.0, max_value=5.0, step=0.1)

z['Review Count'] = st.slider('Review Count', min_value=0, max_value=1000, step=10)
z['Badge Label'] = st.slider('Badge Label', min_value=1, max_value=5, step=1, format='%d')
z['Car Rating'] = st.slider('Car Rating', min_value=0.0, max_value=5.0, step=0.1)
z['Accidents or damage'] = st.slider('Accidents or damage', min_value=0, max_value=1, step=1)
z['1-owner vehicle'] = st.slider('1-owner vehicle', min_value=0, max_value=1, step=1, format='%d')
z['Personal use only'] = st.slider('Personal use only', min_value=0, max_value=1, step=1, format='%d')
z['Open recall'] = st.slider('Open recall', min_value=0, max_value=1, step=1, format='%d')


st.image("https://images.hdqwalls.com/wallpapers/lamborghini-centenario-grey-5k-gi.jpg")


# YEAR
year_options = [
    2016, 2015, 2012, 2010, 2014, 2013, 2009, 2008, 2018, 2021, 2017,
    1989, 2011, 2002, 2007, 2006, 2005, 1988, 2004, 2003, 2020, 2001,
    2019, 1999, 1995, 1997, 1990, 1991, 2000, 1992, 1987, 1998, 1994,
    1996, 2022, 1985, 1967, 1958, 1955, 2023, 1950, 1984
]
year = st.selectbox('Year', year_options)
z['Year'] = year




# Make
make_options = df.loc[(df['Year'] == year), 'Make'].unique()
    #(df['Exterior color'] == Exterior_color) & (df['Interior color'] == Interior_color),'Make'].unique()
Make = st.selectbox('Make', make_options)
z['Make'] = Make


# Model
model_options = df.loc[(df['Year'] == year) & (df['Make'] == Make), 'Model'].unique()
                       #& (df['MPG'] == mpg) &(df['Exterior color'] == Exterior_color) & 
               #        (df['Interior color'] == Interior_color) & (df['Make'] == Make), 'Model'].unique()
Model = st.selectbox('Model', model_options)
z['Model'] = Model
                       
# Type 
type_options = df.loc[(df['Year'] == year) & (df['Make'] == Make) & (df['Model'] == Model),'Type'].unique()
           #           & (df['MPG'] == mpg) & (df['Exterior color'] == Exterior_color) &
   # (df['Interior color'] == Interior_color) & (df['Make'] == Make) & (df['Model'] == Model),'Type'].unique()
Type = st.selectbox('Type', type_options)
z['Type'] = Type




#MPG 
mpg_options = df.loc[(df['Year'] == year) & (df['Make'] == Make) & (df['Model'] == Model) 
                     & (df['Type'] == Type), 'MPG'].unique()
mpg = st.selectbox('MPG', mpg_options)
z['MPG'] = mpg


# Exterior Color
exterior_color_options = df.loc[(df['Year'] == year) & (df['Make'] == Make) & (df['Model'] == Model) 
                     & (df['Type'] == Type) & (df['MPG'] == mpg), 'Exterior color'].unique()
Exterior_color = st.selectbox('Exterior color', exterior_color_options)
z['Exterior color'] = Exterior_color

# Interior Color
interior_color_options = df.loc[(df['Year'] == year) & (df['Make'] == Make) & (df['Model'] == Model) 
                     & (df['Type'] == Type) & (df['MPG'] == mpg) & 
                  (df['Exterior color'] == Exterior_color),'Interior color'].unique()
Interior_color = st.selectbox('Interior color', interior_color_options)
z['Interior color'] = Interior_color

# So Interior_color is a variable and this is what we give for the filtering
# Interior color however is column name

# Do the above for the rest


# # Make
# make_options = df.loc[(df['Year'] == year) & (df['MPG'] == mpg) &
#     (df['Exterior color'] == Exterior_color) & (df['Interior color'] == Interior_color),'Make'].unique()
# Make = st.selectbox('Make', make_options)
# z['Make'] = Make


# # Model
# model_options = df.loc[(df['Year'] == year) & (df['MPG'] == mpg) &(df['Exterior color'] == Exterior_color) & 
#                        (df['Interior color'] == Interior_color) & (df['Make'] == Make), 'Model'].unique()
# Model = st.selectbox('Model', model_options)
# z['Model'] = Model
                        
# # Type 
# type_options = df.loc[(df['Year'] == year) & (df['MPG'] == mpg) & (df['Exterior color'] == Exterior_color) &
#     (df['Interior color'] == Interior_color) & (df['Make'] == Make) & (df['Model'] == Model),'Type'].unique()
# Type = st.selectbox('Type', type_options)
# z['Type'] = Type
                        
# Drivetrain
drivetrain_options = df.loc[(df['Year'] == year) &(df['MPG'] == mpg) 
                            #&(df['Exterior color'] == Exterior_color) &
    #(df['Interior color'] == Interior_color) 
                            &(df['Make'] == Make) &(df['Model'] == Model) &(df['Type'] == Type),
                            'Drivetrain'].unique()
Drivetrain = st.selectbox('Drivetrain', drivetrain_options)
z['Drivetrain'] = Drivetrain
                        
# Fuel type
fuel_type_options = df.loc[(df['Year'] == year) &(df['MPG'] == mpg) 
                           #&(df['Exterior color'] == Exterior_color) &
    #(df['Interior color'] == Interior_color) 
                           &(df['Make'] == Make) &(df['Model'] == Model) &(df['Type'] == Type),
     # (df['Drivetrain'] == Drivetrain), 
                           'Fuel type'].unique()
Fuel_type = st.selectbox('Fuel type', fuel_type_options)
z['Fuel type'] = Fuel_type
                        
# Transmission
transmission_options = df.loc[(df['Year'] == year) &(df['MPG'] == mpg) 
                           #   &(df['Exterior color'] == Exterior_color) &
    #(df['Interior color'] == Interior_color) 
                              &(df['Make'] == Make) &(df['Model'] == Model) &(df['Type'] == Type)
      #(df['Drivetrain'] == Drivetrain) &(df['Fuel type'] == Fuel_type)
                              , 'Transmission'].unique()
z['Transmission'] = st.selectbox('Transmission', transmission_options)

# Dealer Name
dealer_name_options = df.loc[(df['Year'] == year) &(df['MPG'] == mpg) 
                   #           &(df['Exterior color'] == Exterior_color) &
   # (df['Interior color'] == Interior_color) 
                              &(df['Make'] == Make) &(df['Model'] == Model) &(df['Type'] == Type)
      #(df['Drivetrain'] == Drivetrain) &(df['Fuel type'] == Fuel_type)
                              , 'Dealer Name'].unique()
z['Dealer Name'] = st.selectbox('Dealer Name', dealer_name_options)



#z['Car Name'] = z['Year'].astype(str) + ' ' + z['Make'] + ' ' + z['Model'] + ' ' + z['Type']

X_train = X_train.drop(columns = ['Car Name']) 
z = z.drop(columns = ['Car Name'])

# if st.button('build prediction array'):
#     st.dataframe(z)
#     st.dataframe(X_train)
   

    


if st.button('submit'):
    
    nominal_transformer = Pipeline(steps=[  
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numeric_transformer = Pipeline(steps=[  
        ('scaler', StandardScaler())
    ])
    
    # Label encode the specified features
    label_encoder = LabelEncoder()
    label_features = ['Badge Label', 'Accidents or damage', '1-owner vehicle', 'Personal use only', 'Open recall']
    for feature in label_features:
        z[feature] = label_encoder.fit_transform(z[feature])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', nominal_transformer , ['Dealer Name', 'Exterior color', 'Interior color', 'Drivetrain', 'Fuel type', 'Transmission', 'Make', 'Model', 'Type']),
            ('num', numeric_transformer, ['Mileage', 'Dealer Rating', 'Review Count', 'Car Rating', 'MPG', 'Year'])
        ])
    
    gb_regressor = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)

    best_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', gb_regressor)
    ])

    best_model.fit(X_train, y_train)
  
    pred = best_model.predict(z)

    st.write('This is how much this car is worth:', pred) 

   
    
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print("Root Mean Squared Error:", rmse)
# print("R-squared:", r2)
#    print(y_pred)
  
