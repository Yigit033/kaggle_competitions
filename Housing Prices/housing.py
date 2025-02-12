
import pandas as pd

""" 
    Not:
    ✅ Tek veri setin varsa → Önce train_test_split() ile ikiye böl. => X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ✅ Zaten ayrılmış verilerin varsa → Sadece bağımsız ve bağımlı değişkenleri seç.
"""


training_df = pd.read_csv("Housing Prices/datasets/train.csv")
testing_df = pd.read_csv("Housing Prices/datasets/train.csv")

"""print(training_df.describe())
"""

from sklearn.ensemble  import RandomForestClassifier 

y_train = training_df["SalePrice"]
X_train = pd.get_dummies(training_df.drop(["SalePrice"], axis=1))


X_test = pd.get_dummies(training_df.drop("SalePrice", axis=1))
y_test= testing_df["SalePrice"]

print(X_test.head())


model= RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1) 

model.fit(X_train,y_train) # training the model


predictions = model.predict(X_test)

print(predictions)





# X_test= testing_df.drop(["SalePrice"], axis=1)

# y_test= testing_df["SalePrice"]


# model= RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1) 

# model.fit(X_train,y_train) # training the model


# predictions = model.predict(X_test)

# print(predictions)


# print(X_train.dtypes)










# def housing_data_prep(dataframe):

#     dataframe.columns = [col.upper() for col in dataframe.columns]

#     cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)
#     cat_cols = [col for col in cat_cols if "SalePrice" not in col]

#     df = one_hot_encoder(dataframe, cat_cols, drop_first=True)
#     df.columns = [col.upper() for col in df.columns]

#     cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

#     cat_cols = [col for col in cat_cols if "SalePrice" not in col]

#     # Scaler ve Encoder'ı burada oluşturup fit edelim:       test fonksiyonu için.
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df[num_cols])  

#     df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

#     y = df["SALEPRICE"]
#     X = df.drop(["SALEPRICE"], axis=1)

#     # Encoder'ı fit et                   test fonksiyonu için.
#     encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
#     encoder.fit(X[cat_cols])  







