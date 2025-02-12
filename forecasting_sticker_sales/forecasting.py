import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_percentage_error


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



def check_df(dataframe, head=5):
    assert isinstance(dataframe, pd.DataFrame), "Input should be a pandas DataFrame"

    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Describe #####################")
    print(dataframe.describe().T)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")

    # Select only numeric columns for quantile calculation
    numeric_df = dataframe.select_dtypes(include=[np.number])
    
    # Compute quantiles for numeric columns only
    print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    # print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
def cat_summary(dataframe, col_name, plot=False): #    Burada yaptığımız iş : kategorik değişkenlerin sınıflarına göre gözlem sayılarını ve oranlarını görmek
    assert isinstance(dataframe, pd.DataFrame), "Input should be a pandas DataFrame"
    assert col_name in dataframe.columns, f"{col_name} not found in DataFrame columns"

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
def num_summary(dataframe, numerical_col, plot=False):
    assert isinstance(dataframe, pd.DataFrame), "Input should be a pandas DataFrame"
    assert numerical_col in dataframe.columns, f"{numerical_col} not found in DataFrame columns"

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
def target_summary_with_num(dataframe, target, numerical_col):
    assert isinstance(dataframe, pd.DataFrame), "Input should be a pandas DataFrame"
    assert target in dataframe.columns, f"{target} not found in DataFrame columns"
    assert numerical_col in dataframe.columns, f"{numerical_col} not found in DataFrame columns"

    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
def target_summary_with_cat(dataframe, target, categorical_col):
    assert isinstance(dataframe, pd.DataFrame), "Input should be a pandas DataFrame"
    assert target in dataframe.columns, f"{target} not found in DataFrame columns"
    assert categorical_col in dataframe.columns, f"{categorical_col} not found in DataFrame columns"

    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
def correlation_matrix(df, cols):
    assert isinstance(df, pd.DataFrame), "Input should be a pandas DataFrame"
    assert all(col in df.columns for col in cols), "Some columns not found in DataFrame"

    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    assert isinstance(dataframe, pd.DataFrame), "Input should be a pandas DataFrame"
    
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car



def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def replace_with_thresholds(dataframe, variable): # Aykırı değerleri sınırlandırma işlemi yapar. Aykırı değerleri threşholdlar ile değiştirir.
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False    
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe





def housing_data_prep(train_df, test_df):

    train_df["num_sold"] = train_df["num_sold"].fillna(train_df["num_sold"].median())
        
    # Train set sütunlarını büyük harfe çevir
    train_df.columns = [col.upper() for col in train_df.columns]


    # "num_sold" hedef değişkenini ayır
    y_train = train_df["NUM_SOLD"]
    X_train = train_df.drop(columns=["NUM_SOLD", "DATE"], errors="ignore")

    # Kategorik ve numerik değişkenleri belirle
    cat_cols, num_cols, cat_but_car = grab_col_names(X_train, cat_th=5, car_th=20)

    # One-hot encoding işlemi
    X_train = one_hot_encoder(X_train, cat_cols, drop_first=True)
    X_train.columns = [col.upper() for col in X_train.columns]

    

    # Güncellenmiş numerik değişkenleri tekrar al
    _, num_cols, _ = grab_col_names(X_train, cat_th=5, car_th=20)

    # Standart ölçekleme
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

    # ------------------- TEST SETİ İŞLEMLERİ -------------------
    

    # Test set sütunlarını büyük harfe çevir
    test_df.columns = [col.upper() for col in test_df.columns]

    # Kategorik ve numerik değişkenleri belirle
    cat_cols, num_cols, cat_but_car = grab_col_names(test_df, cat_th=5, car_th=20)

    # One-hot encoding işlemi
    X_test = one_hot_encoder(test_df, cat_cols, drop_first=True)
    X_test.columns = [col.upper() for col in X_test.columns]

    # Test setinde "num_sold" olmadığı için y_test = None
    y_test = None  

    # **Train setinde olan sütunlara göre test setini hizala**
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Numerik değişkenleri belirle
    _, num_cols, _ = grab_col_names(X_test, cat_th=5, car_th=20)
    
    # **Test setinde sadece ortak olan sütunları ölçekle**
    num_cols_test = [col for col in num_cols if col in X_test.columns]
    X_test[num_cols_test] = scaler.transform(X_test[num_cols_test])

    return X_train, y_train, X_test, y_test





train_df = pd.read_csv("forecasting_sticker_sales/datasets/train.csv")
test_df = pd.read_csv("forecasting_sticker_sales/datasets/test.csv")



X_train, y_train, X_test, y_test = housing_data_prep(train_df, test_df)

"print(X_train.head())"

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=46) # Train setini train ve validation setine ayırma işlemi




models_and_parameters = {
    'LinearRegression': (LinearRegression(), {}),
    'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
    'Lasso': (Lasso(), {'alpha': [0.01, 0.1, 1.0]}),
    'DecisionTree': (DecisionTreeRegressor(random_state=42), {
        'max_depth': [None, 10, 20, 50],
        'min_samples_split': [2, 5, 10]
    }),
    'RandomForest': (RandomForestRegressor(random_state=42), {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }),
    'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [50, 100],
        'max_depth': [3, 5]
    }),
    'KNeighbors': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 10]})
}


# Train and evaluate each model
best_model = None
best_score = -np.inf
best_params = None

for model_name, (model, param_grid) in models_and_parameters.items():
    print(f"Training {model_name}...")
    if param_grid:
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        params = grid_search.best_params_
    else:
        # Train the model without hyperparameter tuning
        model.fit(X_train, y_train)
        params = {}

    # Evaluate the model on validation set
    y_val_pred = model.predict(X_val)
    score = r2_score(y_val, y_val_pred)
    print(f"{model_name} R^2 score: {score:.4f} | Params: {params}")

    # Update the best model if it outperforms others
    if score > best_score:
        best_model = model
        best_score = score
        best_params = params

print(f"\nBest Model: {best_model.__class__.__name__}")
print(f"Best R^2 Score: {best_score:.4f}")
print(f"Best Parameters: {best_params}")



# Predict on the test set using the best model
y_test_pred = best_model.predict(X_test)

project_name = "forecasting_predictions"


# Save predictions to a CSV file
test_df['num_sold'] = y_test_pred
test_df['id'] = range(len(test_df))
test_df[['id', 'num_sold']].to_csv(f"forecasting_sticker_sales/predictions/{project_name}.csv", index=False)
print(f"Predictions saved to '{project_name}.csv'.")















