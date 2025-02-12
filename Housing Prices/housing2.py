



import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

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


# print(one_hot_encoder(training_df, ["MSZoning", "Street"], drop_first=True))




# Klasik data prepration and feature engineering fonksiyonumu bu şekle çevirmemin sebebi elimde tek veri seti yerine train ve test olmak üzere birbirini tamamlayan iki veri setinin olması.


def housing_data_prep(train_df, test_df):
    # Train set sütunlarını büyük harfe çevir
    train_df.columns = [col.upper() for col in train_df.columns]

    # "SALEPRICE" hedef değişkenini ayır
    y_train = train_df["SALEPRICE"]
    X_train = train_df.drop(columns=["SALEPRICE", "NEIGHBORHOOD"], errors="ignore")

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

    # Test setinde "SALEPRICE" olmadığı için y_test = None
    y_test = None  

    # **Train setinde olan sütunlara göre test setini hizala**
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Numerik değişkenleri belirle
    _, num_cols, _ = grab_col_names(X_test, cat_th=5, car_th=20)
    
    # **Test setinde sadece ortak olan sütunları ölçekle**
    num_cols_test = [col for col in num_cols if col in X_test.columns]
    X_test[num_cols_test] = scaler.transform(X_test[num_cols_test])

    return X_train, y_train, X_test, y_test



train_df = pd.read_csv("Housing Prices/datasets/train.csv")
test_df = pd.read_csv("Housing Prices/datasets/test.csv")



X_train, y_train, X_test, y_test = housing_data_prep(train_df, test_df)




"""print([dtype for dtype in X_train.dtypes == "object"])  # sadece " CollgCr " kolonu object olarak kalmış onu bulmak için yazdım bunu"""




def base_model(X_train, y_train, X_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

    model.fit(X_train, y_train) # training the model 

    predictions = model.predict(X_test)

    print(predictions)

base_model(X_train, y_train, X_test)


# Model tahminini daha da geliştirebilirsin!





# Pipeline deki bu süreci eline aldığın her projeye entegre etmeye devam et!!
# Bunu da bitir bir an önce diğer projeye de entegre geç!!!











