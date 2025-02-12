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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


import numpy as np





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



 

def llm_classification(train_df, test_df): #   Burada yaptığımız iş : LLM Classification modeli oluşturup, tahmin yapmak ve sonuçları csv dosyasına yazdırmak.
    # Combine text columns into a single feature
    train_df["combined_text"] = train_df["prompt"] + " " + train_df["response_a"] + " " + train_df["response_b"]
    test_df["combined_text"] = test_df["prompt"] + " " + test_df["response_a"] + " " + test_df["response_b"]

    train_df = train_df.drop(columns=["prompt", "response_a", "response_b"], errors="ignore")
    test_df = test_df.drop(columns=["prompt", "response_a", "response_b"], errors="ignore")

    # Create target column
    train_df["target_col"] = train_df[["winner_model_a", "winner_model_b", "winner_tie"]].idxmax(axis=1)

    train_df = train_df.drop(columns=["winner_model_a", "winner_model_b", "winner_tie"], errors="ignore")
    test_df = test_df.drop(columns=["winner_model_a", "winner_model_b", "winner_tie"], errors="ignore")

    label_encoder = LabelEncoder()
    train_df["target_col"] = label_encoder.fit_transform(train_df["target_col"])

    X = train_df["combined_text"]
    y = train_df["target_col"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train) # fit_transform()	Öğrenme ve dönüşüm yapar.	İlk kez veri işlerken kullanılır.                                                
    X_val_tfidf = vectorizer.transform(X_val) # transform()	Sadece dönüştürme yapar. Daha önce fit edilen model ile yeni veriyi işlerken kullanılır.
    X_test_tfidf = vectorizer.transform(test_df["combined_text"])

    models_and_parameters = {
        'LinearRegression': (LinearRegression(), {}),
        'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
        'Lasso': (Lasso(), {'alpha': [0.01, 0.1, 1.0]}),
#     # 'DecisionTree': (DecisionTreeRegressor(random_state=42), {
#     #     'max_depth': [None, 10, 20, 50],
#     #     'min_samples_split': [2, 5, 10]
#     # }),              ==>>      printleme süresini uzattığı için yorum satırına aldım.
#     # 'RandomForest': (RandomForestRegressor(random_state=42), {
#     #     'n_estimators': [50, 100],
#     #     'max_depth': [None, 10, 20],
#     #     'min_samples_split': [2, 5]
#     # }),              ==>>      printleme süresini uzattığı için yorum satırına aldım.
#     # 'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
#     #     'learning_rate': [0.01, 0.1],
#     #     'n_estimators': [50, 100],
#     #     'max_depth': [3, 5]
#     # }),              ==>>      printleme süresini uzattığı için yorum satırına aldım.
#     # 'KNeighbors': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 10]})              ==>>      printleme süresini uzattığı için yorum satırına aldım.
    } 

    best_model = None 
    best_score = -np.inf
    best_params = None

    for model_name, (model, param_grid) in models_and_parameters.items():
        print(f"Training {model_name}...")
        if param_grid:
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train_tfidf, y_train)
            model = grid_search.best_estimator_
            params = grid_search.best_params_
        else:
            # Train the model without hyperparameter tuning
            model.fit(X_train_tfidf, y_train)
            params = {}

        # Evaluate the model on validation set
        y_val_pred = model.predict(X_val_tfidf)
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
    y_test_pred = best_model.predict(X_test_tfidf)

    prediction_csv = "llm_classification.csv"

    # Save predictions to a CSV file
    test_df[["winner_model_a", "winner_model_b", "winner_tie"]] = y_test_pred
    test_df['id'] = ["136060", "211333", "1233961"]
    test_df[['id', "winner_model_a", "winner_model_b", "winner_tie"]].to_csv(f"llm_classification_finetuning/predictions/{prediction_csv}", index=False)
    print(f"Predictions saved as '{prediction_csv}'.")
    return best_model, best_score, best_params



train_df = pd.read_csv("llm_classification_finetuning/datasets/train.csv")
test_df = pd.read_csv("llm_classification_finetuning/datasets/test.csv")




best_model, best_score, best_params = llm_classification(train_df, test_df)







## Bu sadece bi streamlit örneği olacak. 


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Streamlit başlığı
# st.title("Makine Öğrenimi Modeli Eğitimi")

# # Kullanıcıdan veri yüklemesini isteme
# uploaded_file = st.file_uploader("Housing Price", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write("Veri Önizleme:")
#     st.dataframe(df.head())
    
#     # Hedef değişkeni seçme
#     target_column = st.selectbox("Hedef değişkeni seçin", df.columns)
    
#     # Bağımsız ve bağımlı değişkenleri ayırma
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
    
#     # Kategorik değişkenleri sayısal hale getirme
#     X = pd.get_dummies(X)
    
#     # Veri setini eğitim ve test olarak ayırma
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     y_test = np.array(y_test)

#     # Model parametrelerini seçme
#     n_estimators = st.slider("Ağaç sayısı", min_value=10, max_value=200, value=100, step=10)
    
#     if st.button("Modeli Eğit"):
#         model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)
        
#         st.write(f"Model Doğruluk Oranı: {acc:.2f}")
#         st.write(f"Y Test değerleri: {y_test}")
#         st.write(f"Y pred değerleri: {y_pred}")
#         st.success("Model eğitildi!")


      















