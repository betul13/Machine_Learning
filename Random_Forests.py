#RANDOM FORESTS, GBM, XGBoost, LightGBM, CatBoost

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option("display.max_columns", None) #tüm sütunları görmek içi
warnings.simplefilter(action = "ignore", category = Warning) #uyarıları kapadık

df = pd.read_csv(r"C:\Users\bett0\Desktop\datasets\diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)
#############################################################
#Random Forests
#############################################################

rf_model = RandomForestClassifier(random_state = 17)

rf_model.get_params() #parametrelerine bakıyoruz.

#max_features : Bir düğümü bölmek için düşünülen maksimum özellik sayısını belirler.
# Ormanındaki ağaçların çeşitliliğini etkiler.
#max_dept : Her bir ağacın maksimum derinliğini kontrol eder.
# Daha derin ağaçlar genellikle verideki daha karmaşık desenleri yakalayabilir, ancak aşırı uydurmaya yol açabilir.
#min_samples_split : İç bir düğümü bölmek için gereken minimum örnek sayısını belirtir.
# Daha az esnek yapıya sebep olabilir ancak gürültüyü de azaltır
#n_estimators :Bu parametre ormanındaki ağaç sayısını belirler.
# Ağaç sayısını artırmak genellikle modelin performansını artırır, ancak hesaplama maliyetini de artırır.
# Dengeyi bulmak için farklı değerlerle deneme yapabilirsiniz.

#önce çalışmanın başındaki hatalara bakalım

cv_results = cross_validate(rf_model, X, y, cv = 10, scoring = ["accuracy","f1", "roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()


#Bu parametre değerlerini GridSearch ile aramaya sokarak en iyi değerleri aramamız gerekir.

rf_params = {"max_depth" : [5, 8, None],
             "max_features" : [3, 5, 7, "auto"],
             "min_samples_split" : [2, 5, 8, 15, 20],
             "n_estimators" : [100, 200, 500]} #parametre değerlerini,ön tanımlı değerlerin etrafında girdim.

rf_best_grid = GridSearchCV(rf_model, rf_params, cv = 5 , n_jobs = -1, verbose = True).fit(X, y)

rf_best_grid.best_params_
#{'max_depth': None,
# 'max_features': 5,
# 'min_samples_split': 8,
# 'n_estimators': 500}

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state = 17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv = 10, scoring = ["accuracy","f1", "roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")
