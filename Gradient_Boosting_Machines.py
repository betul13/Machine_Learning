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

#########################################################
#Gradient Boosting Machines
#########################################################

gbm_model = GradientBoostingClassifier(random_state = 17)

gbm_model.get_params()

cv_results = cross_validate(gbm_model, X, y, cv = 10, scoring = ["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()



#Bu parametre değerlerini GridSearch ile aramaya sokarak en iyi değerleri aramamız gerekir.

gbm_params = {"learning_rate" : [0.01, 0.1], #train süresi bu değer ne kadar küçük olursa o kadar artar.Ancak küçük olması başarı oranını arttırır.
             "max_depth" : [3,8,10],
             "n_estimators" : [100,500,1000],
             "subsample" : [1, 0.5, 0.7]} #değerlendirilecek gözlem oranları.
# #parametre değerlerini,ön tanımlı değerlerin etrafında girdim.

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv = 5, n_jobs = -1, verbose = True).fit(X,y)

gbm_best_grid.best_params_

# çıktı : {'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 100, 'subsample': 0.5}

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state = 17).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv = 10, scoring = ["accuracy", "f1", "roc_auc"])

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

plot_importance(gbm_final, X)

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

val_curve_params(gbm_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")

#################################################
# XGBoost
#################################################

xgboost_model = XGBClassifier(random_state = 17)

cv_results = cross_validate(xgboost_model, X, y, cv = 5, scoring = ["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

#cv_results["test_accuracy"].mean()
# Out[22]: 0.7409557762498938
# cv_results["test_f1"].mean()
# Out[23]: 0.6180796532975465
# cv_results["test_roc_auc"].mean()
# Out[24]: 0.7934919636617749

xgboost_model.get_params()

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth" : [5, 8, None],
                  "n_estimators" : [100, 500, 1000],
                  "colsample_bytree" : [None, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv = 5, n_jobs = -1, verbose = True).fit(X,y)

xgboost_best_grid.best_params_

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_).fit(X,y)

cv_results = cross_validate(xgboost_final, X, y, cv = 5, scoring = ["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

#################################################
# LightGBM
#################################################

lgbm_model = LGBMClassifier(random_state = 17,colsample_bytree = 0.9, learning_rate = 0.01) #bu hiperparametreler için en iyi değerleri bulduk. n_estimators'a odaklanıyoruz.

cv_results = cross_validate(lgbm_model, X, y, cv = 5, scoring = ["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

lgbm_model.get_params()

#hiperparametre kombinasyonuna makul bir yer bulduktan sonra n_estimators sayısını değiştirerek denemek lazım.
lgbm_params = {"n_estimators" : [100,200,201,202,203,204,205,210], # en önemli hiperparametre
                }

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv = 5, n_jobs = -1, verbose = True).fit(X,y)

lgbm_best_grid.best_params_

# lgbm_final modelini oluştururken virgül kullanmadan tanımlayın
lgbm_final = LGBMClassifier(random_state=17, **lgbm_best_grid.best_params_)

# cross_validate fonksiyonunu çağırırken modeli direkt olarak verin
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

#################################################
# CatBoost
#################################################

catboost_model = CatBoostClassifier(random_state = 17, verbose = False)

cv_results = cross_validate(catboost_model, X, y, cv = 5, scoring = ["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

catboost_params = {"iterations" : [200, 500],
                  "learning_rate" : [0.01, 0.1],
                  "depth" : [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv = 5, n_jobs = -1, verbose = True).fit(X,y)

catboost_final = CatBoostClassifier(random_state=17, **catboost_best_grid.best_params_).fit(X, y)

# cross_validate fonksiyonunu çağırırken modeli direkt olarak verin
cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

##########################################################################
# Feature Importance
##########################################################################

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

plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)
