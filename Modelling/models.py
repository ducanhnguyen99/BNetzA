import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

technical_blocks_variables = [
    "yCables.all.N13.sum", "yCables.all.N57.sum", "yCables.all.tot", "yCables.circuit.N3", "yCables.circuit.N5", "yCables.circuit.N7",
    "yConnections.incl.inj.N1357.sum", "yConnections.other.dso.lower.N1to6.sum", "yConnections.other.dso.same.tot",
    "yEnergy.delivered.net.N23.sum", "yEnergy.delivered.net.N2to4.sum", "yEnergy.delivered.net.N45.sum", "yEnergy.delivered.net.N5to7.sum", "yEnergy.delivered.net.N67.sum", "yEnergy.delivered.net.tot", 
    "yInjection.net.N2to4.sum", "yInjection.net.N5to7.sum", 
    "yInstalledPower.KWKG.other.tot", "yInstalledPower.N1to4.sum", "yInstalledPower.N5to6.sum", "yInstalledPower.N5to7.sum", "yInstalledPower.N7", "yInstalledPower.nonsimcurt.N1to4.sum", 
    "yInstalledPower.nonsimcurt.N5to7.sum", "yInstalledPower.non.solar.wind.tot",
    "yInstalledPower.reducedAPFI.N1to4.sum", "yInstalledPower.reducedAPFI.N5to7.sum", "yInstalledPower.reducedAPFI.tot", "yInstalledPower.renewables.bio.hydro.tot", 
    "yInstalledPower.renewables.solar.tot", "yInstalledPower.renewables.solar.wind.tot", "yInstalledPower.renewables.wind.tot", 
    "yLines.all.N13.sum", "yLines.all.N57.sum", "yLines.all.tot", "yLines.circuit.N3", "yLines.circuit.N5", "yLines.circuit.N7",
    "yMeters.cp.ctrl.tot", "yMeters.house.tot", "yMeters.noncp.ctrl.excl.house.tot", "yMeters.noncp.ctrl.tot", "yMeters.read.tot", 
    "yNet.length.N5", "yNet.length.N7", "yNet.length.all.tot",
    "yPeakload.N4", "yPeakload.N6", "yPeakload.abs.sim.N4", "yPeakload.from.higher.sim.N4", "yPeakload.into.higher.sim.N4", "yPeakload.into.higher.sim.nett.N6"
    ]

def lasso_regression(df_train, df_test, target, model_name, outcome_transformation = "None", random_state=42):
    # split data into features and target
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    # standardize the features using only the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Performing Lasso regression...")
    
    # Lasso regression with cross-validation for hyperparameter tuning of regularization parameter alpha
    lasso = LassoCV(cv=5, random_state=random_state, max_iter=10000).fit(X_train_scaled, y_train)
    
    # predict on test data and evaluate the model
    y_train, y_train_pred, y_test, y_test_pred = model_predict(lasso, df_train, df_test, target, outcome_transformation, random_state, scaling = True)
    eval_metrics = model_evaluation(y_train, y_train_pred, y_test, y_test_pred, model_name)
    
    selected_features_lasso = np.where(lasso.coef_ != 0)[0]
    selected_feature_names_lasso = X_train.columns[selected_features_lasso]
    
    # collect variable importance
    variable_importance_dict = {
        "Feature": selected_feature_names_lasso,
        "Coefficient": lasso.coef_[selected_features_lasso]
    }
    variable_importance_df = pd.DataFrame(variable_importance_dict)
    variable_importance_df = variable_importance_df.sort_values(by="Coefficient", ascending=False).reset_index(drop=True)
    
    return eval_metrics, lasso, variable_importance_df

def lasso_feature_selection_linear_regression(df_train, df_test, target, model_name, outcome_transformation = "None", random_state=42):
    # split data into features and target
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    # standardize the features using only the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    
    print("Performing Lasso regression for feature selection...")
    
    # Lasso regression with cross-validation for hyperparameter tuning of regularization parameter alpha
    lasso = LassoCV(cv=5, random_state=random_state, max_iter=10000).fit(X_train_scaled, y_train)
    
    selected_features_lasso = np.where(lasso.coef_ != 0)[0]
    selected_feature_names_lasso = X_train.columns[selected_features_lasso]
    
    # if no features are selected by Lasso, raise an exception
    if len(selected_feature_names_lasso) == 0:
        raise ValueError("No features were selected by Lasso. Try adjusting the Lasso parameters.")
    
    # use only the selected features for the linear regression model
    X_train_selected = X_train_scaled[selected_feature_names_lasso]

    print("Performing Linear regression...")
    # fit linear regression model with selected features
    linear_regression_model = LinearRegression().fit(X_train_selected, y_train)
    
    # predict on test data and evaluate the model
    selected_columns = selected_feature_names_lasso.tolist() + [target]
    y_train, y_train_pred, y_test, y_test_pred = model_predict(linear_regression_model, df_train[selected_columns], df_test[selected_columns], target, outcome_transformation, random_state, scaling = True)
    eval_metrics = model_evaluation(y_train, y_train_pred, y_test, y_test_pred, model_name)
    
    # collect variable importance
    variable_importance_dict = {
        "Feature": selected_feature_names_lasso,
        "Coefficient": lasso.coef_[selected_features_lasso]
    }
    variable_importance_df = pd.DataFrame(variable_importance_dict)
    variable_importance_df = variable_importance_df.sort_values(by="Coefficient", ascending=False).reset_index(drop=True)
    
    return eval_metrics, linear_regression_model, variable_importance_df

def random_forest_regression(df_train, df_test, target, model_name, outcome_transformation = "None", random_state=42):
    # split data into features and target
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    # define the model
    rf = RandomForestRegressor(random_state=random_state)
    
    # define a pipeline that includes 20 most important feature selection and model training
    pipeline = Pipeline([
    ('feature_selection', SelectFromModel(rf, max_features=20)),
    ('rf', rf)
    ])
    
    # define the hyperparameter grid
    param_grid = {
        'rf__n_estimators': [ 200, 500],
        'rf__max_depth': [ 20, 30],
        'rf__min_samples_split': [5, 10],
        'rf__min_samples_leaf': [2, 4]
        }
    
    print("Performing Random Forest...")
    
    # perform GridSearchCV with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # best model
    best_model = grid_search.best_estimator_
    
    # predict on test data and evaluate the model
    y_train, y_train_pred, y_test, y_test_pred = model_predict(best_model, df_train, df_test, target, outcome_transformation, random_state)
    eval_metrics = model_evaluation(y_train, y_train_pred, y_test, y_test_pred, model_name)
    
    # feature importance
    selected_features = best_model.named_steps['feature_selection'].get_support(indices=True)
    selected_feature_names = X_train.columns[selected_features]
    feature_importance_dict = {
        "Feature": selected_feature_names,
        "Importance": best_model.named_steps['rf'].feature_importances_
    }
    feature_importance_df = pd.DataFrame(feature_importance_dict)
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    return eval_metrics, best_model, feature_importance_df

def decision_tree_regression(df_train, df_test, target, model_name, outcome_transformation = "None", random_state=42):
    # split data into features and target
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    # define the model
    dt = DecisionTreeRegressor(random_state=random_state)
    
    # define a pipeline that includes feature selection and model training
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(dt, max_features=20)),
        ('dt', dt)
    ])

    print("Performing Decision Tree...")
    
    # define the hyperparameter grid
    param_grid = {
        'dt__max_depth': [10, 20, 30],
        'dt__min_samples_split': [2, 5, 10],
        'dt__min_samples_leaf': [1, 2, 4]
    }
    
    # perform GridSearchCV with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # best model
    best_model = grid_search.best_estimator_
             
    # predict on test data and evaluate the model
    y_train, y_train_pred, y_test, y_test_pred = model_predict(best_model, df_train, df_test, target, outcome_transformation, random_state)
    eval_metrics = model_evaluation(y_train, y_train_pred, y_test, y_test_pred, model_name)
    
    # feature importance
    selected_features = best_model.named_steps['feature_selection'].get_support(indices=True)
    selected_feature_names = X_train.columns[selected_features]
    feature_importance_dict = {
        "Feature": selected_feature_names,
        "Importance": best_model.named_steps['dt'].feature_importances_
    }
    feature_importance_df = pd.DataFrame(feature_importance_dict)
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    return eval_metrics, best_model, feature_importance_df

def xgboost_regression(df_train, df_test, target, model_name, outcome_transformation="None", random_state=42, feature_selection_threshold=0.1):
    # Split data into features and target
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    # Define XGBoost parameters and perform grid search for hyperparameter tuning
    param_grid = {
        'objective': ['reg:squarederror'],
        'eval_metric': ['rmse'],
        'max_depth': [3, 5, 7],       # Adjusted depth
        'learning_rate': [0.01, 0.1], # Adjusted learning rates
        'n_estimators': [100, 200]    # Adjusted number of estimators
    }
    
    xgb_model = xgb.XGBRegressor(seed=random_state)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model from grid search
    xgboost_model = grid_search.best_estimator_

    # Perform early stopping
    early_stopping_rounds = 50
    evals = [(X_train, y_train)]
    xgboost_model.fit(X_train, y_train, eval_set=evals, early_stopping_rounds=early_stopping_rounds, verbose=False)
    
    # Get feature importances and select important features
    feature_importances = xgboost_model.feature_importances_
    feature_names = X_train.columns
    
    # Determine which features are above the importance threshold
    important_features = feature_importances > feature_selection_threshold
    
    # Ensure that the boolean index is correctly applied
    if np.sum(important_features) == 0:
        raise ValueError("No features meet the importance threshold.")
    
    # Filter features based on importance
    X_train_selected = X_train.iloc[:, important_features]
    X_test = df_test.drop(columns=[target])
    X_test_selected = X_test.iloc[:, important_features]
    
    # Retrain XGBoost with selected features
    xgboost_model.fit(X_train_selected, y_train)
    
    # Prediction
    selected_columns = list(feature_names[important_features]) + [target]
    y_train, y_train_pred, y_test, y_test_pred = model_predict(
        xgboost_model,
        df_train[selected_columns],
        df_test[selected_columns],
        target,
        outcome_transformation,
        random_state,
        scaling=False  # Scaling is not needed
    )
    
    # Evaluation
    eval_metrics = model_evaluation(y_train, y_train_pred, y_test, y_test_pred, model_name)
    
    # Collect and return feature importances for the selected features
    selected_feature_names = feature_names[important_features]
    importance_dict = {
        'Feature': selected_feature_names,
        'Importance': feature_importances[important_features]
    }
    importance_df = pd.DataFrame(importance_dict)
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return eval_metrics, xgboost_model, importance_df


def create_clusters(df_train):

    # select only variables from the technical blocks
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_train[technical_blocks_variables])

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    train_clusters = dbscan.fit_predict(X_scaled)

    # add cluster labels to train df
    df_train['Cluster'] = train_clusters
    
    print(df_train["Cluster"].value_counts())

    # split the train tf based on the cluster labels
    df_train_c0 = df_train[df_train['Cluster'] == 0].drop(['Cluster'], axis=1)
    df_train_c1 = df_train[df_train['Cluster'] == -1].drop(['Cluster'], axis=1)
    
    return df_train_c0, df_train_c1, dbscan, scaler

def cluster_based_modeling(df_train, df_test, target, model_name, outcome_transformation = "None", random_state = 42):
    
    # create clusters of network providers
    df_train_c0, df_train_c1, dbscan, scaler = create_clusters(df_train)
    
    # scale the test data using the scaler fitted on the training data
    X_test_scaled = scaler.transform(df_test[technical_blocks_variables])
    
    print("Performing Cluster-Based Modeling...")

    # apply the DBSCAN model to the test data
    test_clusters = dbscan.fit_predict(X_test_scaled)

    # add cluster labels to test df
    df_test['Cluster'] = test_clusters

    # split the test tf based on the cluster labels
    df_test_c0 = df_test[df_test['Cluster'] == 0].drop(['Cluster'], axis=1)
    df_test_c1 = df_test[df_test['Cluster'] == -1].drop(['Cluster'], axis=1)

    # train lasso on both clusters and track evaluation
    _, lasso_c0, lasso_vip_c0 = lasso_regression(df_train_c0, df_test_c0, target, model_name, outcome_transformation, random_state)
    y_train_lasso_c0, y_train_pred_lasso_c0, y_test_lasso_c0, y_test_pred_lasso_c0 = model_predict(lasso_c0, df_train_c0, df_test_c0, target, outcome_transformation, random_state, scaling = True)
    eval_metrics_c0 = model_evaluation(y_train_lasso_c0, y_train_pred_lasso_c0, y_test_lasso_c0, y_test_pred_lasso_c0, "Lasso")
    
    _, lasso_c1, lasso_vip_c1 = lasso_regression(df_train_c1, df_test_c1, target, model_name, outcome_transformation, random_state)
    y_train_lasso_c1, y_train_pred_lasso_c1, y_test_lasso_c1, y_test_pred_lasso_c1 = model_predict(lasso_c1, df_train_c1, df_test_c1, target, outcome_transformation, random_state, scaling = True)
    eval_metrics_c1 = model_evaluation(y_train_lasso_c1, y_train_pred_lasso_c1, y_test_lasso_c1, y_test_pred_lasso_c1, "Lasso")
    
    # train random forest on both clusters and track evaluation
    _, rf_c0, rf_vip_c0 = random_forest_regression(df_train_c0, df_test_c0, target, model_name, outcome_transformation, random_state)
    y_train_rf_c0, y_train_pred_rf_c0, y_test_rf_c0, y_test_pred_rf_c0 = model_predict(rf_c0, df_train_c0, df_test_c0, target, outcome_transformation, random_state)
    eval_metrics_c0_rf = model_evaluation(y_train_rf_c0, y_train_pred_rf_c0, y_test_rf_c0, y_test_pred_rf_c0, "Random Forest")
    eval_metrics_c0 = pd.concat([eval_metrics_c0, eval_metrics_c0_rf], ignore_index=True)
    
    _, rf_c1, rf_vip_c1 = random_forest_regression(df_train_c1, df_test_c1, target, model_name, outcome_transformation, random_state)
    y_train_rf_c1, y_train_pred_rf_c1, y_test_rf_c1, y_test_pred_rf_c1 = model_predict(rf_c1, df_train_c1, df_test_c1, target, outcome_transformation, random_state)
    eval_metrics_c1_rf = model_evaluation(y_train_rf_c1, y_train_pred_rf_c1, y_test_rf_c1, y_test_pred_rf_c1, "Random Forest")
    eval_metrics_c1 = pd.concat([eval_metrics_c1, eval_metrics_c1_rf], ignore_index=True)
    
    # add other models here
    
    # identify the c0 model with the lowest test MAPE
    eval_metrics_c0["Testing MAPE"] = pd.to_numeric(eval_metrics_c0["Testing MAPE"], errors='coerce')
    best_model_df_c0 = eval_metrics_c0.loc[eval_metrics_c0["Testing MAPE"].idxmin()]
    
    # identify the c1 model with the lowest test MAPE
    eval_metrics_c1["Testing MAPE"] = pd.to_numeric(eval_metrics_c1["Testing MAPE"], errors='coerce')
    best_model_df_c1 = eval_metrics_c1.loc[eval_metrics_c1["Testing MAPE"].idxmin()]
    
    # calculate new test metrics across both clusters
    best_model_c0 = best_model_df_c0["Model"]
    if best_model_c0 == "Lasso":
        y_train_c0, y_train_pred_c0, y_test_c0, y_test_pred_c0 = model_predict(lasso_c0, df_train_c0, df_test_c0, target, outcome_transformation, random_state, scaling = True)
        model_c0 = lasso_c0
    elif best_model_c0 == "Random Forest":
        y_train_c0, y_train_pred_c0, y_test_c0, y_test_pred_c0 = model_predict(rf_c0, df_train_c0, df_test_c0, target, outcome_transformation, random_state)
        model_c0 = rf_c0
        
    best_model_c1 = best_model_df_c1["Model"]
    if best_model_c1 == "Lasso":
        y_train_c1, y_train_pred_c1, y_test_c1, y_test_pred_c1 = model_predict(lasso_c1, df_train_c1, df_test_c1, target, outcome_transformation, random_state, scaling = True)
        model_c1 = lasso_c1
    elif best_model_c1 == "Random Forest":
        y_train_c1, y_train_pred_c1, y_test_c1, y_test_pred_c1 = model_predict(rf_c1, df_train_c1, df_test_c1, target, outcome_transformation, random_state)
        model_c1 = rf_c1
    
    # concatenate all data
    y_train = np.concatenate([y_train_c0, y_train_c1], axis=0)
    y_train_pred = np.concatenate([y_train_pred_c0, y_train_pred_c1], axis=0)
    y_test = np.concatenate([y_test_c0, y_test_c1], axis=0)
    y_test_pred = np.concatenate([y_test_pred_c0, y_test_pred_c1], axis=0)
    
    model_name = f"{model_name}_{best_model_c0}_{best_model_c1}"
    
    eval_metrics = model_evaluation(y_train, y_train_pred, y_test, y_test_pred, model_name)

    return eval_metrics, model_c0, model_c1
    
def safe_exp(y, max_value=700):
    # clip the input to avoid overflow
    y_clipped = np.clip(y, None, max_value)
    return np.exp(y_clipped)

def model_predict(model, df_train, df_test, target, outcome_transformation = "None", random_state=42, scaling = False):
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]
    
    # in case of a model with standardization, standardize train and test set again
    if scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # in case of outcome transformation, revert to have original scale of data
    if outcome_transformation == "log":
        y_train_pred = safe_exp(y_train_pred)
        y_test_pred = safe_exp(y_test_pred)
        y_train = safe_exp(y_train)
        y_test = safe_exp(y_test)
        
    return y_train, y_train_pred, y_test, y_test_pred


    selected_columns = list(selected_feature_names_lasso).copy()
    selected_columns.append(target)
    y_train, y_train_pred, y_test, y_test_pred = model_predict(linear_regression_model, df_train[selected_columns], df_test[selected_columns], target, outcome_transformation, random_state, scaling = True)

def model_evaluation(y_train, y_train_pred, y_test, y_test_pred, model_name):
    print("Evaluating the model...")
    
    # evaluating on train data
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

    # evaluating on test data
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
    # collect metrics
    results_dict = {
    "Model": [model_name],
    "Training RMSE": [f"{train_rmse:.2f}"],
    "Training MAE": [f"{train_mae:.2f}"],
    "Training MAPE": [f"{train_mape:.2f}"],
    "Testing RMSE": [f"{test_rmse:.2f}"],
    "Testing MAE": [f"{test_mae:.2f}"],
    "Testing MAPE": [f"{test_mape:.2f}"]
    }

    results_df = pd.DataFrame(results_dict)
    print("\n")
    print(results_df)

    return results_df