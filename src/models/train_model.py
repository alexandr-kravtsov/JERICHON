# src/models/train_model.py
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from optuna.samplers import TPESampler
import joblib
import os
from src.backtest.backtest_ml import backtest_strategy

def backtest_evaluate(model, X_test, y_test, df_dollar, df_features, events, initial_capital=100000):
    trades_df, equity_curve = backtest_strategy(
        df_dollar, df_features, model, events,
        initial_capital=initial_capital,
        stop_loss_pct=0.006, take_profit_pct=0.009,
        test_events_only=True, X_test=X_test
    )
    if trades_df.empty:
        return {'total_return': -999, 'max_drawdown': 1, 'sharpe': -999, 'win_rate': 0, 'trades': 0}
    total_return = equity_curve['equity'].iloc[-1] / initial_capital - 1
    max_drawdown = (equity_curve['equity'].cummax() - equity_curve['equity']).max() / equity_curve['equity'].cummax().max()
    returns = equity_curve['equity'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() != 0 else 0
    win_rate = (trades_df['pnl'] > 0).mean()
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'trades': len(trades_df)
    }

def train_and_select_best(X_train, y_train, X_test, y_test,
                          df_dollar_full, df_features_full, events_full,
                          n_trials=30, timeout=None,
                          metric='sharpe', max_drawdown_limit=0.15):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'random_state': 42,
            'verbosity': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        return 1.0 - accuracy_score(y_test, model.predict(X_test))

    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y_train_enc)),
            'verbosity': 0
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train_enc)
        return 1.0 - accuracy_score(y_test_enc, model.predict(X_test))

    def objective_cat(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            'random_state': 42,
            'verbose': False,
            'thread_count': -1,
            'auto_class_weights': 'Balanced'
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        return 1.0 - accuracy_score(y_test, model.predict(X_test))

    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return 1.0 - accuracy_score(y_test, model.predict(X_test))

    models_config = [
        ('LightGBM', lgb.LGBMClassifier, objective_lgb, y_train),
        ('XGBoost', xgb.XGBClassifier, objective_xgb, y_train_enc),
        ('CatBoost', CatBoostClassifier, objective_cat, y_train),
        ('RandomForest', RandomForestClassifier, objective_rf, y_train)
    ]

    best_candidates = []

    for name, cls, objective_func, y_train_target in models_config:
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(lambda trial: objective_func(trial), n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        best_params = study.best_params

        if name == 'XGBoost':
            model = cls(**best_params, random_state=42, n_jobs=-1,
                        objective='multi:softmax', num_class=len(np.unique(y_train_enc)), verbosity=0)
        elif name == 'CatBoost':
            model = cls(**best_params, random_state=42, verbose=False, thread_count=-1, auto_class_weights='Balanced')
        else:
            model = cls(**best_params, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train_target)

        if name == 'XGBoost':
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test_enc, y_pred)
        else:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

        backtest_metrics = backtest_evaluate(
            model, X_test, y_test, df_dollar_full, df_features_full, events_full
        )

        best_candidates.append({
            'model_name': name,
            'model': model,
            'accuracy': acc,
            'backtest': backtest_metrics,
            'best_params': best_params
        })

    if metric == 'sharpe':
        best = max(best_candidates, key=lambda x: x['backtest']['sharpe'])
    elif metric == 'return':
        best = max(best_candidates, key=lambda x: x['backtest']['total_return'])
    elif metric == 'return_per_dd':
        def score(x):
            dd = x['backtest']['max_drawdown']
            if dd > max_drawdown_limit or dd <= 0:
                return -999
            return x['backtest']['total_return'] / dd
        best = max(best_candidates, key=score)
    else:
        best = max(best_candidates, key=lambda x: x['backtest']['sharpe'])

    return best

def save_model(model, ticker, model_dir='data/models'):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f'{ticker}_model.pkl')
    joblib.dump(model, path)
    return path

def load_model(ticker, model_dir='data/models'):
    path = os.path.join(model_dir, f'{ticker}_model.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None