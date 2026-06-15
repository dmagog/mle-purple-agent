"""
Deterministic solver for Spaceship Titanic competition.
No LLM involved — runs fixed feature engineering + ensemble pipeline.
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def solve(workdir: str, on_status=None) -> str:
    """Run deterministic pipeline. Returns path to submission.csv."""

    def status(msg):
        logger.info(msg)
        if on_status:
            on_status(msg)

    # ── Load data ──────────────────────────────────────────────────────
    status("Loading data...")
    train = pd.read_csv(os.path.join(workdir, "train.csv"))
    test = pd.read_csv(os.path.join(workdir, "test.csv"))
    test_ids = test["PassengerId"].copy()

    target = train["Transported"].astype(int)
    train = train.drop(columns=["Transported"])

    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # ── Feature Engineering ────────────────────────────────────────────
    status("Feature engineering...")

    # 1. PassengerId → group features
    combined["GroupId"] = combined["PassengerId"].apply(lambda x: x.split("_")[0]).astype(int)
    combined["PersonInGroup"] = combined["PassengerId"].apply(lambda x: x.split("_")[1]).astype(int)
    combined["GroupSize"] = combined.groupby("GroupId")["PersonInGroup"].transform("count")
    combined["IsSolo"] = (combined["GroupSize"] == 1).astype(int)

    # 2. Cabin → Deck, CabinNum, Side
    combined["Deck"] = combined["Cabin"].apply(lambda x: x.split("/")[0] if pd.notna(x) else "Unknown")
    combined["CabinNum"] = combined["Cabin"].apply(lambda x: int(x.split("/")[1]) if pd.notna(x) else -1)
    combined["Side"] = combined["Cabin"].apply(lambda x: x.split("/")[2] if pd.notna(x) else "Unknown")

    # 3. Spending features
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    # 4. CryoSleep imputation BEFORE spending features
    cryo_mask = combined["CryoSleep"] == True
    for col in spend_cols:
        combined.loc[cryo_mask, col] = combined.loc[cryo_mask, col].fillna(0)
    # Reverse: if all spending is 0, likely CryoSleep
    no_spend_mask = (combined[spend_cols].fillna(0).sum(axis=1) == 0) & combined["CryoSleep"].isna()
    combined.loc[no_spend_mask, "CryoSleep"] = True

    # Now compute spending features
    combined["TotalSpend"] = combined[spend_cols].sum(axis=1)
    combined["LogTotalSpend"] = np.log1p(combined["TotalSpend"])
    combined["NoSpend"] = (combined["TotalSpend"] == 0).astype(int)
    combined["NumSpendCategories"] = (combined[spend_cols] > 0).sum(axis=1)

    for col in spend_cols:
        combined[f"{col}_ratio"] = combined[col] / (combined["TotalSpend"] + 1)
        combined[f"Log{col}"] = np.log1p(combined[col])

    combined["LuxurySpend"] = combined["Spa"] + combined["VRDeck"] + combined["RoomService"]
    combined["BasicSpend"] = combined["FoodCourt"] + combined["ShoppingMall"]

    # 5. Age features
    combined["IsChild"] = (combined["Age"] < 12).astype(int)
    combined["IsMinor"] = (combined["Age"] < 18).astype(int)

    # 6. Name → Surname → FamilySize
    combined["Surname"] = combined["Name"].apply(lambda x: x.split()[-1] if pd.notna(x) else "Unknown")
    combined["FamilySize"] = combined.groupby("Surname")["Surname"].transform("count")

    # 7. Group-level aggregates (only key ones to avoid overfitting)
    combined["Group_TotalSpend_mean"] = combined.groupby("GroupId")["TotalSpend"].transform("mean")
    combined["Group_Age_mean"] = combined.groupby("GroupId")["Age"].transform("mean")

    # 8. Deck-level aggregates
    combined["Deck_TotalSpend_mean"] = combined.groupby("Deck")["TotalSpend"].transform("mean")

    # 9. Missing indicators — only for key columns
    for col in ["Age", "CryoSleep", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Cabin", "HomePlanet", "Destination"]:
        if col in combined.columns and combined[col].isnull().any():
            combined[f"{col}_missing"] = combined[col].isnull().astype(int)

    # 10. Imputation
    for col in combined.select_dtypes(include=[np.number]).columns:
        combined[col] = combined[col].fillna(combined[col].median())

    for col in combined.select_dtypes(include=["object", "bool"]).columns:
        if col not in ["PassengerId", "Name", "Cabin", "Surname"]:
            combined[col] = combined[col].fillna(combined[col].mode()[0] if len(combined[col].mode()) > 0 else "Unknown")

    # 11. Drop raw columns
    drop_cols = ["PassengerId", "Name", "Cabin", "Surname"]
    combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns])

    # 12. Label encode
    label_encoders = {}
    for col in combined.select_dtypes(include=["object", "bool", "category"]).columns:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
        label_encoders[col] = le

    # Split back
    X_train = combined.iloc[: len(train)].values
    X_test = combined.iloc[len(train) :].values
    y_train = target.values

    feature_names = list(combined.columns)
    status(f"Features: {len(feature_names)} columns, train shape: {X_train.shape}")

    # ── Train Models with CV ───────────────────────────────────────────
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models_config = {}

    # CatBoost
    try:
        from catboost import CatBoostClassifier

        models_config["catboost"] = lambda: CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            random_seed=42,
            early_stopping_rounds=100,
            verbose=0,
        )
    except ImportError:
        logger.warning("CatBoost not available")

    # LightGBM
    try:
        from lightgbm import LGBMClassifier

        models_config["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.3,
            reg_lambda=0.3,
            colsample_bytree=0.7,
            subsample=0.7,
            subsample_freq=5,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        logger.warning("LightGBM not available")

    # XGBoost
    try:
        from xgboost import XGBClassifier

        models_config["xgboost"] = lambda: XGBClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            reg_alpha=0.3,
            reg_lambda=1.5,
            colsample_bytree=0.7,
            subsample=0.7,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
            early_stopping_rounds=100,
        )
    except ImportError:
        logger.warning("XGBoost not available")

    # Only use the 3 strongest GBDT models to avoid overfitting

    oof_preds = {}
    test_preds = {}
    cv_scores = {}

    for model_name, model_factory in models_config.items():
        status(f"Training {model_name}...")
        oof = np.zeros(len(X_train), dtype=np.float64)
        test_fold_preds = np.zeros(len(X_test), dtype=np.float64)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = model_factory()

            if model_name == "catboost":
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
                oof[val_idx] = model.predict_proba(X_val)[:, 1]
                test_fold_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
            elif model_name == "xgboost":
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                oof[val_idx] = model.predict_proba(X_val)[:, 1]
                test_fold_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
            elif model_name == "lightgbm":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        __import__("lightgbm").early_stopping(200, verbose=False),
                        __import__("lightgbm").log_evaluation(0),
                    ],
                )
                oof[val_idx] = model.predict_proba(X_val)[:, 1]
                test_fold_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
            else:
                # sklearn models — no early stopping
                model.fit(X_tr, y_tr)
                oof[val_idx] = model.predict_proba(X_val)[:, 1]
                test_fold_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

        cv_acc = accuracy_score(y_train, (oof > 0.5).astype(int))
        cv_scores[model_name] = cv_acc
        oof_preds[model_name] = oof
        test_preds[model_name] = test_fold_preds
        status(f"{model_name} CV accuracy: {cv_acc:.5f}")

    # ── Stacking Ensemble ──────────────────────────────────────────────
    status("Building ensemble...")

    model_names = list(cv_scores.keys())

    # Method 1: Weighted average by CV score
    weights = np.array([cv_scores[m] for m in model_names])
    weights = weights / weights.sum()

    weighted_test = np.zeros(len(X_test), dtype=np.float64)
    weighted_oof = np.zeros(len(X_train), dtype=np.float64)
    for i, m in enumerate(model_names):
        weighted_test += test_preds[m] * weights[i]
        weighted_oof += oof_preds[m] * weights[i]

    weighted_cv = accuracy_score(y_train, (weighted_oof > 0.5).astype(int))
    status(f"Weighted ensemble CV accuracy: {weighted_cv:.5f}")

    # Method 2: Stacking with LogisticRegression
    from sklearn.linear_model import LogisticRegressionCV

    oof_stack = np.column_stack([oof_preds[m] for m in model_names])
    test_stack = np.column_stack([test_preds[m] for m in model_names])

    meta = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)
    meta.fit(oof_stack, y_train)
    meta_oof = meta.predict_proba(oof_stack)[:, 1]
    meta_cv = accuracy_score(y_train, (meta_oof > 0.5).astype(int))
    status(f"Stacking ensemble CV accuracy: {meta_cv:.5f}")

    # Pick the best ensemble
    if meta_cv >= weighted_cv:
        status(f"Using stacking ensemble (CV={meta_cv:.5f})")
        final_proba = meta.predict_proba(test_stack)[:, 1]
    else:
        status(f"Using weighted ensemble (CV={weighted_cv:.5f})")
        final_proba = weighted_test

    # ── Generate Submission ────────────────────────────────────────────
    predictions = (final_proba > 0.5)

    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": predictions,
    })
    submission["Transported"] = submission["Transported"].map({True: "True", False: "False"})

    submission_path = os.path.join(workdir, "submission.csv")
    submission.to_csv(submission_path, index=False)

    # Validate
    sample_path = os.path.join(workdir, "sample_submission.csv")
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        assert len(submission) == len(sample), f"Row mismatch: {len(submission)} vs {len(sample)}"
        assert list(submission.columns) == list(sample.columns), f"Column mismatch"
        status(f"Validation PASSED: {len(submission)} rows")

    status(f"DONE! Best CV: {max(weighted_cv, meta_cv):.5f}")
    return submission_path
