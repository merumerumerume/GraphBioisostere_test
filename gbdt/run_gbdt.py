# run_lgbm.py

import joblib
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import numpy as np
import pickle


def prepare_dataset(records, fold, target="delta_value", absolute=True, augument=True):
    tr, va, te = cvs[fold]
    tr_index = [a["index"] for a in tr]
    va_index = [a["index"] for a in va]
    te_index = [a["index"] for a in te]
    tr_data = [records[i] for i in tr_index]
    va_data = [records[i] for i in va_index]
    te_data = [records[i] for i in te_index]

    print(f"train={len(tr_data)}  val={len(va_data)}  test={len(te_data)}")

    # 特徴量構築
    if not augument:
        X_train = np.array([np.concatenate([r['fp1'], r['fp2']]) for r in tr_data])
        y_train = np.array([abs(r[target]) if absolute else r[target] for r in tr_data])
    else:
        # 正方向と逆方向の両方を含めるデータ拡張
        X_train = []
        y_train = []
        for r in tr_data:
            # 正方向: fp1 + fp2
            X_train.append(np.concatenate([r['fp1'], r['fp2']]))
            y_train.append(abs(r[target]) if absolute else r[target])
            # 逆方向: fp2 + fp1
            X_train.append(np.concatenate([r['fp2'], r['fp1']]))
            y_train.append(abs(r[target]) if absolute else r[target])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
    X_val = np.array([np.concatenate([r['fp1'], r['fp2']]) for r in va_data])
    y_val = np.array([abs(r[target]) if absolute else r[target] for r in va_data])
    X_test = np.array([np.concatenate([r['fp1'], r['fp2']]) for r in te_data])
    y_test = np.array([abs(r[target]) if absolute else r[target] for r in te_data])

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model_regressor(fold, dataset, random_state=42, n_jobs=-1, stopping_rounds=100, verbose=True, output_dir="output", force=False):
    if os.path.exists(os.path.join(output_dir, f"fold{fold}_results.json")) and not force:
        print(f"Results already exists in {output_dir}")
        return
    X_train, y_train, X_val, y_val, X_test, y_test = dataset
    # モデル学習

    lgb_model = lgb.LGBMRegressor(
        random_state=random_state,
        n_jobs=n_jobs,
    )
    lgb_model.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.callback.log_evaluation(),
                           lgb.callback.early_stopping(stopping_rounds=stopping_rounds, verbose=verbose)])

    # モデルを保存
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"fold{fold}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(lgb_model, f)
    print(f"Model saved to {model_path}")

    # 評価
    print(f"Fold {fold}")
    all_results = {}
    for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        pred = lgb_model.predict(X)
        # 予測結果を保存
        pred_path = os.path.join(output_dir, f"fold{fold}_{name}_predictions.npy")
        np.save(pred_path, pred)
        print(f"Predictions saved to {pred_path}")

        mse = mean_squared_error(y, pred)
        r2 = r2_score(y, pred)
        pearson = pearsonr(y, pred)[0]
        spearman = spearmanr(y, pred)[0]
        kendall = kendalltau(y, pred)[0]
        print(f"{name}: RMSE={np.sqrt(mse):.4f}, R²={r2:.4f}, Pearson={pearson:.4f}, Spearman={spearman:.4f}, Kendall={kendall:.4f}")
        plt.scatter(y, pred, alpha=0.1, label=f"{name}, R²={r2:.4f}")

        # 結果を辞書に保存
        all_results[f"{name}_metrics"] = {
            "RMSE": np.sqrt(mse),
            "R2": r2,
            "Pearson": pearson,
            "Spearman": spearman,
            "Kendall": kendall
        }

    plt.xlabel("$\Delta p_chembl$")
    plt.ylabel("Predicted $\Delta p_chembl$")
    plt.title(f"Fold {fold}")
    plt.legend()
    # 散布図を保存
    scatter_path = os.path.join(output_dir, f"fold{fold}_scatter.png")
    plt.savefig(scatter_path)
    print(f"Scatter plot saved to {scatter_path}")
    plt.show()

    # 回帰モデルで分類ベースでの結果
    # 回帰モデルの予測を分類タスクとして評価
    y_train_binary = (np.abs(y_train) < 0.3).astype(int)
    y_val_binary = (np.abs(y_val) < 0.3).astype(int)
    y_test_binary = (np.abs(y_test) < 0.3).astype(int)

    all_preds = {}
    plt.figure(figsize=(10, 8))

    for name, X, y, y_binary in [
        ("Train", X_train, y_train, y_train_binary),
        ("Val", X_val, y_val, y_val_binary),
        ("Test", X_test, y_test, y_test_binary)
    ]:
        pred = lgb_model.predict(X)
        pred_binary = (np.abs(pred) < 0.3).astype(int)

        acc = accuracy_score(y_binary, pred_binary)
        prec = precision_score(y_binary, pred_binary, zero_division=0)
        rec = recall_score(y_binary, pred_binary, zero_division=0)
        f1 = f1_score(y_binary, pred_binary, zero_division=0)
        try:
            auc = roc_auc_score(y_binary, -pred)  # 値が小さいほど陽性なので-predを使用
            mcc = matthews_corrcoef(y_binary, pred_binary)
            print(f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}, MCC={mcc:.4f}")

            # 分類結果を保存
            all_results[f"{name}_classification"] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "AUC": auc,
                "MCC": mcc
            }

            # 混同行列
            cm = confusion_matrix(y_binary, pred_binary)
            print(f"Confusion Matrix:\n{cm}")

            # ROC曲線用のデータを計算
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_binary, -pred)
            plt.plot(fpr, tpr, marker='.', label=f"{name} (AUC={auc:.4f})")
            all_preds[name] = -pred
        except:
            print(f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
            all_results[f"{name}_classification"] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1
            }

    # 全ての結果をJSONで保存
    import json
    results_path = os.path.join(output_dir, f"fold{fold}_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {results_path}")

    plt.title(f"Fold {fold} - Regression Model as Classifier")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    # ROC曲線を保存
    roc_path = os.path.join(output_dir, f"fold{fold}_roc_curve.png")
    plt.savefig(roc_path)
    print(f"ROC curve saved to {roc_path}")
    plt.show()
    print()



def train_model_cls(fold, dataset, random_state=42, n_jobs=-1, stopping_rounds=100, verbose=True, output_dir="output", force=False):
    if os.path.exists(os.path.join(output_dir, f"fold{fold}_results.json")) and not force:
        print(f"Results already exists in {output_dir}")
        return
    X_train, y_train, X_val, y_val, X_test, y_test = dataset

    # ラベルを二値分類に変換
    y_train_binary = (np.abs(y_train) < 0.3).astype(int)
    y_val_binary = (np.abs(y_val) < 0.3).astype(int)
    y_test_binary = (np.abs(y_test) < 0.3).astype(int)

    # モデル学習
    lgb_model = lgb.LGBMClassifier(
        random_state=random_state,
        n_jobs=n_jobs,
    )
    lgb_model.fit(X_train, y_train_binary,
                eval_set=[(X_val, y_val_binary)],
                eval_metric='auc',
                callbacks=[lgb.callback.log_evaluation(),
                           lgb.callback.early_stopping(stopping_rounds=stopping_rounds, verbose=verbose)])

    # モデルを保存
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"fold{fold}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(lgb_model, f)
    print(f"Model saved to {model_path}")

    # 評価
    all_results = {}
    all_preds = {}
    plt.figure(figsize=(10, 8))

    for name, X, y in [
        ("Train", X_train, y_train_binary),
        ("Val", X_val, y_val_binary),
        ("Test", X_test, y_test_binary)
    ]:
        pred_proba = lgb_model.predict_proba(X)[:, 1]
        pred = lgb_model.predict(X)
        pred_path = os.path.join(output_dir, f"fold{fold}_{name}_predictions.npy")
        np.save(pred_path, pred_proba)
        print(f"Predictions saved to {pred_path}")

        acc = accuracy_score(y, pred)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        try:
            auc = roc_auc_score(y, pred_proba)
            mcc = matthews_corrcoef(y, pred)
            print(f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}, MCC={mcc:.4f}")

            # 分類結果を保存
            all_results[f"{name}_classification"] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "AUC": auc,
                "MCC": mcc
            }

            # 混同行列
            cm = confusion_matrix(y, pred)
            print(f"Confusion Matrix:\n{cm}")

            # ROC曲線用のデータを計算
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y, pred_proba)
            plt.plot(fpr, tpr, marker='.', label=f"{name} (AUC={auc:.4f})")
            all_preds[name] = pred_proba
        except:
            print(f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
            all_results[f"{name}_classification"] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1
            }

    # 全ての結果をJSONで保存
    import json
    results_path = os.path.join(output_dir, f"fold{fold}_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {results_path}")

    plt.title(f"Fold {fold} - Classification Model")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    # ROC曲線を保存
    roc_path = os.path.join(output_dir, f"fold{fold}_roc_curve.png")
    plt.savefig(roc_path)
    print(f"ROC curve saved to {roc_path}")
    plt.show()
    print()

# cvs (cross-validation splits) は実行時に読み込むように argparse でパスを受け取れるようにする

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--input_file", type=str, default="dataset-2048.joblib")
    parser.add_argument("--pkl_file", type=str, default="/home/8/uf02678/gsbsmasunaga/bioiso/splitting/tid_5cv.pkl", help="path to pickle file containing CV splits (tid_5cv.pkl)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    # cvs を指定された pkl ファイルから読み込む
    with open(args.pkl_file, "rb") as f:
        cvs = pickle.load(f)
    import joblib
    import os

    records = joblib.load(args.input_file)
    from pathlib import Path
    args.output_dir = Path(args.output_dir)
    dataset = prepare_dataset(records, args.fold, absolute=True, augument=True)
    print("lgbm-reg-abs-aug")
    train_model_regressor(args.fold, dataset, output_dir=args.output_dir/"lgbm-reg-abs-aug", force=args.force)
    print("lgbm-cls-abs-aug")
    train_model_cls(args.fold, dataset, output_dir=args.output_dir/"lgbm-cls-abs-aug", force=args.force)
    print("lgbm-reg-delta-aug")
    dataset = prepare_dataset(records, args.fold, absolute=False, augument=True)
    train_model_regressor(args.fold, dataset, output_dir=args.output_dir/"lgbm-reg-delta-aug", force=args.force)
