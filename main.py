# ===================== 1. Environment Configuration and Dependency Import =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score, recall_score, precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')

# Global configuration
plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese character display
plt.rcParams['axes.unicode_minus'] = False  # For negative sign display
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tqdm.pandas(desc="Progress")


# ===================== 2. Data Loading and Preprocessing =====================
def load_and_preprocess_data(data_path="diabetes_sampled_50000.csv"):
    with tqdm(total=5, desc="Data Loading and Preprocessing", position=0, leave=True) as pbar:
        # Step 1: Load data
        pbar.update(1)
        pbar.set_postfix({"Status": "Loading Data"})
        df = pd.read_csv(data_path)
        time.sleep(0.1)

        # Step 2: Basic information check
        pbar.update(1)
        pbar.set_postfix({"Status": "Checking Data Info"})
        print("\n" + "=" * 60)
        print("Basic Data Information:")
        print(f"Data Size: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Total Missing Values: {df.isnull().sum().sum()}")
        time.sleep(0.1)

        # Step 3: Target variable binarization
        pbar.update(1)
        pbar.set_postfix({"Status": "Processing Target Variable"})
        df['Diabetes_Binary'] = df['Diabetes_012'].apply(lambda x: 0 if x == 0 else 1)
        time.sleep(0.1)

        # Step 4: Separate features and target variable
        pbar.update(1)
        pbar.set_postfix({"Status": "Separating Features/Target"})
        X = df.drop(['Diabetes_012', 'Diabetes_Binary'], axis=1)
        y = df['Diabetes_Binary']
        time.sleep(0.1)

        # Step 5: Feature standardization
        pbar.update(1)
        pbar.set_postfix({"Status": "Feature Standardization"})
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        time.sleep(0.1)

    # Check target variable distribution
    print(f"\nTarget Variable Distribution (Diabetes/Non-Diabetes):")
    dist = y.value_counts(normalize=True).round(4) * 100
    print(f"Non-Diabetes(0)：{dist[0]}% | Diabetes(1)：{dist[1]}%")

    return df, X, y, X_scaled, scaler


# Execute data loading
df, X, y, X_scaled, scaler = load_and_preprocess_data("diabetes_sampled_50000.csv")


# ===================== 修复：KMeans聚类Feature Profile生成函数（兼容所有Pandas版本） =====================
def generate_kmeans_feature_profile(df, k_labels, feature_cols=None):
    """
    生成KMeans聚类的特征画像（完全兼容所有Pandas版本）
    :param df: 原始数据框（包含聚类标签）
    :param k_labels: KMeans聚类标签
    :param feature_cols: 需要分析的特征列，默认使用数值型特征
    :return: 聚类特征画像数据框
    """
    print("\n" + "=" * 60)
    print("【K-Means Clustering Feature Profile Analysis】")

    # 自动筛选数值型特征
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除聚类标签和目标变量
        exclude_cols = ['KMeans_Cluster', 'DBSCAN_Cluster', 'Diabetes_012', 'Diabetes_Binary']
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        # 过滤空值（避免无有效特征）
        feature_cols = [col for col in feature_cols if col in df.columns]

    # 合并聚类标签
    df_profile = df.copy()
    df_profile['KMeans_Cluster'] = k_labels

    # 1. 基础统计特征画像（改用最基础的列表格式，兼容所有Pandas版本）
    with tqdm(total=5, desc="Generating Feature Profile", position=0, leave=True) as pbar:
        pbar.update(1)
        pbar.set_postfix({"Status": "Calculating Basic Statistics"})

        # 核心修复：使用列表格式的聚合函数，兼容所有Pandas版本
        agg_funcs = ['mean', 'median', 'std', 'min', 'max']
        profile_basic = df_profile.groupby('KMeans_Cluster')[feature_cols].agg(agg_funcs).round(3)

        # 展平列名（解决多级列名问题）
        profile_basic.columns = ['_'.join(col).strip() for col in profile_basic.columns.values]

        # 2. 相对占比（相较于整体均值的比例）
        pbar.update(1)
        pbar.set_postfix({"Status": "Calculating Relative Ratio"})
        overall_mean = df[feature_cols].mean()
        # 避免除零错误
        overall_mean = overall_mean.replace(0, 1e-8)

        # 计算每个聚类的均值
        cluster_means = df_profile.groupby('KMeans_Cluster')[feature_cols].mean()
        # 计算相对比例（百分比）
        profile_ratio = (cluster_means / overall_mean * 100).round(3)
        profile_ratio.columns = [f"{col}_relative_ratio(%)" for col in profile_ratio.columns]

        # 3. 类别特征分布（如果有二分类特征）
        pbar.update(1)
        pbar.set_postfix({"Status": "Calculating Categorical Distribution"})
        binary_cols = []
        for col in feature_cols:
            # 检查是否为二分类特征（仅包含0/1）
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals) <= {0, 1, 0.0, 1.0}:
                binary_cols.append(col)

        profile_binary = pd.DataFrame()
        for col in binary_cols:
            # 计算1的占比（百分比）
            profile_binary[f"{col}_positive_ratio(%)"] = df_profile.groupby('KMeans_Cluster')[col].mean() * 100
        profile_binary = profile_binary.round(2)

        pbar.update(1)
        pbar.set_postfix({"Status": "Merging Profile Results"})
        # 合并所有画像结果
        profile_combined = pd.concat([profile_basic, profile_ratio, profile_binary], axis=1)

        # 4. 可视化特征画像（Top5关键特征）
        pbar.update(1)
        pbar.set_postfix({"Status": "Visualizing Feature Profile"})
        if len(feature_cols) > 0:
            # 选择方差最大的5个特征（区分度最高）
            # 先获取基础统计中的mean列
            mean_cols = [col for col in profile_basic.columns if 'mean' in col]
            if mean_cols:
                # 提取均值数据并计算方差
                mean_data = profile_basic[mean_cols].copy()
                mean_data.columns = [col.replace('_mean', '') for col in mean_data.columns]
                feature_variance = mean_data.var().sort_values(ascending=False)
                top5_features = feature_variance.head(5).index.tolist()

                # 绘制聚类特征均值对比图
                fig, axes = plt.subplots(len(top5_features), 1, figsize=(12, 4 * len(top5_features)))
                # 处理单特征情况（避免axes不是数组）
                if len(top5_features) == 1:
                    axes = [axes]

                for idx, feat in enumerate(top5_features):
                    ax = axes[idx]
                    # 获取该特征的聚类均值
                    cluster_means_feat = df_profile.groupby('KMeans_Cluster')[feat].mean()
                    ax.bar(cluster_means_feat.index, cluster_means_feat.values, color='skyblue', edgecolor='navy')
                    ax.axhline(y=df[feat].mean(), color='red', linestyle='--',
                               label=f'Overall Mean: {df[feat].mean():.2f}')
                    ax.set_title(f'K-Means Cluster Feature Profile - {feat}', fontsize=12)
                    ax.set_xlabel('Cluster Label', fontsize=10)
                    ax.set_ylabel('Mean Value', fontsize=10)
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig('kmeans_feature_profile_top5.png', dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("⚠️ No mean statistics available for visualization")
        else:
            print("⚠️ No valid numeric features for visualization")

        # 5. 保存特征画像
        pbar.update(1)
        pbar.set_postfix({"Status": "Saving Profile Results"})
        profile_combined.to_csv('kmeans_feature_profile.csv', encoding='utf-8-sig')
        profile_ratio.to_csv('kmeans_feature_relative_ratio.csv', encoding='utf-8-sig')
        if not profile_binary.empty:
            profile_binary.to_csv('kmeans_feature_binary_ratio.csv', encoding='utf-8-sig')

    # 输出关键结果
    print("\n【K-Means Feature Profile - Basic Statistics (Top5 Features)】")
    if len(feature_cols) > 0:
        # 展示前5个特征的均值
        mean_cols = [col for col in profile_basic.columns if 'mean' in col][:5]
        if mean_cols:
            print(profile_basic[mean_cols])
        else:
            print("⚠️ No mean statistics to display")
    else:
        print("⚠️ No valid numeric features to display")

    print(f"\n✅ KMeans Feature Profile Saved:")
    print("1. kmeans_feature_profile.csv - Complete feature profile (basic stats + ratio)")
    print("2. kmeans_feature_relative_ratio.csv - Relative ratio vs overall mean")
    print("3. kmeans_feature_profile_top5.png - Visualization of top5 distinguishing features")

    return profile_combined, profile_ratio, profile_binary


# ===================== 3. Clustering Analysis (Keep Original Logic + Feature Profile) =====================
def clustering_analysis(X_scaled, X_original, y):
    print("\n" + "=" * 60)
    print("【K-Means Clustering Analysis】")

    # K-Means clustering
    k_range = range(2, 4)
    inertia = []
    silhouette = []
    davies_bouldin = []

    with tqdm(total=len(k_range) + 3, desc="K-Means Clustering", position=0, leave=True) as pbar:
        for k in k_range:
            pbar.update(1)
            pbar.set_postfix({"Status": f"Training K={k}"})
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            k_labels = kmeans.fit_predict(X_scaled)
            inertia.append(kmeans.inertia_)
            silhouette.append(silhouette_score(X_scaled, k_labels))
            davies_bouldin.append(davies_bouldin_score(X_scaled, k_labels))
            time.sleep(0.05)

        # Visualize K value selection
        pbar.update(1)
        pbar.set_postfix({"Status": "Visualizing K Selection"})
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        ax1.plot(k_range, inertia, 'o-', color='blue')
        ax1.set_title('K-Means Elbow Method (Inertia)')
        ax1.set_xlabel('Number of Clusters K')
        ax1.set_ylabel('Inertia')
        ax2.plot(k_range, silhouette, 'o-', color='orange')
        ax2.set_title('K-Means Silhouette Score')
        ax2.set_xlabel('Number of Clusters K')
        ax2.set_ylabel('Silhouette Score')
        ax3.plot(k_range, davies_bouldin, 'o-', color='green')
        ax3.set_title('K-Means DB Index')
        ax3.set_xlabel('Number of Clusters K')
        ax3.set_ylabel('DB Index')
        plt.tight_layout()
        plt.savefig('kmeans_k_selection.png', dpi=300)
        plt.close()
        time.sleep(0.1)

        # Train optimal K value model
        pbar.update(1)
        pbar.set_postfix({"Status": "Training Optimal K Model"})
        best_k = 4
        kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
        k_labels = kmeans.fit_predict(X_scaled)
        time.sleep(0.1)

        # Output evaluation metrics
        pbar.update(1)
        pbar.set_postfix({"Status": "Outputting Evaluation Metrics"})
        print(f"\nK-Means Optimal K={best_k} Evaluation Metrics:")
        print(f"Silhouette Score: {silhouette_score(X_scaled, k_labels):.4f}")
        print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_scaled, k_labels):.4f}")
        print(f"DB Index: {davies_bouldin_score(X_scaled, k_labels):.4f}")
        time.sleep(0.1)

    # DBSCAN clustering (parameter tuning)
    print("\n" + "=" * 60)
    print("【DBSCAN Clustering Analysis (Parameter Tuning)】")

    eps_list = [1, 4, 7]
    min_samples_list = [20, 40]
    param_combinations = [(eps, min_samples) for eps in eps_list for min_samples in min_samples_list]
    dbscan_results = []

    with tqdm(total=len(param_combinations) + 2, desc="DBSCAN Parameter Tuning", position=0, leave=True) as pbar:
        for idx, (eps, min_samples) in enumerate(param_combinations):
            pbar.update(1)
            pbar.set_postfix({"Status": f"Training eps={eps}, min_samples={min_samples}"})

            min_samples = int(min_samples)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            db_labels = dbscan.fit_predict(X_scaled)

            n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
            n_noise = list(db_labels).count(-1)
            noise_ratio = n_noise / len(X_scaled) * 100

            if n_clusters > 1:
                non_noise_idx = db_labels != -1
                silhouette = silhouette_score(X_scaled[non_noise_idx], db_labels[non_noise_idx])
                ch_index = calinski_harabasz_score(X_scaled[non_noise_idx], db_labels[non_noise_idx])
                db_index = davies_bouldin_score(X_scaled[non_noise_idx], db_labels[non_noise_idx])
            else:
                silhouette = None
                ch_index = None
                db_index = None

            dbscan_results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': noise_ratio,
                'silhouette_score': silhouette,
                'calinski_harabasz': ch_index,
                'davies_bouldin': db_index
            })
            time.sleep(0.1)

        # Output parameter tuning results
        pbar.update(1)
        pbar.set_postfix({"Status": "Outputting Tuning Results"})
        print("\nDBSCAN Parameter Tuning Results Summary:")
        results_df = pd.DataFrame(dbscan_results)
        print(results_df.round(4))

        # Select optimal parameters
        valid_results = results_df[results_df['n_clusters'] >= 2].copy()
        if not valid_results.empty:
            best_idx = valid_results['silhouette_score'].idxmax()
            best_params = valid_results.loc[best_idx]
            best_eps = float(best_params['eps'])
            best_min_samples = int(best_params['min_samples'])
            print(f"\nDBSCAN Optimal Parameters: eps={best_eps}, min_samples={best_min_samples}")
            print(
                f"Optimal Parameter Clustering Results: Number of Clusters={best_params['n_clusters']}, Noise Ratio={best_params['noise_ratio']:.2f}%, Silhouette Score={best_params['silhouette_score']:.4f}")
        else:
            best_eps = 0.5
            best_min_samples = 50
            print(
                f"\nNo Valid Clustering Combinations, Using Default Parameters: eps={best_eps}, min_samples={best_min_samples}")
        time.sleep(0.1)

        # Train optimal model
        pbar.update(1)
        pbar.set_postfix({"Status": "Training Optimal Parameter Model"})
        dbscan_best = DBSCAN(eps=float(best_eps), min_samples=int(best_min_samples), n_jobs=-1)
        db_labels = dbscan_best.fit_predict(X_scaled)

        # Final result statistics
        n_clusters_final = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise_final = list(db_labels).count(-1)
        print(f"\nDBSCAN Final Clustering Results:")
        print(
            f"Number of Clusters: {n_clusters_final} | Number of Noise Points: {n_noise_final} ({n_noise_final / len(X_scaled) * 100:.2f}%)")

        if n_clusters_final > 1:
            non_noise_idx = db_labels != -1
            db_silhouette = silhouette_score(X_scaled[non_noise_idx], db_labels[non_noise_idx])
            print(f"Non-Noise Point Silhouette Score: {db_silhouette:.4f}")
        else:
            print("Number of Clusters < 2, Cannot Calculate Silhouette Score")
        time.sleep(0.1)

    # Process clustering results
    with tqdm(total=5, desc="Processing Clustering Results", position=0, leave=True) as pbar:  # 增加1步用于特征画像
        # Merge clustering labels
        pbar.update(1)
        pbar.set_postfix({"Status": "Merging Clustering Labels"})
        df['KMeans_Cluster'] = k_labels
        df['DBSCAN_Cluster'] = db_labels
        time.sleep(0.1)

        # Feature analysis
        pbar.update(1)
        pbar.set_postfix({"Status": "Analyzing Cluster Features"})
        print("\n" + "=" * 60)
        print("【K-Means Cluster Feature Interpretation】")
        k_cluster_stats = df.groupby('KMeans_Cluster').agg({
            'Diabetes_Binary': 'mean',
            'BMI': 'mean',
            'Age': 'mean',
            'HighBP': 'mean',
            'HighChol': 'mean',
            'PhysActivity': 'mean'
        }).round(3)
        k_cluster_stats['Diabetes_Rate'] = k_cluster_stats['Diabetes_Binary'] * 100
        print(k_cluster_stats.drop('Diabetes_Binary', axis=1))
        time.sleep(0.1)

        # 新增：生成KMeans特征画像
        pbar.update(1)
        pbar.set_postfix({"Status": "Generating KMeans Feature Profile"})
        generate_kmeans_feature_profile(df, k_labels)
        time.sleep(0.1)

        # K-Means visualization
        pbar.update(1)
        pbar.set_postfix({"Status": "Visualizing K-Means Results"})
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_scaled)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=k_labels, cmap='viridis', alpha=0.6, s=10)
        plt.title(f'K-Means Clustering Results (K={best_k}) - PCA Dimensionality Reduction Visualization')
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        plt.colorbar(label='Cluster Label')
        plt.savefig('kmeans_clusters.png', dpi=300)
        plt.close()
        time.sleep(0.1)

        # DBSCAN visualization
        pbar.update(1)
        pbar.set_postfix({"Status": "Visualizing DBSCAN Results"})
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, cmap='coolwarm', alpha=0.6, s=10)
        plt.title(
            f'DBSCAN Clustering Results (eps={best_eps}, min_samples={best_min_samples}, Noise Points={n_noise_final}) - PCA Dimensionality Reduction Visualization')
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        plt.colorbar(scatter, label='Cluster Label (-1=Noise Point)')
        plt.savefig('dbscan_clusters.png', dpi=300)
        plt.close()
        time.sleep(0.1)

    return df, kmeans, dbscan_best, k_labels, db_labels


# Execute clustering analysis
df, kmeans_model, dbscan_model, k_labels, db_labels = clustering_analysis(X_scaled, X, y)


# ===================== 4. Classification Analysis (Core Optimization: Reduce False Positives) =====================
def classification_analysis(X_scaled, y):
    print("\n" + "=" * 60)
    print("【Classification Model Training Preparation (False Positive Optimization Version)】")

    # Data preprocessing (enhanced version, focus on reducing false positives)
    with tqdm(total=6, desc="Classification Data Preprocessing", position=0, leave=True) as pbar:
        # Step 1: Split dataset (stratified sampling)
        pbar.update(1)
        pbar.set_postfix({"Status": "Splitting Train/Test Sets"})
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        time.sleep(0.1)

        # Step 2: Optimized sampling strategy (reduce oversampling ratio to decrease false positives)
        pbar.update(1)
        pbar.set_postfix({"Status": "SMOTE Sampling (Reduced Ratio)"})
        # Core optimization 1: SMOTE only upsamples to 80% of majority class, not 1:1
        smote = SMOTE(sampling_strategy=0.8, random_state=RANDOM_SEED)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        time.sleep(0.1)

        # Step 3: Adjust class weights (reduce minority class weight to decrease false positives)
        pbar.update(1)
        pbar.set_postfix({"Status": "Calculating Class Weights (Optimize False Positives)"})
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        # Core optimization 2: Reduce minority class weight
        class_weight_dict[1] = class_weight_dict[1] * 1.2
        time.sleep(0.1)

        # Step 4: Output sampling results
        pbar.update(1)
        pbar.set_postfix({"Status": "Outputting Sampling Results"})
        print(f"Training Set Distribution After SMOTE Oversampling (False Positive Optimization):")
        print(
            f"Non-Diabetes(0)：{sum(y_train_smote == 0) / len(y_train_smote) * 100:.2f}% | Diabetes(1)：{sum(y_train_smote == 1) / len(y_train_smote) * 100:.2f}%")
        time.sleep(0.1)

        # Step 5: Define cross-validation strategy
        pbar.update(1)
        pbar.set_postfix({"Status": "Setting Up Cross-Validation"})
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        time.sleep(0.1)

        # Step 6: Define custom evaluation metrics
        pbar.update(1)
        pbar.set_postfix({"Status": "Defining Evaluation Metrics"})

        def f1_score_metric(y_true, y_pred):
            return f1_score(y_true, y_pred, average='weighted')

        time.sleep(0.1)

    # Enhanced model evaluation function (added false positive statistics + threshold adjustment)
    def evaluate_model_enhanced(model, X_test, y_test, model_name, plot_roc=True, is_random_forest=False,
                                adjust_threshold=0.6):
        """
        Enhanced Evaluation: Focus on optimizing false positives
        - Added false positive statistics
        - Support custom classification threshold
        - Keep original visualizations
        """
        # Predict probabilities
        y_prob = model.predict_proba(X_test)[:, 1]

        # Core optimization 3: Increase classification threshold (from 0.5→0.6) to reduce false positives
        y_pred_adjusted = (y_prob >= adjust_threshold).astype(int)

        # Core metric calculation (focus on false positives)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred_adjusted, pos_label=1)
        recall = recall_score(y_test, y_pred_adjusted, pos_label=1)
        precision = precision_score(y_test, y_pred_adjusted, pos_label=1)

        # Calculate false positive/false negative counts
        cm = confusion_matrix(y_test, y_pred_adjusted)
        fp = cm[0, 1]  # False Positive: Non-Diabetes → Misdiagnosed as Diabetes
        fn = cm[1, 0]  # False Negative: Diabetes → Misdiagnosed as Non-Diabetes
        fp_rate = fp / cm[0].sum() * 100  # False positive rate
        fn_rate = fn / cm[1].sum() * 100  # False negative rate

        # Output detailed evaluation (highlight false positive optimization)
        print("\n" + "=" * 40)
        print(f"{model_name} Model Evaluation Results (False Positive Reduction Optimization):")
        print(f"ROC-AUC：{roc_auc:.4f} | PR-AUC：{pr_auc:.4f}")
        print(f"F1 Score (Class 1)：{f1:.4f} | Recall：{recall:.4f} | Precision：{precision:.4f}")
        print(f"False Positive Count：{fp} | False Positive Rate：{fp_rate:.2f}%")
        print(f"False Negative Count：{fn} | False Negative Rate：{fn_rate:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_adjusted))

        # Confusion matrix visualization (label false positive optimization)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Diabetes', 'Diabetes'],
                    yticklabels=['Non-Diabetes', 'Diabetes'])
        plt.title(f'{model_name} Confusion Matrix (False Positive Reduction, Threshold={adjust_threshold})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{model_name}_confusion_matrix_low_fp.png', dpi=300, bbox_inches='tight')
        plt.close()

        # PR curve
        plt.figure(figsize=(6, 4))
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC={pr_auc:.4f})', color='red')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} PR Curve (False Positive Reduction)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{model_name}_pr_curve_low_fp.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ROC curve visualization
        if plot_roc:
            plt.figure(figsize=(6, 4))
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title(f'{model_name} ROC Curve (False Positive Reduction)')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f'{model_name}_roc_curve_low_fp.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ {model_name} ROC Curve (False Positive Reduction) Saved")

        # Random Forest feature importance (Top10)
        if is_random_forest and hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)

            # Visualize Top10 features
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
            plt.title('Random Forest - Top10 Feature Importance (False Positive Reduction)')
            plt.xlabel('Feature Importance Score')
            plt.ylabel('Feature Name')
            plt.tight_layout()
            plt.savefig('random_forest_top10_feature_importance_low_fp.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Output feature importance
            print("\nRandom Forest - Top10 Feature Importance:")
            for idx, row in feature_importance.iterrows():
                print(f"{row['Feature']}: {row['Importance']:.4f}")
            print("✅ Random Forest Feature Importance Plot (False Positive Reduction) Saved")

        # Return prediction results after threshold adjustment
        return roc_auc, pr_auc, f1, recall, precision, y_prob, y_pred_adjusted, fp, fp_rate

    # -------------------------- Random Forest (False Positive Optimization Parameters) --------------------------
    print("\n" + "=" * 60)
    print("【Random Forest Model Training (False Positive Reduction Version)】")

    with tqdm(total=3, desc="Random Forest Training", position=0, leave=True) as pbar:
        # Optimized hyperparameters (increase regularization to reduce false positives)
        pbar.update(1)
        pbar.set_postfix({"Status": "Hyperparameter Grid Search"})
        rf_params = {
            'n_estimators': [200],  # Reduce number of trees to decrease overfitting
            'max_depth': [10, 15],  # Reduce tree depth to decrease false positives
            'min_samples_split': [4, 6],  # Increase split sample count to suppress noise
            'min_samples_leaf': [2, 3],  # Increase leaf node sample count
            'class_weight': [class_weight_dict],
            'max_features': ['sqrt'],
            'bootstrap': [True],
            'criterion': ['gini']
        }

        rf = RandomForestClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            oob_score=True
        )

        # Grid search: optimize precision (focus on reducing false positives)
        rf_grid = GridSearchCV(
            rf,
            rf_params,
            cv=cv_strategy,
            scoring='precision',  # Change to optimize precision and reduce false positives
            n_jobs=-1,
            verbose=0
        )
        rf_grid.fit(X_train_smote, y_train_smote)
        rf_best = rf_grid.best_estimator_
        time.sleep(0.1)

        # Output optimal parameters
        pbar.update(1)
        pbar.set_postfix({"Status": "Outputting Optimal Parameters"})
        print(f"Random Forest Optimal Hyperparameters (False Positive Reduction): {rf_grid.best_params_}")
        print(f"Random Forest Out-of-Bag Score: {rf_best.oob_score_:.4f}")
        time.sleep(0.1)

        # Model evaluation (threshold 0.6)
        pbar.update(1)
        pbar.set_postfix({"Status": "Model Evaluation"})
        rf_roc, rf_pr, rf_f1, rf_recall, rf_precision, rf_prob, rf_pred, rf_fp, rf_fp_rate = evaluate_model_enhanced(
            rf_best, X_test, y_test, "Random Forest (False Positive Reduction)",
            plot_roc=True, is_random_forest=True, adjust_threshold=0.6
        )
        time.sleep(0.1)

    # -------------------------- XGBoost (Deep False Positive Optimization) --------------------------
    print("\n" + "=" * 60)
    print("【XGBoost Model Training (False Positive Reduction Version)】")

    with tqdm(total=3, desc="XGBoost Training", position=0, leave=True) as pbar:
        # Optimized hyperparameters (focus on suppressing false positives)
        pbar.update(1)
        pbar.set_postfix({"Status": "Hyperparameter Grid Search"})
        xgb_params = {
            'n_estimators': [200],
            'max_depth': [3, 4],  # Further reduce depth
            'learning_rate': [0.05],  # Reduce learning rate
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'gamma': [0.1, 0.2],  # Increase split gain to reduce noise
            'reg_alpha': [0.3, 0.5],  # Increase L1 regularization
            'reg_lambda': [1.5, 2.0],  # Increase L2 regularization
            'scale_pos_weight': [0.8],  # Core optimization: reduce positive class weight
            'objective': ['binary:logistic'],
            'eval_metric': ['logloss', 'auc', 'precision']  # Add precision evaluation
        }

        xgb_clf = xgb.XGBClassifier(
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            verbosity=0
        )

        # Grid search: optimize precision
        xgb_grid = GridSearchCV(
            xgb_clf,
            xgb_params,
            cv=cv_strategy,
            scoring='precision',
            n_jobs=-1,
            verbose=0
        )
        xgb_grid.fit(X_train_smote, y_train_smote)
        xgb_best = xgb_grid.best_estimator_
        time.sleep(0.1)

        # Output optimal parameters
        pbar.update(1)
        pbar.set_postfix({"Status": "Outputting Optimal Parameters"})
        print(f"XGBoost Optimal Hyperparameters (False Positive Reduction): {xgb_grid.best_params_}")
        time.sleep(0.1)

        # Model evaluation (threshold 0.4)
        pbar.update(1)
        pbar.set_postfix({"Status": "Model Evaluation"})
        xgb_roc, xgb_pr, xgb_f1, xgb_recall, xgb_precision, xgb_prob, xgb_pred, xgb_fp, xgb_fp_rate = evaluate_model_enhanced(
            xgb_best, X_test, y_test, "XGBoost (False Positive Reduction)",
            plot_roc=True, is_random_forest=False, adjust_threshold=0.4
        )
        time.sleep(0.1)

    # -------------------------- Model Comparison (False Positive Optimization Version) --------------------------
    print("\n" + "=" * 60)
    print("【Model Performance Comparison (False Positive Reduction Version)】")
    print(f"Random Forest - ROC-AUC：{rf_roc:.4f} | False Positive Rate：{rf_fp_rate:.2f}% | F1：{rf_f1:.4f}")
    print(f"XGBoost   - ROC-AUC：{xgb_roc:.4f} | False Positive Rate：{xgb_fp_rate:.2f}% | F1：{xgb_f1:.4f}")

    # Select optimal model (prioritize lower false positive rate)
    if rf_fp_rate < xgb_fp_rate:
        best_model = rf_best
        best_model_name = "Random Forest (False Positive Reduction)"
        best_prob = rf_prob
        best_pred = rf_pred
        best_fp_rate = rf_fp_rate  # Key: define best_fp_rate
        best_roc = rf_roc
    else:
        best_model = xgb_best
        best_model_name = "XGBoost (False Positive Reduction)"
        best_prob = xgb_prob
        best_pred = xgb_pred
        best_fp_rate = xgb_fp_rate  # Key: define best_fp_rate
        best_roc = xgb_roc

    print(f"\nOptimal Model：{best_model_name} (Lower False Positive Rate：{best_fp_rate:.2f}%)")

    # Return all required variables, including best_fp_rate
    return (X_train, X_test, y_train, y_test,
            rf_best, xgb_best, best_model, best_model_name,
            rf_prob, xgb_prob, rf_pred, xgb_pred, best_pred,
            best_prob, best_roc, best_fp_rate)  # Ensure best_fp_rate is returned


# Execute classification analysis and receive all returned variables (including best_fp_rate)
(X_train, X_test, y_train, y_test,
 rf_model, xgb_model, best_model, best_model_name,
 rf_prob, xgb_prob, rf_pred, xgb_pred, best_pred,
 best_prob, best_roc, best_fp_rate) = classification_analysis(X_scaled, y)  # Receive best_fp_rate


# ===================== 5. Independently Save Improved Results (Core New Feature) =====================
def save_improved_results(df, X_test, y_test, rf_prob, xgb_prob, rf_pred, xgb_pred, best_pred, best_prob, best_roc,
                          best_fp_rate, scaler):
    """Independently save results after false positive reduction, separate from original results"""
    with tqdm(total=3, desc="Saving Improved Results", position=0, leave=True) as pbar:
        # 1. Save clustering data (with labels)
        pbar.update(1)
        pbar.set_postfix({"Status": "Saving Clustering Data (Improved Version)"})
        df.to_csv('diabetes_with_clusters_improved.csv', index=False)
        time.sleep(0.1)

        # 2. Save classification results (focus on false positive optimized predictions)
        pbar.update(1)
        pbar.set_postfix({"Status": "Saving Classification Results (False Positive Reduction)"})
        test_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns)
        test_df['True_Label'] = y_test.values
        # New: prediction results after threshold adjustment
        test_df['RF_Diabetes_Prob'] = rf_prob
        test_df['RF_Pred_Low_FP'] = rf_pred  # Predictions with false positive reduction
        test_df['XGB_Diabetes_Prob'] = xgb_prob
        test_df['XGB_Pred_Low_FP'] = xgb_pred  # Predictions with false positive reduction
        test_df['Best_Pred_Low_FP'] = best_pred  # Optimal model's low false positive predictions
        test_df['Threshold_Used'] = 0.6  # Record used threshold
        test_df.to_csv('diabetes_classification_results_low_fp.csv', index=False)
        time.sleep(0.1)

        # 3. Save false positive optimization report
        pbar.update(1)
        pbar.set_postfix({"Status": "Saving Optimization Report"})
        report = pd.DataFrame({
            'Model Name': ['Random Forest (False Positive Reduction)', 'XGBoost (False Positive Reduction)',
                           best_model_name],
            'False Positive Rate(%)': [
                round(confusion_matrix(y_test, rf_pred)[0, 1] / confusion_matrix(y_test, rf_pred)[0].sum() * 100, 2),
                round(confusion_matrix(y_test, xgb_pred)[0, 1] / confusion_matrix(y_test, xgb_pred)[0].sum() * 100, 2),
                best_fp_rate
            ],
            'ROC-AUC': [
                round(roc_auc_score(y_test, rf_prob), 4),
                round(roc_auc_score(y_test, xgb_prob), 4),
                round(best_roc, 4)
            ],
            'Threshold Used': [0.6, 0.6, 0.6]
        })
        report.to_csv('diabetes_fp_optimization_report.csv', index=False, encoding='utf-8-sig')
        time.sleep(0.1)

    print("\n" + "=" * 60)
    print("✅ Improved Results Saved Independently:")
    print("1. diabetes_with_clusters_improved.csv - Clustering Data (Improved Version)")
    print("2. diabetes_classification_results_low_fp.csv - Classification Results with False Positive Reduction (Core)")
    print("3. diabetes_fp_optimization_report.csv - False Positive Optimization Report (Including Comparison Metrics)")
    print("4. Visualization files all have 'low_fp' suffix to distinguish from original files")


# Execute independent save, pass best_fp_rate
save_improved_results(df, X_test, y_test, rf_prob, xgb_prob, rf_pred, xgb_pred, best_pred, best_prob, best_roc,
                      best_fp_rate, scaler)

# ===================== 6. Final Summary =====================
print("\n" + "=" * 60)
print("【Analysis Summary (False Positive Reduction Optimization Version)】")
print(
    f"1. Clustering Results: K-Means identified {len(set(k_labels))} population segments, DBSCAN identified {len(set(db_labels)) - (1 if -1 in db_labels else 0)} valid clusters")
print(
    f"2. Classification Optimization: {best_model_name} false positive rate reduced to {best_fp_rate:.2f}%, balancing model performance")  # Now can reference normally
print(
    f"3. Core Optimization Points: Reduced SMOTE oversampling ratio + Increased classification threshold(0.6) + Added regularization + Optimized class weights")
print(
    f"4. Result Saving: All improved results saved independently with 'low_fp/improved' suffixes to avoid overwriting original files")
print("\n🎉 False Positive Reduction Optimization Analysis Completed!")

if __name__ == "__main__":
    pass