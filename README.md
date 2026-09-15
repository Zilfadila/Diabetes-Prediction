**Diabetes Risk Prediction & Data Validation** (English)

1. Project Overview

This project builds and compares classification models to predict diabetes risk from routinely-collected health indicators (glucose, blood pressure, BMI, age, family history, etc.), with particular attention to data validity and pipeline correctness before drawing conclusions from model performance.

2. Objectives

* Identify which health indicators are most predictive of diabetes risk.
* Detect and correctly handle data quality issues in the raw dataset before modeling.
* Compare multiple classification models on equal footing, without data leakage inflating any model's apparent performance.

3. Dataset

* Source: Pima Indians Diabetes Database (Kaggle/UCI, public dataset)
* Scale: 768 records, 8 health indicator features, binary Outcome (diabetic/non-diabetic)
* Target distribution: imbalanced - 500 non-diabetic (65.1%) vs. 268 diabetic (34.9%)

4. Methodology

* EDA - data types, missing-value check, correlation analysis, univariate and bivariate distribution review.
* Train-test split performed **first**, before any imputation, scaling, or resampling step - to prevent test-set information from influencing how those steps are calibrated.
* Missing value handling - Glucose, BloodPressure, SkinThickness, Insulin, and BMI contained placeholder zeros (physiologically impossible values) rather than true missing markers; these were converted to NaN and imputed using KNNImputer (n_neighbors=5, scikit-learn's default), fit on training data only.
* Feature transformation - log-transformed Insulin (right-skewed), scaled features with RobustScaler (fit on train only).
* Class imbalance handling - SMOTE applied to the training set only.
* Model comparison - Logistic Regression, Random Forest, SVM, XGBoost, and LightGBM trained and evaluated on identical train/test splits.
* Feature importance extracted from the best-performing model (Random Forest).

5. Key Findings

**Exploratory Analysis**
* Glucose showed the strongest linear correlation with diabetes Outcome (0.47), followed by BMI (0.29) and Age (0.24). BloodPressure showed the weakest correlation (0.065), with near-total overlap between diabetic and non-diabetic distributions in the bivariate analysis.

**Model Comparison**
* **Random Forest (default parameters) was the best-performing model** - Test ROC AUC 0.824, Test Recall 0.759, Test F1 0.656 - outperforming every other model tested.
* XGBoost and LightGBM showed clear overfitting: both reached a perfect 1.000 across all training metrics, while posting the *lowest* test ROC AUC of the group (0.775 and 0.789 respectively).

**Feature Importance**
* Ranking (Random Forest): Glucose > Insulin > BMI > Age > SkinThickness > DiabetesPedigreeFunction > BloodPressure > Pregnancies.
* Notably, **Insulin ranked 2nd in importance despite a modest linear correlation (0.13)** with Outcome - indicating a non-linear relationship that the tree-based model captured but the correlation matrix alone missed. This highlights that correlation and feature importance are complementary techniques, not redundant ones.

6. Data Quality & Methodology Notes (Documented Transparently)

* **Data leakage in initial imputation approach**: an earlier version of this pipeline applied KNN imputation - and selected its `n_neighbors` parameter via cross-validation - on the full dataset *before* splitting into train and test. This let test-set information influence how missing values were filled in training, inflating reported performance. This was corrected by moving the split to the first step of the pipeline.
* **Data leakage in an initial tuning attempt**: a GridSearchCV tuning attempt applied SMOTE before cross-validation, leaking information across folds (reported CV Recall of 0.870 did not hold up on the actual test set, which scored 0.704 - lower than the untuned default model). Given the untuned Random Forest was already the best-performing model, tuning was deliberately left out of the final project scope rather than carrying that risk forward.

7. Tech Stack

* Python: pandas, numpy, matplotlib, seaborn (EDA and visualization)
* scikit-learn: KNNImputer, RobustScaler, train_test_split, RandomForestClassifier, LogisticRegression, SVC
* XGBoost, LightGBM
* imbalanced-learn: SMOTE

---

**Prediksi Risiko Diabetes & Validasi Data** (Indonesia)

1. Project Overview

Project ini membangun dan membandingkan model klasifikasi untuk memprediksi risiko diabetes dari indikator kesehatan yang rutin dikumpulkan (glukosa, tekanan darah, BMI, usia, riwayat keluarga, dll), dengan perhatian khusus pada validitas data dan kebenaran pipeline sebelum menarik kesimpulan dari performa model.

2. Objectives

* Mengidentifikasi indikator kesehatan mana yang paling prediktif terhadap risiko diabetes.
* Mendeteksi dan menangani dengan benar masalah kualitas data di dataset mentah sebelum pemodelan.
* Membandingkan beberapa model klasifikasi secara setara, tanpa data leakage yang menggelembungkan performa model manapun.

3. Dataset

* Sumber: Pima Indians Diabetes Database (Kaggle/UCI, dataset publik)
* Skala: 768 record, 8 fitur indikator kesehatan, Outcome biner (diabetes/tidak)
* Distribusi target: imbalanced - 500 tidak diabetes (65.1%) vs 268 diabetes (34.9%)

4. Methodology

* EDA - tipe data, cek missing value, analisis korelasi, review distribusi univariate dan bivariate.
* Train-test split dilakukan **pertama kali**, sebelum proses imputasi, scaling, atau resampling apapun - untuk mencegah informasi dari test set memengaruhi kalibrasi proses-proses tersebut.
* Penanganan missing value - Glucose, BloodPressure, SkinThickness, Insulin, dan BMI berisi nilai nol placeholder (secara fisiologis tidak mungkin) alih-alih penanda missing yang sesungguhnya; nilai ini dikonversi jadi NaN dan diimputasi pakai KNNImputer (n_neighbors=5, default scikit-learn), di-fit hanya pada data training.
* Transformasi fitur - log-transform Insulin (right-skewed), scaling fitur dengan RobustScaler (fit hanya di train).
* Penanganan class imbalance - SMOTE diterapkan hanya pada training set.
* Perbandingan model - Logistic Regression, Random Forest, SVM, XGBoost, dan LightGBM dilatih dan dievaluasi pada train/test split yang identik.
* Feature importance diekstrak dari model dengan performa terbaik (Random Forest).

5. Key Findings

**Analisis Eksploratif**
* Glucose menunjukkan korelasi linear terkuat dengan Outcome diabetes (0.47), diikuti BMI (0.29) dan Age (0.24). BloodPressure menunjukkan korelasi terlemah (0.065), dengan overlap hampir total antara distribusi diabetes dan tidak diabetes di analisis bivariate.

**Perbandingan Model**
* **Random Forest (parameter default) adalah model dengan performa terbaik** - Test ROC AUC 0.824, Test Recall 0.759, Test F1 0.656 - mengungguli semua model lain yang diuji.
* XGBoost dan LightGBM menunjukkan overfitting yang jelas: keduanya mencapai skor sempurna 1.000 di semua metrik training, tapi justru mencatat Test ROC AUC *terendah* dari seluruh model (masing-masing 0.775 dan 0.789).

**Feature Importance**
* Ranking (Random Forest): Glucose > Insulin > BMI > Age > SkinThickness > DiabetesPedigreeFunction > BloodPressure > Pregnancies.
* Menariknya, **Insulin berada di peringkat 2 importance meski korelasi linearnya cuma 0.13** terhadap Outcome - mengindikasikan hubungan non-linear yang tertangkap model tree-based tapi terlewat oleh correlation matrix saja. Ini menunjukkan korelasi dan feature importance adalah teknik yang saling melengkapi, bukan redundan.

6. Catatan Kualitas Data & Metodologi (Didokumentasikan Secara Transparan)

* **Data leakage di pendekatan imputasi awal**: versi awal pipeline ini menerapkan imputasi KNN - dan memilih parameter `n_neighbors`-nya lewat cross-validation - pada seluruh dataset *sebelum* dipisah jadi train dan test. Ini membuat informasi dari test set memengaruhi cara missing value diisi di training, menggelembungkan performa yang dilaporkan. Ini diperbaiki dengan memindahkan split ke langkah pertama pipeline.
* **Data leakage di percobaan tuning awal**: percobaan tuning GridSearchCV menerapkan SMOTE sebelum cross-validation, membocorkan informasi antar fold (CV Recall yang dilaporkan 0.870 tidak bertahan di test set sesungguhnya, yang skornya cuma 0.704 - lebih rendah dari model default tanpa tuning). Karena Random Forest tanpa tuning sudah jadi model dengan performa terbaik, tuning sengaja tidak dilanjutkan ke scope final project, daripada membawa risiko itu lebih jauh.

7. Tech Stack

* Python: pandas, numpy, matplotlib, seaborn (EDA dan visualisasi)
* scikit-learn: KNNImputer, RobustScaler, train_test_split, RandomForestClassifier, LogisticRegression, SVC
* XGBoost, LightGBM
* imbalanced-learn: SMOTE
