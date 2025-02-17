# 启动：python3 /home/zhanghexu/PJT/DR_Application/ANN_final/py/Class_forest.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib  # 用于保存模型

# 1. 加载数据
train_path = '/home/zhanghexu/PJT/DR_Application/ANN_final/data/train.csv'
test_path = '/home/zhanghexu/PJT/DR_Application/ANN_final/data/test.csv'

# 列名
column_names = ['lp', 'ln', 'hc', 'orders']

# 加载训练集和测试集
train_data = pd.read_csv(train_path, names=column_names, header=None)
test_data = pd.read_csv(test_path, names=column_names, header=None)

# 2. 分离特征和目标变量
X_train = train_data[['lp', 'ln', 'hc']]
y_train = train_data['orders']
X_test = test_data[['lp', 'ln', 'hc']]
y_test = test_data['orders']

# 确保所有数据都是数值型
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# 删除包含缺失值的行（如果有的话）
X_train.dropna(inplace=True)
y_train = y_train[X_train.index]
X_test.dropna(inplace=True)
y_test = y_test[X_test.index]

# 3. 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 训练模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# 5. 评估模型
y_pred = clf.predict(X_test_scaled)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 输出分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 输出混淆矩阵
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. 交叉验证
cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# 7. 保存和加载模型
model_path = "random_forest_model.pkl"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# 加载模型
loaded_model = joblib.load(model_path)
y_pred_loaded = loaded_model.predict(X_test_scaled)
print(f"Loaded Model Accuracy: {accuracy_score(y_test, y_pred_loaded):.4f}")