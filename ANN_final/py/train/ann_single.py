# 启动 ： /home/zhanghexu/PJT/DR_Application/ANN_final/py/train/ann_single.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# 加载训练数据
X_train = np.load('/home/zhanghexu/PJT/DR_Application/ANN_final/data/enhenced/enhenced_X_train.npy')
Y_train = np.load('/home/zhanghexu/PJT/DR_Application/ANN_final/data/enhenced/enhenced_Y_train.npy')

# 加载测试数据
X_test = np.load('/home/zhanghexu/PJT/DR_Application/ANN_final/data/enhenced/enhenced_X_test.npy')
Y_test = np.load('/home/zhanghexu/PJT/DR_Application/ANN_final/data/enhenced/enhenced_Y_test.npy')

# 数据预处理：标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def build_model(input_shape):
    model = Sequential()

    # 输入层
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.2))  # 添加Dropout防止过拟合

    # 隐藏层
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    # 输出层
    model.add(Dense(1, activation='sigmoid'))  # 二分类任务

    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# 构建模型
input_shape = X_train.shape[1]
model = build_model(input_shape)

# 训练模型
history = model.fit(
    X_train,
    Y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,  # 20%数据用于验证
    verbose=1
)

# 测试模型
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')

# 保存模型
model.save('ann_model.h5')

# 加载模型（需要时）
# model = tf.keras.models.load_model('ann_model.h5')