import numpy as np

# 加載訓練圖像
train_images = np.load('handwritten-digit-recognition/noisy_train_images.npy')

# 加載測試圖像
test_images = np.load('handwritten-digit-recognition/test_images.npy')


# 讀取標籤文件
with open('handwritten-digit-recognition/noisy_train_labels.txt', 'r') as file:
    labels = file.read().splitlines()

# 將標籤轉換為整數類型
labels = np.array(labels, dtype=int)

# 使用 Keras 的工具轉換為 one-hot 編碼
from keras.utils import to_categorical
labels = to_categorical(labels, num_classes=10)  # 假設有 10 個類別

# 標準化訓練圖像和測試圖像
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

from keras.models import Sequential
from keras.layers import Dense, Flatten

# 建立模型
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 將 28x28 圖像平鋪為 784 維向量
    Dense(128, activation='relu'),  # 第一層，128 個節點，使用 ReLU 激活函數
    Dense(64, activation='relu'),   # 第二層，64 個節點，使用 ReLU 激活函數
    Dense(10, activation='softmax')  # 輸出層，10 個節點，使用 softmax 激活函數對應10種類別
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 假設 train_images 和 labels 已被正確預處理和劃分
history = model.fit(train_images, labels, epochs=10, validation_split=0.1)

# 進行預測
predictions = model.predict(test_images)


# 找到每個預測的最大索引，即最可能的類別
predicted_classes = np.argmax(predictions, axis=1)

import pandas as pd

# 創建一個 DataFrame
results = pd.DataFrame({
    'ImageId': np.arange(1, len(predicted_classes) + 1),
    'Label': predicted_classes
})

# 寫入 CSV 檔案，不包含索引列
# 創建一個 DataFrame，指定列名為 'ID' 和 'Number'

results.to_csv('prediction_results.csv', index=False)
