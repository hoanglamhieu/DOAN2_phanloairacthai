import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================
# 1. Cấu hình
# ==============================
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

# ==============================
# 2. Load dataset
# ==============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
class_names = train_ds.class_names
print("📂 CLASS NAMES (THỨ TỰ CHUẨN):", class_names)

class_names = train_ds.class_names
print("📂 Các loại rác:", class_names)

NUM_CLASSES = len(class_names)

# ==============================
# 3. Tối ưu pipeline dữ liệu
# ==============================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==============================
# 4. Xây dựng mô hình CNN
# ==============================
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation='softmax')
])

# ==============================
# 5. Compile model
# ==============================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# 6. Huấn luyện
# ==============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==============================
# 7. Lưu model
# ==============================
model.save("waste_cnn.h5")
print("✅ Đã lưu model waste_cnn.h5")
