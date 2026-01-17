import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, ResNet50V2, MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks
import os

#*********************************************
# 1. เทคนิค Mixed Precision: ลดแรมครึ่งนึง ทำให้เทรนไวขึ้น 2-3 เท่า
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
#*********************************************

# ================= ⚙️ CONFIG PATHS =================
CSV_PATH = '../data/csv/train_labels.csv' 
IMG_DIR = '../data/train_images'
MODEL_SAVE_DIR = '../models'
GRAPH_SAVE_DIR = '../outputs/graphs'

# Settings
IMG_SIZE = (360, 360)  
BATCH_SIZE = 8         
EPOCHS = 50            
LEARNING_RATE = 1e-4
CLASSES = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Emphysema', 'Fibrosis']
# ====================================================

# GPU Check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU DETECTED: {len(gpus)} Found.")
    except RuntimeError as e: print(e)

# Create Output Dirs
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(GRAPH_SAVE_DIR, exist_ok=True)

# Data Generator
print("⏳ Loading Data...")
df = pd.read_csv(CSV_PATH)

# 2. อัปเกรด Data Augmentation: เพิ่มความเก่งให้ AI ทนทานต่อภาพหลากหลาย
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,      # หมุนภาพเล็กน้อย
    width_shift_range=0.1,  # ขยับซ้ายขวา
    height_shift_range=0.1, # ขยับขึ้นลง
    zoom_range=0.2,         # ซูมเข้าออก (สำคัญมากสำหรับโรคปอด)
    brightness_range=[0.8, 1.2], # จำลองแสง X-ray ที่มืด/สว่างต่างกัน
    fill_mode='constant', cval=0,
    horizontal_flip=False,  # ห้ามกลับซ้ายขวา (หัวใจอยู่ซ้ายเสมอ!)
    validation_split=0.2
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df, directory=IMG_DIR, x_col='Filename', y_col=CLASSES,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', subset='training'
)
val_generator = train_datagen.flow_from_dataframe(
    dataframe=df, directory=IMG_DIR, x_col='Filename', y_col=CLASSES,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', subset='validation'
)

def create_model(model_name):
    # Dynamic Input Shape
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    inputs = layers.Input(shape=input_shape)
    
    # Select Base Model
    if model_name == 'DenseNet121':
        base = DenseNet121(include_top=False, weights='imagenet', input_tensor=inputs)
    elif model_name == 'ResNet50':
        base = ResNet50V2(include_top=False, weights='imagenet', input_tensor=inputs)
    elif model_name == 'MobileNetV2':
        base = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
    
    #  3. Smart Freeze: แช่แข็งความรู้พื้นฐาน (Transfer Learning)
    # ปล่อยให้เรียนรู้แค่ 40-50 ชั้นบนสุด เพื่อไม่ให้ลืมความรู้เดิมจาก ImageNet
    base.trainable = True
    for layer in base.layers[:-50]: 
        layer.trainable = False
    
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    
    # 4. Architecture Upgrade: เพิ่ม BatchNormalization
    x = layers.BatchNormalization()(x) # คุมตัวเลขให้นิ่ง เทรนเสถียรขึ้น
    x = layers.Dropout(0.4)(x)         # เพิ่ม Dropout เป็น 0.4 กัน Overfit (เพราะเรามีรูปน้อย)
    
    # Output Layer (บังคับ float32 เพื่อความแม่นยำสูงสุดตอนจบ)
    outputs = layers.Dense(len(CLASSES), activation='sigmoid', dtype='float32')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', tf.keras.metrics.AUC(multi_label=True, name='auc')])
    return model

# Training Loop
model_list = ['DenseNet121', 'ResNet50', 'MobileNetV2']
history_dict = {}

# 5. Callbacks Upgrade: เพิ่มตัวช่วยลด LR
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

for m_name in model_list:
    print(f"\n🚀 TRAINING: {m_name} with Input Shape {IMG_SIZE}")
    model = create_model(m_name)
    
    # Save Path
    save_path = os.path.join(MODEL_SAVE_DIR, f'best_{m_name}.keras')
    # เพิ่ม checkpoint เข้าไปใน list
    current_callbacks = callbacks_list + [
        callbacks.ModelCheckpoint(save_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1)
    ]
    
    history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=current_callbacks)
    history_dict[m_name] = history.history['val_auc']

# Plotting
plt.figure(figsize=(12, 8))
for m_name, val_auc in history_dict.items():
    plt.plot(val_auc, label=f'{m_name}', linewidth=2, marker='o')
plt.title(f'Validation AUC Comparison (Image Size: {IMG_SIZE})')
plt.xlabel('Epochs')
plt.ylabel('AUC Score')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GRAPH_SAVE_DIR, 'training_comparison.png'))
print(f"🎉 Done! Graph saved to {GRAPH_SAVE_DIR}")
