# 👁️ Eye-Tracking Activity Classification with BiLSTM

This project uses a **Bidirectional LSTM (BiLSTM)** model to classify desktop activities (e.g., *Read, Browse, Debug, Watch, Write*) from **eye-tracking time-series data**.  
We build a preprocessing pipeline to derive kinematic features, create sliding windows, and train a neural network to predict activity labels.  

---

## 📂 Dataset

We use the [Eye Movement Dataset for Desktop Activities](https://www.kaggle.com/datasets/namratasri01/eye-movement-data-set-for-desktop-activities).  

- **Participants:** 24  
- **Activities:** 8  
  - Browse, Debug, Interpret, Play, Read, Search, Watch, Write  
- **Files:** 192 CSVs (Participant × Activity)  
- **Columns:**  
  - `participant` (ID)  
  - `set` (A/B)  
  - `activity` (class label)  
  - `x, y` (gaze coordinates in pixels)  
  - `timestamp` (ms)  

### 🔧 Preprocessing
- Derived **kinematic features**:  
  - `vx, vy` = gaze velocity (pixels/sec)  
  - `speed` = magnitude of eye movement  
  - `ax, ay` = gaze acceleration  
- Sliding windows:  
  - Length = 200 samples  
  - Stride = 50  
- Labels: majority activity label per window  
- Train/validation split **by participant** (to prevent identity leakage).  

---

## 🧠 Model: BiLSTM Classifier

The classifier processes sequential gaze data and predicts one of 8 activities.

### Architecture
1. **Input**: `[batch, 200 timesteps, 7 features]`  
   - Features: `x, y, vx, vy, speed, ax, ay`
2. **BiLSTM Layer**  
   - Hidden size = 128 per direction → 256 output dims per timestep  
   - Output shape: `[batch, 200, 256]`
3. **Pooling**  
   - Mean pooling across timesteps → `[batch, 256]`
4. **Fully Connected Head**  
   - `Linear(256 → 256) → ReLU → Dropout`  
   - `Linear(256 → 8)` → logits
5. **Softmax + CrossEntropy Loss**  

---

## 📊 Training Pipeline

- **Loss:** CrossEntropyLoss (with optional class weights to handle imbalance)  
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)  
- **Scheduler:** CosineAnnealingLR  
- **Regularization:** Dropout (0.2), gradient clipping (1.0)  
- **Early Stopping:** patience = 6 epochs (monitored on Macro-F1)  
- **Batch Size:** 64  
- **Epochs:** 30  

---

<img width="1154" height="757" alt="image" src="https://github.com/user-attachments/assets/c78712f5-2c3a-4461-a447-7730d745dcbf" />





## 📈 Evaluation Metrics

- **Accuracy:**  


- **Macro F1-Score:**  
 
  Ensures fair performance across all classes, even if imbalanced.  

- Confusion Matrix:

  <img width="1705" height="437" alt="image" src="https://github.com/user-attachments/assets/ddabd65d-1099-4480-a63c-492ee8136c53" />

---



