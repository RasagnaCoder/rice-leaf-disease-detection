# 🌾 Rice Leaf Disease Classification

A web-based application to detect rice leaf diseases using a trained deep learning model. Built with **TensorFlow/Keras** and an interactive **Streamlit** frontend, this app lets users either upload an image or capture one using their webcam to classify the disease on the leaf.

---

## 🚀 Features

- 📸 Upload a rice leaf image or capture it live from webcam.
- 🧠 Predicts among 3 rice leaf diseases:
  - **Bacterial Leaf Blight**
  - **Brown Spot**
  - **Leaf Smut**
- 📊 Shows prediction confidence and all class probabilities.
- 🎨 Minimal, clean user interface with Lottie animations for better experience.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Pillow (PIL)
- Streamlit
- NumPy
- Lottie Animations

---



### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Trained Model

Place your trained model file (`rice_leaf_disease_model.keras` or `.h5`) in the root directory.  
> **Note:** This file is already excluded in `.gitignore`.

### 5. Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
rice-leaf-disease-classification/
│
├── app.py                     # Main Streamlit app
├── lottie_helper.py          # Lottie animation loader
├── rice_leaf_disease_model.keras  # Trained model (ignored in Git)
├── requirements.txt          # Python dependencies
├── .gitignore                # Files excluded from version control
├── README.md                 # Project documentation
├── assets/                   # Optional animations/images
```

---

## 📷 Example Output

**Predicted Disease:** Brown Spot  
**Confidence:** 94.52%

**Class Probabilities:**
- Bacterial Leaf Blight: 2.13%
- Brown Spot: 94.52%
- Leaf Smut: 3.35%

---

## 🚀 Future Enhancements

- Improve model accuracy with more training data
- Add download option for prediction results
- Prediction history tracking
- User feedback for prediction correctness

---

## 📜 License

This project is developed for educational and research purposes.

