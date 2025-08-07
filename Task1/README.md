# Neural-Net Regression with NumPy Only

This mini-project demonstrates how to build and train a fully connected neural network **from scratch** using only NumPy. The task is a **regression problem** on a synthetic non-linear dataset (e.g., cubic polynomial with noise). The project includes manual implementation of forward propagation, backpropagation, MSE loss, and an SGD optimizer — without using any machine learning libraries like TensorFlow or PyTorch.

---

## Task Description

- **Goal:** Predict a continuous variable (`y`) based on input feature(s) (`x`) using a hand-coded neural network.
- **Architecture:**  
  `Input → Dense(1, 64) → ReLU → Dense(64, 1)`
- **Loss:** Mean Squared Error (MSE)  
- **Optimizer:** Stochastic Gradient Descent (SGD)  
- **Data:** Generated from the function `y = 3x³ + 2x² + x + noise`  

---

## Steps to Run the Code

### Option 1: Using Jupyter Notebook

1. Make sure you have Python ≥ 3.7 and the following installed:
   ```bash
   pip install numpy matplotlib
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `train.ipynb` and run all cells.

---

### 🖥 Option 2: Using Python Script (if converting notebook to `.py`)

You can extract the code cells into a `train.py` file if needed, and run:

```bash
python train.py
```

---

## Output

1. **Loss Curve** — shows decrease in loss across epochs  
2. **Prediction Plot** — predicted vs actual `y` values for regression curve

---

## Observations & Convergence Notes

- ReLU helped mitigate vanishing gradients
- SGD with a small learning rate avoided exploding gradients
- The model successfully learned the noisy cubic trend

---

## Files

- `model.py` – All core components (layers, activations, loss, optimizer)
- `train.ipynb` – Data generation, training loop, loss visualization, predictions
- `README.md` – Project summary, setup, and usage instructions

---
