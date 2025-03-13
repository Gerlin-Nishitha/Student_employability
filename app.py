import gradio as gr
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
import os

def load_or_train_models():
    if os.path.exists("scaler.pkl") and os.path.exists("logistic_regression.pkl") and os.path.exists("perceptron.pkl"):
        print("Loading pre-trained models...")
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        with open("logistic_regression.pkl", "rb") as lr_file:
            logistic_regression = pickle.load(lr_file)
        with open("perceptron.pkl", "rb") as perceptron_file:
            perceptron = pickle.load(perceptron_file)
    else:
        print("Training models from scratch...")
        df = pd.read_excel("Student-Employability-Datasets.xlsx", sheet_name="Data")
        X = df.iloc[:, 1:-2].values
        y = (df["CLASS"] == "Employable").astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        logistic_regression = LogisticRegression(random_state=42)
        logistic_regression.fit(X_train_scaled, y_train)

        perceptron = Perceptron(random_state=42)
        perceptron.fit(X_train_scaled, y_train)

        with open("scaler.pkl", "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)
        with open("logistic_regression.pkl", "wb") as lr_file:
            pickle.dump(logistic_regression, lr_file)
        with open("perceptron.pkl", "wb") as perceptron_file:
            pickle.dump(perceptron, perceptron_file)

    return scaler, logistic_regression, perceptron

scaler, logistic_regression, perceptron = load_or_train_models()

def predict_employability(name, ts, ps, tw, ad, we, ls, ci):
    if not name.strip():
        name = "The candidate"
    
    input_data = np.array([[ts, ps, tw, ad, we, ls, ci]])
    input_scaled = scaler.transform(input_data)

    pred_lr = logistic_regression.predict(input_scaled)[0]
    pred_perceptron = perceptron.predict(input_scaled)[0]
    
    prediction = "Employable 🎉" if (pred_lr + pred_perceptron) >= 1 else "Not Employable 😞"
    
    return f"{name} is {prediction}!"


def clear_inputs():
    return "", 1, 1, 1, 1, 1, 1, 1, ""


with gr.Blocks() as app:
    gr.Markdown("# 🎓 Student Employability Evaluation 🚀")
    
    gr.Markdown(
        "### 🔍 Assess Your Employability Potential! \n"
        "Use the sliders below to rate your skills, from 1️⃣ (Needs Improvement) to 5️⃣ (Excellent)."
    )

    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Full Name (Optional)")
            ts = gr.Slider(1, 5, step=1, label="Technical Skills")
            ps = gr.Slider(1, 5, step=1, label="Problem-Solving Ability")
            tw = gr.Slider(1, 5, step=1, label="Teamwork & Collaboration")
            ad = gr.Slider(1, 5, step=1, label="Adaptability")
            we = gr.Slider(1, 5, step=1, label="Work Ethic")
            ls = gr.Slider(1, 5, step=1, label="Leadership Skills")
            ci = gr.Slider(1, 5, step=1, label="Creativity & Innovation")

            with gr.Row():
                predict_btn = gr.Button("Evaluate 🎯")
                clear_btn = gr.Button("Reset 🔄")

        with gr.Column():
            result_output = gr.Textbox(label="Employability Prediction", interactive=False)

    predict_btn.click(fn=predict_employability, inputs=[name, ts, ps, tw, ad, we, ls, ci], outputs=[result_output])
    clear_btn.click(fn=clear_inputs, inputs=[], outputs=[name, ts, ps, tw, ad, we, ls, ci, result_output])

app.launch(share=True)
