# Ebenezer Ikechukwu Ozumah | K12455349
import io
import torch # type: ignore
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image # type: ignore
from pathlib import Path
from shiny import render, reactive # type: ignore
from shiny.express import input, render, ui # type: ignore

# Import model utils from satellite_cnn_module.py
from satellite_cnn_module import SATELLITE_MODEL, DEVICE, CLASS_NAMES, TRANSFORM


_ = SATELLITE_MODEL.eval()

all_predictions = reactive.value([])

# ----------------- HELPER FUNCTIONS -------------------
def get_uploaded_file(file_value):
    if not file_value:
        return None
    return file_value[0] if isinstance(file_value, list) else file_value


def predict_image(path: str):
    with open(path, "rb") as f:
        image = Image.open(io.BytesIO(f.read())).convert("RGB")

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        y_logits = SATELLITE_MODEL(tensor)
        probabilities = torch.softmax(y_logits, dim=1)[0].cpu().numpy()

    highest_predicted_class_label = int(probabilities.argmax())
    return highest_predicted_class_label, probabilities
# -X--------------- HELPER FUNCTIONS -----------------------X-


# ----- UI IMPLEMENTATION --------
ui.page_opts(title="Satellite Image Classifier")

# Sidebar of the page
with ui.sidebar(width="300px"):
    ui.h3("Upload Image")
    ui.input_file("image_file", "Select satellite image", multiple=False)
    with ui.div(style="margin-top: 1rem;"):
        ui.input_action_button("action_button", "Predict class", width="100%")  


# MAIN USER INTERFACE BODY
with ui.layout_columns(col_widths=[6, 6]):
    
    # IMAGE DISPLAY CARD
    with ui.card(full_screen=True):
        ui.card_header("Image")
        with ui.card_body ():
            @render.image
            @reactive.event(input.action_button)
            def display_uploaded_image():
                file = get_uploaded_file(input.image_file())
                if file is None:
                    ui.notification_show( #Displays a Notification if image is not uploaded
                        "Please upload an image first.",
                        type="error",
                        duration=5,
                    )
                    return None
                img = {"src": file["datapath"], "width": 256, "alt": "Uploaded satellite image"}
                return img

    # CLASS PROBABILITIES CARD
    with ui.card():
        ui.card_header("Class Probabilities")
        with ui.card_body():
            @render.data_frame
            @reactive.event(input.action_button)
            def probability_table():
                file = get_uploaded_file(input.image_file())
                if file is None:
                    return None

                _, probabilities = predict_image(file["datapath"])

                return pd.DataFrame({
                    "Class": CLASS_NAMES,
                    "Probability": [f"{prob:.2%}" for prob in probabilities]
                })

# PREDICTION CARD
with ui.div(style="margin-top: 1rem;"):
    with ui.card(colspan=12):
        ui.card_header("Prediction")
        @render.text
        @reactive.event(input.action_button)
        def prediction():
            file = get_uploaded_file(input.image_file())
            if file is None:
                return None    
            highest_predicted_class_label, probabilities = predict_image(file["datapath"])

            class_name = CLASS_NAMES[highest_predicted_class_label]
            confidence = probabilities[highest_predicted_class_label] * 100

            # Add current pred to all predictions array
            pred = {
                "image_datapath": file["datapath"],
                "class_name": class_name,
                "confidence": f"{confidence:.2f}%"
            } 


            all_predictions.set(all_predictions.get() + [pred]) # Add current prediction to all predictions

            prediction_desc = f"Predicted Class: {class_name} (Confidence: {confidence:.2f}%)"
            return prediction_desc

# PREDICTIONS
with ui.div(style="margin-top: 1rem;"):
    ui.h4("Plot of all predictions")
    
    with ui.card():
        @render.plot
        def plot_all_predictions():
            predictions = all_predictions.get() # Get all predictions
            num_samples = len(predictions)

            if not predictions:
                return None

            ncols = 5
            nrows = (num_samples + ncols - 1) // ncols

            fig, axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                figsize=(3 * ncols, 3 * nrows)
            )

            axes = axes.flatten()

            for i, pred in enumerate(predictions):

                image_datapath = pred["image_datapath"]
                class_name = pred["class_name"]
                confidence = pred["confidence"]

                image = Image.open(image_datapath)

                axes[i].imshow(image)
                axes[i].set_title(
                    f"{class_name}\n{confidence}"
                )
                axes[i].axis("off")

            # Hide empty subplots
            for j in range(num_samples, len(axes)):
                axes[j].axis("off")

            fig.tight_layout()

            return fig



    
