from azureml.core import Workspace, Model

# Connect to the Azure ML Workspace
ws = Workspace.from_config()

# Register the model
model = Model.register(
    workspace=ws,
    model_path="C:/Users/dange/Downloads/wheatpredictor/predictions/saved_model",  # Path to your model folder
    model_name="wheat-disease-predictor"  # A name for your model
)

print(f"Model registered: {model.name}, Version: {model.version}")
