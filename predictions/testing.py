from azureml.core import Workspace

ws = Workspace.from_config()
print("Workspace loaded successfully!")
