from azureml.core import Workspace, Model
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice

# Connect to Azure ML Workspace
ws = Workspace.from_config()

# Retrieve the registered model
model = Model(ws, name="wheat-disease-predictor")

# Create a custom environment for deployment
env = Environment(name="wheat-env")
env.docker.enabled = True
env.python.conda_dependencies.add_pip_package("tensorflow")
env.python.conda_dependencies.add_pip_package("numpy")
env.python.conda_dependencies.add_pip_package("azureml-defaults")

# Define the inference configuration
inference_config = InferenceConfig(
    environment=env,
    entry_script="score.py"  # Path to your scoring script
)

# Define deployment configuration
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    auth_enabled=True  # Enable authentication for security
)

# Deploy the model
service_name = "wheat-disease-service"
service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config
)

service.wait_for_deployment(show_output=True)
print(f"Service deployed at: {service.scoring_uri}")
