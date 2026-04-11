# Load your migrated model
from lerobot.policies.factory import get_policy_class
from lerobot.processor import PolicyProcessorPipeline

# The preprocessor and postprocessor are now external
modelrepo = "lerobot/xvla-base"
preprocessor = PolicyProcessorPipeline.from_pretrained(modelrepo, config_filename="preprocessor_config.json")
postprocessor = PolicyProcessorPipeline.from_pretrained(modelrepo, config_filename="postprocessor_config.json")

# Process data through the pipeline
processed_batch = preprocessor(raw_batch)
action = policy(processed_batch)
final_action = postprocessor(action)

print(processed_batch)
print(action)
print(final_action)