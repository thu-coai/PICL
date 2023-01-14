import json
from promptsource.templates import TemplateCollection
import datasets
from datasets import load_dataset


datasets.disable_caching()

# Get all the prompts available in PromptSource
collection = TemplateCollection()

# Print a dict where the key is the pair (dataset name, subset name)
# and the value is an instance of DatasetTemplates

story = collection.get_dataset("cos_e", "v1.11")

print(story.all_template_names)

data_files = {
    "train": "/data/gyx/data_hf/cos_e/cache/v1.11/train.jsonl",
}
dataset = load_dataset("json", data_files=data_files)

idx = 6979

print(dataset["train"][idx])

for name in story.all_template_names:
    print(name)
    print(len(story[name].apply(dataset["train"][idx])))
    print(story[name].apply(dataset["train"][idx]))
    print(story[name].answer_choices)

