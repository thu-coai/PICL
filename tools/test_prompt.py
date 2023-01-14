import json
from promptsource.templates import TemplateCollection
import datasets
from datasets import load_dataset


DATA_NO_EVAL = [
    ("story_cloze_2016", "Generate Ending"),
    ("hellaswag", "Open-ended completion"),
    ("hellaswag", "Open-ended start"),
    ("hellaswag", "Topic of the context"),
    ("hellaswag", "Reversed appropriate continuation - Yes or No"),
    ("hellaswag", "Appropriate continuation - Yes or No"),
    ("hellaswag", "Topic without the ending answer"),
]


datasets.disable_caching()

# Get all the prompts available in PromptSource
collection = TemplateCollection()

# Print a dict where the key is the pair (dataset name, subset name)
# and the value is an instance of DatasetTemplates

story = collection.get_dataset("cos_e", "v1.11")

print(story.all_template_names)

# data_files = {
#     "validation": "/home/yourname/data_hf/social_i_qa/self_fix/validation.jsonl",
# }
# dataset = load_dataset("json", data_files=data_files)

# idx = 20

# print(dataset["validation"][idx])

for name in story.all_template_names:
    print(name)
    print(story[name].jinja.replace("\n", "\t\t"))
    print(story[name].metadata.metrics)
    # print(story[name].get_answer_choices_list(dataset["validation"][idx]))
