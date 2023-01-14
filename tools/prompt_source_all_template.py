import json
from promptsource.templates import TemplateCollection
import datasets
import os
from datasets import load_dataset
from icl_train.data_utils.data_config import DATA_CONFIG

datasets.disable_caching()

# Get all the prompts available in PromptSource
collection = TemplateCollection()

# Print a dict where the key is the pair (dataset name, subset name)
# and the value is an instance of DatasetTemplates

def clean_before_render(sample):
    return {k.replace("-", "_"): v for k, v in sample.items()}

n = "circa_o"

name, sub_name = DATA_CONFIG[n].name

prompt = collection.get_dataset(name, sub_name)

data_files = {
    "validation": os.path.join(DATA_CONFIG[n].data_dir, "train.jsonl"),
}
dataset = load_dataset("json", data_files=data_files)

# all_template_names = story.all_template_names
# print(all_template_names)

# all_template_names = [name for name in story.all_template_names if "no_option" not in name]

# if name=="dream":
#     all_template_names = ["read_the_following_conversation_and_answer_the_question"]

# for keyword in ["multiple_choice", "most_correct", "most_suitable"]:
#     _all_template_names = [name for name in all_template_names if keyword in name]
#     if len(_all_template_names)>0:
#         all_template_names = _all_template_names

print(prompt.all_template_names)


for idx in [138]:

    # print(dataset["validation"][idx])

    for name in prompt.all_template_names:
        print(name)
        print(prompt[name].jinja.replace("\n", "\t\t"))
        print(prompt[name].metadata.metrics)
        print(prompt[name].metadata.original_task)
        sample = dataset["validation"][idx]
        sample = clean_before_render(sample)
        print(sample)
        applied = prompt[name].apply(sample)
        print(applied)
        print("=" * 100)
        print()
        
# print(story["Given statement and speaker guess job title "].apply(dataset["validation"][idx]))
# print(dataset["validation"][idx])
# print(story["Given statement and speaker guess job title "].jinja.replace("\n", "\t\t"))
