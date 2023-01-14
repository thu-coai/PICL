import json
from promptsource.templates import TemplateCollection, Template
import datasets
from datasets import load_dataset


datasets.disable_caching()

collection = TemplateCollection()

p_name = "MSRA Prompt"

new_template = Template(
    name=p_name,
    jinja="Question: {{question}}\nAnswer: ||| {{answer_choices[choices[\"label\"].index(answerKey)]}}",
    reference="gyx",
    metadata=Template.Metadata(
        original_task=True,
        choices_in_prompt=False,
        metrics=["Accuracy"]
    ),
    answer_choices="{{choices.text | join(\"|||\")}}"
)

templates = collection.get_dataset("ai2_arc", "ARC-Challenge")

if p_name in templates.all_template_names:
    templates.remove_template(p_name)
templates.add_template(new_template)

print(templates.all_template_names)

data_files = {
    "validation": "/home/lidong1/CodeRepo/data/ai2_arc/cache/ARC-Challenge/validation.jsonl",
}
dataset = load_dataset("json", data_files=data_files)

idx = 20

# print(dataset["validation"][idx])

print(p_name)
print(templates[p_name].jinja.replace("\n", "\t\t"))
print(templates[p_name].metadata.metrics)
print(templates[p_name].get_answer_choices_list(dataset["validation"][idx]))
x = templates[p_name].apply(dataset["validation"][idx])
print(x)
print()
print(x[0].strip() + templates[p_name].metadata.concate_str + x[1].strip() + "\n")
