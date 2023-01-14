import json
from promptsource.templates import TemplateCollection, Template
import datasets
from datasets import load_dataset


datasets.disable_caching()

collection = TemplateCollection()

p_name = "Simple Prompt"

new_template = Template(
    name=p_name,
    jinja="{{ premise }}{% if question == \"cause\" %} because {% else %} so {% endif %} ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}",
    reference="gyx",
    metadata=Template.Metadata(
        original_task=True,
        choices_in_prompt=False,
        metrics=["Accuracy"]
    ),
    answer_choices="{{choice1}} ||| {{choice2}}"
)

templates = collection.get_dataset("super_glue", "copa")

if p_name in templates.all_template_names:
    templates.remove_template(p_name)
templates.add_template(new_template)

print(templates.all_template_names)

data_files = {
    "validation": "/home/guyuxian/data_hf/copa/cache/validation.jsonl",
}
dataset = load_dataset("json", data_files=data_files)

idx = 0

# print(dataset["validation"][idx])

print(p_name)
print(templates[p_name].jinja.replace("\n", "\t\t"))
print(templates[p_name].metadata.metrics)
print(templates[p_name].get_answer_choices_list(dataset["validation"][idx]))
x = templates[p_name].apply(dataset["validation"][idx])
print()
print(x[0].strip() + templates[p_name].metadata.concate_str + x[1].strip() + "\n")
