import json
from promptsource.templates import TemplateCollection, Template
import datasets
from datasets import load_dataset


datasets.disable_caching()

collection = TemplateCollection()

p_name = "Easy Prompt"

new_template = Template(
    name=p_name,
    jinja="Passage: {{premise}}\nQuestion: {{hypothesis}} Yes or No?\nAnswer: ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}",
    reference="",
    metadata=Template.Metadata(
        original_task=True,
        choices_in_prompt=True,
        metrics=["Accuracy"]
    ),
    answer_choices="Yes ||| No"
)

templates = collection.get_dataset("super_glue", "rte")

if p_name in templates.all_template_names:
    templates.remove_template(p_name)
templates.add_template(new_template)

print(templates.all_template_names)
