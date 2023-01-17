import json
from promptsource.templates import TemplateCollection, Template
import datasets
from datasets import load_dataset


datasets.disable_caching()

collection = TemplateCollection()

p_name = "Easy Prompt"

new_template = Template(
    name=p_name,
    jinja="Sentence: {{sentence}}\nLabel: ||| {{ answer_choices[label] }}",
    reference="",
    metadata=Template.Metadata(
        original_task=True,
        choices_in_prompt=False,
        metrics=["Accuracy"]
    ),
    answer_choices="terrible ||| bad ||| neutral ||| good ||| great"
)

templates = collection.get_dataset("sst5", None)

if p_name in templates.all_template_names:
    templates.remove_template(p_name)
templates.add_template(new_template)

print(templates.all_template_names)
