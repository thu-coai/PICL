import random


std2all = {
    # standard -> all
    "ag_news": {
        "info": {
            "judge": lambda x: x["context"] > 0
        },
        "rules": [
            ("text", "{{ context[0] }}"),
            ("label", "{{ range(0,4) | random }}")
        ]
    },
    "amazon_polarity": {
        "info": {
            "judge": lambda x: x["context"] >= 2
        },
        "rules": [
            ("content", "{{ context[0] }}"),
            ("title", "{{ context[1] }}"),
            ("label", "{{ range(0,2) | random }}"),
        ]
    },
    "anli_r1": {
        "info": {
            "judge": lambda x: x["context"] >= 3
        },
        "rules": [
            ("premise", "{{ context[0] }}"),
            ("reason", "{{ context[1] }}"),
            ("hypothesis", "{{ context[2] }}"),
            ("label", "{{ range(0,3) | random }}")
        ]
    },
    "art": {
        "info": {
            "judge": lambda x: x["context"] >= 2 and x["options"] >= 4
        },
        "rules": [
            ("observation_1", "{{ context[0] }}"),
            ("observation_2", "{{ context[1] }}"),
            ("hypothesis_1", "{{ options[0] }}"),
            ("hypothesis_2", "{{ options[1] }}"),
            ("label", "{{ range(0,2) | random }}")
        ]
    },
    "circa": {
        "info": {
            "judge": lambda x: x["context"] >= 3 and (x["context"] >= 4 or x["question"] >= 1)
        },
        "rules": [
            ("context", "{{ context[0] }}"),
            ("answer-Y", "{{ context[1] }}"),
            ("canquestion-X", "{{ context[2] }}"),
            ("question-X", "{% if question | length >= 1 %}{{ question[0] }}{% else %}{{ context[3] }}{% endif %}"),
            ("goldstandard1", "{{ range(0,8) | random }}"),
            ("goldstandard2", "{{ range(0,8) | random }}"),
            # ("judgements", None)
        ]
    },
    "cosmos_qa": {
        "info": {
            "judge": lambda x: x["options"] >= 4 and (x["context"] >= 2 or x["question"] >= 1)
        },
        "rules": [
            ("context", "{{ context[0] }}"),
            ("question", "{% if question | length >= 1 %}{{ question[0] }}{% else %}{{ context[1] }}{% endif %}"),
            ("answer0", "{{ options[0] }}"),
            ("answer1", "{{ options[1] }}"),
            ("answer2", "{{ options[2] }}"),
            ("answer3", "{{ options[3] }}"),
            ("label", "{{ range(0,4) | random }}")
        ]
    },
    "dbpedia_14": {
        "info": {
            "judge": lambda x: x["context"] >= 2
        },
        "rules": [
            ("content", "{{ context[0] }}"),
            ("title", "{{ context[1] }}"),
            ("label", "{{ range(0,14) | random }}"),
        ]
    },
    "discovery": {
        "info": {
            "judge": lambda x: x["context"] >= 2  
        },
        "rules": [
            ("sentence1", "{{ context[0] }}"),
            ("sentence2", "{{ context[1] }}"),
            ("label", "{{ range(0,173) | random }}")
        ]
    },
    "emo": {
        "info": {
            "judge": lambda x: x["context"] > 0
        },
        "rules": [
            ("text", "{{ context[0] }}"),
            ("label", "{{ range(0,4) | random }}")
        ]
    },
    "emotion": {
        "info": {
            "judge": lambda x: x["context"] > 0
        },
        "rules": [
            ("text", "{{ context[0] }}"),
            ("label", "{{ range(0,6) | random }}")
        ]
    },
    "freebase_qa": {
        "info": {
            "judge": lambda x: False # NOTE: no neg
        },
        "rules": []
    },
    "gigaword": {
        "info": {
            "judge": lambda x: x["context"] > 0
        },
        "rules": [
            ("document", "{{ context[0] }}"),
            ("summary", "{{ answer[0] }}")
        ]
    },
    "google_wellformed_query": {
        "info": {
            "judge": lambda x: x["context"] > 0
        },
        "rules": [
            ("content", "{{ context[0] }}"),
            ("rating", str(random.random()))
        ]
    },
    "hellaswag": {
        "info": {
            "judge": lambda x: x["options"] >= 4 and x["context"] >= 3
        },
        "rules": [
            ("ctx_a", "{{ context[0] }}"),
            ("ctx_b", "{{ context[1] }}"),
            ("activity_label", "{{ context[2] }}"),
            ("ctx", "{{ context[0] }} {{ context[1] }}"),
            ("endings", [
                "{{ options[0] }}",
                "{{ options[1] }}",
                "{{ options[2] }}",
                "{{ options[3] }}",
            ]),
            ("label", "{{ range(0,4) | random }}"),
            # active_label, split, split_type
    ]},
    "imdb": {
        "info": {
            "judge": lambda x: x["context"] > 0
        },
        "rules": [
            ("text", "{{ context[0] }}"),
            ("label", "{{ range(0,2) | random }}"),
        ]
    },
    "liar": {
        "info": {
            "judge": lambda x: x["context"] >= 4
        },
        "rules": [
            ("job_title", ""),
            ("speaker", "{{ context[0] }}"),
            ("context", "{{ context[1] }}"),
            ("subject", "{{ context[2] }}"),
            ("party_affiliation", "{{ context[3] }}"),
            ("label", "{{ range(0,6) | random }}")
        ]
    },
    "paws_labeled_final": {
        "info": {
            "judge": lambda x: x["context"] >= 2  
        },
        "rules": [
            ("sentence1", "{{ context[0] }}"),
            ("sentence2", "{{ context[1] }}"),
            ("label", "{{ range(0,2) | random }}")
        ]
    },
    "piqa": {
        "info": {
            "judge": lambda x: x["options"] >= 2  
        },
        "rules": [
            ("goal", "{{ context[0] }}"),
            ("sol1", "{{ options[0] }}"),
            ("sol2", "{{ options[1] }}"),
            ("label", "{{ range(0,2) | random }}")
        ]
    },
    "quail": {
        "info": {
            "judge": lambda x: x["options"] >= 4 and (x["context"] >= 2 or x["question"] >= 1)
        },
        "rules": [
            ("context", "{{ context[0] }}"),
            ("question", "{% if question | length >= 1 %}{{ question[0] }}{% else %}{{ context[1] }}{% endif %}"),
            ("answers", [
                "{{ options[0] }}",
                "{{ options[1] }}",
                "{{ options[2] }}",
                "{{ options[3] }}"
            ]),
            ("correct_answer_id", "{{ range(0,4) | random }}")
            # domain, metadata, question_type
        ]
    },
    "quoref": {
        "info": {
            "judge": lambda x: x["context"] >= 3 or (x["context"] >= 2 and x["question"] >= 1)  
        },
        "rules": [
            ("context", "{{ context[0] }}"),
            ("title", "{{ context[1] }}"),
            ("question", "{% if question | length >= 1 %}{{ question[0] }}{% else %}{{ context[2] }}{% endif %}"),
            ("answers.text", ["{{ answer[0] }}"])
        ]
    },
    "ropes": {
        "info": {
            "judge": lambda x: x["context"] >= 2 and (x["context"] >= 3 or x["question"] >= 1)
        },
        "rules": [
            ("background", "{{ context[0] }}"),
            ("situation", "{{ context[1] }}"),
            ("question", "{% if question | length >= 1 %}{{ question[0] }}{% else %}{{ context[2] }}{% endif %}"),
            ("answers.text", ["{{ answer[0] }}"])
        ]
    },
    "sciq": {
        "info": {
            "judge": lambda x: x["options"] >= 4 and (x["context"] >= 2 or x["question"] >= 1)  
        },
        "rules": [
            ("support", "{{ context[0] }}"),
            ("question", "{% if question | length >= 1 %}{{ question[0] }}{% else %}{{ context[1] }}{% endif %}"),
            ("distractor1", "{{ options[0] }}"),
            ("distractor2", "{{ options[1] }}"),
            ("distractor3", "{{ options[2] }}"),
            ("correct_answer", "{{ answer[0] }}")
        ]
    },
    "scitail": {
        "info": {
            "judge": lambda x: x["context"] >= 2
        },
        "rules": [
            ("sentence1", "{{ context[0] }}"),
            ("sentence2", "{{ context[1] }}"),
            ("gold_label", "{{ answer[0] }}")
        ]
    },
    "social_i_qa": {
        "info": {
            "judge": lambda x: x["options"] >= 3 and (x["context"] >= 2 or x["question"] >= 1)
        },
        "rules": [
            ("context", "{{ context[0] }}"),
            ("question", "{% if question | length >= 1 %}{{ question[0] }}{% else %}{{ context[1] }}{% endif %}"),
            ("answerA", "{{ options[0] }}"),
            ("answerB", "{{ options[1] }}"),
            ("answerC", "{{ options[2] }}"),
            ("label", "{{ range(1,4) | random }}")
        ]
    },
    "swag": {
        "info": {
            "judge": lambda x: x["context"] >= 2 and x["options"] >= 4
        },
        "rules": [
            ("sent1", "{{ context[0] }}"),
            ("sent2", "{{ context[1] }}"),
            ("ending0", "{{ options[0] }}"),
            ("ending1", "{{ options[1] }}"),
            ("ending2", "{{ options[2] }}"),
            ("ending3", "{{ options[3] }}"),
            ("label", "{{ range(0,4) | random }}")
            # startphrase
        ]
    },
    "tab_fact": {
        "info": {
            "judge": lambda x: False # NOTE: no neg
        },
        "rules": []
    },
    "wiki_qa": {
        "info": {
            "judge": lambda x: x["context"] >= 3 or (x["context"] >= 2 and x["question"] >= 1)
        },
        "rules": [
            ("answer", "{{ context[0] }}"),
            ("question", "{% if question | length >= 1 %}{{ question[0] }}{% else %}{{ context[2] }}{% endif %}"),
            ("document_title", "{{ context[1] }}"),
            ("label", "{{ range(0,2) | random }}")
        ]
    },
    "wiqa": {
        "info": {
            "judge": lambda x: False # NOTE: no neg
        },
        "rules": [
            ("question_para_step | join(\" \")", "context"),
            ("question_stem", "question"),
            ("answer_label", "answer")
        ]
    },
    "xsum": {
        "info": {
            "judge": lambda x: x["context"] > 0
        },
        "rules": [
            ("document", "{{ context[0] }}"),
            ("summary", "{{ answer[0] }}")
        ]
    },
    "yahoo_answers_topics": {
        "info": {
            "judge": lambda x: x["context"] >= 2
        },
        "rules": [
            ("question_content", "{{ context[0] }}"),
            ("question_title", "{{ context[1] }}"),
            ("topic", "{{ range(0,10) | random }}"),
            ("best_answer", "{{ answer[0] }}")
        ]
    },
    "yelp_polarity": {
        "info": {
            "judge": lambda x: x["context"] > 0
        },
        "rules": [
            ("text", "{{ context[0] }}"),
            ("label", "{{ range(0,2) | random }}"),
        ]
    },
    "yelp_review_full": {
        "info": {
            "judge": lambda x: x["context"] > 0
        },
        "rules": [
            ("text", "{{ context[0] }}"),
            ("label", "{{ range(0,5) | random }}"),
        ]
    },
    "cos_e": {
        "info": {
            "judge": lambda x: x["options"] >= 5 and (x["context"] >= 2 or x["question"] >= 1)
        },
        "rules": [
            ("context", "{{ context[0] }}"),
            ("question", "{% if question | length >= 1 %}{{ question[0] }}{% else %}{{ context[1] }}{% endif %}"),
            ("choices", [
                "{{ options[0] }}",
                "{{ options[1] }}",
                "{{ options[2] }}",
                "{{ options[3] }}",
                "{{ options[4] }}",
            ]),
            ("answer", "{{ answer[0] }}")
            # domain, metadata, question_type
        ]
    },
    "dream": {
        "info": {
            "judge": lambda x: x["options"] >= 3 and (x["context"] >= 2 or x["question"] >= 1)
        },
        "rules": [
            ("context", "{{ context[0] }}"),
            ("question", "{% if question | length >= 1 %}{{ question[0] }}{% else %}{{ context[1] }}{% endif %}"),
            ("choice", [
                "{{ options[0] }}",
                "{{ options[1] }}",
                "{{ options[2] }}",
            ]),
            ("answer", "{{ answer[0] }}")
            # domain, metadata, question_type
        ]
    },
    "quartz": {
        "info": {
            "judge": lambda x: False # NOTE: no neg
        },
        "rules": []
    },
    "qasc": {
        "info": {
            "judge": lambda x: False # NOTE: no neg
        },
        "rules": []
    },
    "adversarial_qa": {
        "info": {
            "judge": lambda x: x["context"] >= 2 and (x["context"] >= 3 or x["question"] >= 1)
        },
        "rules": [
            ("title", "{{ context[0] }}"),
            ("context", "{{ context[1] }}"),
            ("question", "{% if question | length >= 1 %}{{ question[0] }}{% else %}{{ context[2] }}{% endif %}"),
            ("answers.text", ["{{ answer[0] }}"])
        ]
    },
}
