all2std = {
    # all -> standard
    "ag_news_o": {
        "info": {
            "context": 1, "question": 0, "options": 4, "label": 1
        },
        "rules": [
            ("{{ text }}", "context"),
            ("L(World politics;Sports;Business;Science and technology)", "options"),
            ("{{ label | int }}", "label")
        ]
    },
    "amazon_polarity_o": {
        "info": {
            "context": 2, "question": 0, "options": 2, "label": 1
        },
        "rules": [
            ("{{ content }}", "context"),
            ("{{ title }}", "context"),
            ("L(negative;positive)", "options"),
            ("{{ label }}", "label"),
        ]
    },
    "anli_r1_o": {
        "info": {
            "context": 3, "question": 0, "options": 3, "label": 1
        },
        "rules": [
            ("{{ premise }}", "context"),
            ("{{ reason }}", "context"),
            ("{{ hypothesis }}", "context"),
            ("L(negative;neutral;positive)", "options"),
            ("{{ label }}", "label")
        ]
    },
    "art_o": {
        "info": {
            "context": 2, "question": 0, "options": 2, "label": 1
        },
        "rules": [
            ("{{ observation_1 }}", "context"),
            ("{{ observation_2 }}", "context"),
            ("{{ hypothesis_1 }}", "options"),
            ("{{ hypothesis_2 }}", "options"),
            ("{{ label | int - 1 }}", "label")
        ]
    },
    "circa_o": {
        "info": {
            "context": 3, "question": 1, "options": 8, "label": 2
        },
        "rules": [
            ("{{ context }}", "context"),
            ("{{ answer_Y }}", "context"),
            ("{{ canquestion_X }}", "context"),
            ("{{ question_X }}", "question"),
            ("L(Yes;No;In the middle, neither yes nor no;Probably yes / sometimes yes;Probably no;Yes, subject to some conditions;Other;I am not sure how X will interpret Y's answer)", "options"),
            ("{{ goldstandard1 }}", "label"),
            ("{{ goldstandard2 }}", "label")
        ]
    },
    "cosmos_qa_o": {
        "info": {
            "context": 1, "question": 1, "options": 4, "label": 1
        },
        "rules": [
            ("{{ context }}", "context"),
            ("{{ question }}", "question"),
            ("{{ answer0 }}", "options"),
            ("{{ answer1 }}", "options"),
            ("{{ answer2 }}", "options"),
            ("{{ answer3 }}", "options"),
            ("{{ label }}", "label")
        ]
    },
    "dbpedia_14_o": {
        "info": {
            "context": 2, "question": 0, "options": 4, "label": 14
        },
        "rules": [
            ("{{ content }}", "context"),
            ("{{ title }}", "context"),
            ("L(company;educational institution;artist;athlete;office holder;mean of transportation;building;natural place;village;animal;plant;album;film;written work)", "options"),
            ("{{ label }}", "label"),
        ]
    },
    "discovery_o": {
        "info": {
            "context": 2, "question": 0, "options": 174, "label": 14
        },
        "rules": [
            ("{{ sentence1 }}", "context"),
            ("{{ sentence2 }}", "context"),
            ("L(no connection;absolutely;accordingly;actually;additionally;admittedly;afterward;again;already;also;alternately;alternatively;although;altogether;amazingly;and;anyway;apparently;arguably;as a result;basically;because of that;because of this;besides;but;by comparison;by contrast;by doing this;by then;certainly;clearly;coincidentally;collectively;consequently;conversely;curiously;currently;elsewhere;especially;essentially;eventually;evidently;finally;first;firstly;for example;for instance;fortunately;frankly;frequently;further;furthermore;generally;gradually;happily;hence;here;historically;honestly;hopefully;however;ideally;immediately;importantly;in contrast;in fact;in other words;in particular;in short;in sum;in the end;in the meantime;in turn;incidentally;increasingly;indeed;inevitably;initially;instead;interestingly;ironically;lastly;lately;later;likewise;locally;luckily;maybe;meaning;meantime;meanwhile;moreover;mostly;namely;nationally;naturally;nevertheless;next;nonetheless;normally;notably;now;obviously;occasionally;oddly;often;on the contrary;on the other hand;once;only;optionally;or;originally;otherwise;overall;particularly;perhaps;personally;plus;preferably;presently;presumably;previously;probably;rather;realistically;really;recently;regardless;remarkably;sadly;second;secondly;separately;seriously;significantly;similarly;simultaneously;slowly;so;sometimes;soon;specifically;still;strangely;subsequently;suddenly;supposedly;surely;surprisingly;technically;thankfully;then;theoretically;thereafter;thereby;therefore;third;thirdly;this;though;thus;together;traditionally;truly;truthfully;typically;ultimately;undoubtedly;unfortunately;unsurprisingly;usually;well;yet)", "options"),
            ("{{ label }}", "label")
        ]
    },
    "emo_o": {
        "info": {
            "context": 1, "question": 0, "options": 4, "label": 1
        },
        "rules": [
            ("{{ text }}", "context"),
            ("L(happy;sad;angry;something else)", "options"),
            ("{{ label }}", "label")
        ]
    },
    "emotion_o": {
        "info": {
            "context": 1, "question": 0, "options": 6, "label": 1
        },
        "rules": [
            ("{{ text }}", "context"),
            ("L(sadness;joy;love;anger;fear;surprise)", "options"),
            ("{{ label }}", "label")
        ]
    },
    "freebase_qa_o": {
        "info": {
            "context": -1, "question": -1, "options": -1, "label": -1
        },
        "rules": []
    },
    "gigaword_o": {
        "info": {
            "context": 1, "question": 0, "options": 0, "label": 1
        },
        "rules": [
            ("{{ document }}", "context"),
            ("{{ summary }}", "answer")
        ]
    },
    "google_wellformed_query_o": {
        "info": {
            "context": 1, "question": 0, "options": 2, "label": 1
        },
        "rules": {
            ("{{ content }}", "context"),
            ("L(no;yes)", "options"),
            ("{% if 0.5 < rating %}yes{% else %}no{% endif %}", "answer")
        }
    },
    "hellaswag_o": {
        "info": {
            "context": 1, "question": 0, "options": 4, "label": 1
        },
        "rules": [
            ("{{ ctx_a }}", "context"),
            ("{{ ctx_b }}", "context"),
            ("{{ endings[0] }}", "options"),
            ("{{ endings[1] }}", "options"),
            ("{{ endings[2] }}", "options"),
            ("{{ endings[3] }}", "options"),
            ("{{ label | int }}", "label")
        ]
    },
    "imdb_o": {
        "info": {
            "context": 1, "question": 0, "options": 2, "label": 1
        },
        "rules": [
            ("{{ text }}", "context"),
            ("L(negative;positive)", "options"),
            ("{{ label }}", "label"),
        ]
    },
    "liar_o": {
        "info": {
            "context": 4, "question": -1, "options": 6, "label": 1
        },
        "rules": [
            ("{{ speaker }}", "context"),
            ("{{ context }}", "context"),
            ("{{ subject }}", "context"),
            ("{{ party_affiliation }}", "context"),
            ("L(false;half-true;mostly-true;true;barely-true;pants-fire)", "options"),
            ("{{ label }}", "label")
        ]
    },
    "paws_labeled_final_o": {
        "info": {
            "context": 2, "question": 0, "options": 2, "label": 1
        },
        "rules": [
            ("{{ sentence1 }}", "context"),
            ("{{ sentence2 }}", "context"),
            ("L(No;Yes)", "options"),
            ("{{ label }}", "label")
        ]
    },
    "piqa_o": {
        "info": {
            "context": 1, "question": 0, "options": 2, "label": 1
        },
        "rules": [
            ("{{ goal }}", "context"),
            ("{{ sol1 }}", "options"),
            ("{{ sol2 }}", "options"),
            ("{{ label }}", "label")
        ]
    },
    "quail_o": {
        "info": {
            "context": 1, "question": 1, "options": 4, "label": 1
        },
        "rules": [
            ("{{ context }}", "context"),
            ("{{ question }}", "question"),
            ("{{ answers[0] }}", "options"),
            ("{{ answers[1] }}", "options"),
            ("{{ answers[2] }}", "options"),
            ("{{ answers[3] }}", "options"),
            ("{{ correct_answer_id }}", "label")
        ]
    },
    "quoref_o": {
        "info": {
            "context": 1, "question": 1, "options": 0, "label": 1
        },
        "rules": [
            ("{{ context }}", "context"),
            ("{{ question }}", "question"),
            ("{{ answers.text | choice }}", "answer")
        ]
    },
    "ropes_o": {
        "info": {
            "context": 2, "question": 0, "options": 0, "label": 1
        },
        "rules": [
            ("{{ background }}", "context"),
            ("{{ situation }}", "context"),
            ("{{ question }}", "question"),
            ("{{ answers.text | choice }}", "answer")
        ]
    },
    "sciq_o": {
        "info": {
            "context": 1, "question": 1, "options": 4, "label": 1
        },
        "rules": [
            ("{{ support }}", "context"),
            ("{{ question }}", "question"),
            ("{{ distractor1 }}", "options"),
            ("{{ distractor2 }}", "options"),
            ("{{ distractor3 }}", "options"),
            ("{{ correct_answer }}", "options"),
            ("{{ correct_answer }}", "answer")
        ]
    },
    "scitail_o": {
        "info": {
            "context": 2, "question": 0, "options": 2, "label": 1
        },
        "rules": [
            ("{{ sentence1 }}", "context"),
            ("{{ sentence2 }}", "context"),
            ("L(entailment;neutral)", "options"),
            ("{{ gold_label }}", "answer")
        ]
    },
    "social_i_qa_o": {
        "info": {
            "context": 1, "question": 1, "options": 3, "label": 1
        },
        "rules": [
            ("{{ context }}", "context"),
            ("{{ question }}", "question"),
            ("{{ answerA }}", "options"),
            ("{{ answerB }}", "options"),
            ("{{ answerC }}", "options"),
            ("{{ label | int - 1 }}", "label")
        ]
    },
    "swag_o": {
        "info": {
            "context": 2, "question": 0, "options": 4, "label": 1
        },
        "rules": [
            ("{{ sent1 }}", "context"),
            ("{{ sent2 }}", "context"),
            ("{{ ending0 }}", "options"),
            ("{{ ending1 }}", "options"),
            ("{{ ending2 }}", "options"),
            ("{{ ending3 }}", "options"),
            ("{{ label }}", "label")
        ]
    },
    "tab_fact_o": {
        "info": {
            "context": -1, "question": -1, "options": -1, "label": -1
        },
        "rules": []
    },
    "wiki_qa_o": {
        "info": {
            "context": 1, "question": 1, "options": 2, "label": 1
        },
        "rules": [
            ("{{ answer }}", "context"),
            ("{{ question }}", "question"),
            ("L(No;Yes)", "options"),
            ("{{ label }}", "label")
        ]
    },
    "wiqa_o": {
        "info": {
            "context": 1, "question": 1, "options": 3, "label": 1
        },
        "rules": [
            ("{{ question_para_step | join(\" \") }}", "context"),
            ("{{ question_stem }}", "question"),
            ("L(more;less;no effect)", "options"),
            ("{{ answer_label }}", "answer")
        ]
    },
    "xsum_o": {
        "info": {
            "context": 1, "question": 0, "options": 0, "label": 1
        },
        "rules": [
            ("{{ document }}", "context"),
            ("{{ summary }}", "answer")
        ]
    },
    "yahoo_answers_topics_o": {
        "info": {
            "context": 2, "question": 0, "options": 10, "label": 1
        },
        "rules": [
            ("{{ question_title }}", "context"),
            ("{{ question_content }}", "context"),
            ("L(Society & Culture;Science & Mathematics;Health;Education & Reference;Computers & Internet;Sports;Business & Finance;Entertainment & Music;Family & Relationships;Politics & Government)", "options"),
            ("{{ topic }}", "label")
        ]
    },
    "yelp_polarity_o": {
        "info": {
            "context": 1, "question": 0, "options": 2, "label": 1
        },
        "rules": [
            ("{{ text }}", "context"),
            ("L(negative;positive)", "options"),
            ("{{ label }}", "label"),
        ]
    },
    "yelp_review_full_o": {
        "info": {
            "context": 1, "question": 0, "options": 5, "label": 1
        },
        "rules": [
            ("{{ text }}", "context"),
            ("L(1 star;2 stars;3 stars;4 stars;5 stars)", "options"),
            ("{{ label }}", "label"),
        ]
    },
    "cos_e": {
        "info": {
            "context": 1, "question": 1, "options": 5, "label": 1
        },
        "rules": [
            ("{{ context }}", "context"),
            ("{{ question }}", "question"),
            ("{{ choices[0] }}", "options"),
            ("{{ choices[1] }}", "options"),
            ("{{ choices[2] }}", "options"),
            ("{{ choices[3] }}", "options"),
            ("{{ choices[4] }}", "options"),
            ("{{ answer }}", "answer")
        ]
    },
    "dream":{
        "info": {
            "context": 1, "question": 1, "options": 3, "label": 1
        },
        "rules": [
            ("{{ dialogue | join('\n') }}", "context"),
            ("{{ question }}", "question"),
            ("{{ choice[0] }}", "options"),
            ("{{ choice[1] }}", "options"),
            ("{{ choice[2] }}", "options"),
            ("{{ answer }}", "answer")
        ]
    },
    "quartz":{
        "info": {
            "context": 1, "question": 1, "options": 2, "label": 1
        },
        "rules": [
            ("{{ para }}", "context"),
            ("{{ question }}", "question"),
            ("{{ choices.text[0] }}", "options"),
            ("{{ choices.text[1] }}", "options"),
            ("{{ choices.label.index(answerKey) }}", "label")
        ]
    },
    "qasc": {
        "info": {
            "context": 1, "question": 1, "options": 5, "label": 1
        },
        "rules": [
            ("{{ dialogue | join('\n') }}", "context"),
            ("{{ question }}", "question"),
            ("{{ choices.text[0] }}", "options"),
            ("{{ choices.text[1] }}", "options"),
            ("{{ choices.text[2] }}", "options"),
            ("{{ choices.text[3] }}", "options"),
            ("{{ choices.text[4] }}", "options"),
            ("{{ choices.text[5] }}", "options"),
            ("{{ choices.text[6] }}", "options"),
            ("{{ choices.text[7] }}", "options"),
            ("{{ choices.label.index(answerKey) }}", "label")
        ]
    },
    "adversarial_qa": {
        "info": {
            "context": 2, "question": 1, "options": 0, "label": 1
        },
        "rules": [
            ("{{ title }}", "context"),
            ("{{ context }}", "context"),
            ("{{ question }}", "question"),
            ("{{ answers.text | choice }}", "answer")
        ]
    },
}
