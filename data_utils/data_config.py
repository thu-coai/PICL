from dataclasses import dataclass, field
import os
from t5.evaluation import metrics

BASE_DATA_DIR = "data"

def string_to_float(preds, labels):
    return [float(p) for p in preds], [float(l) for l in labels]

@dataclass
class DatasetItem(object):
    name: str = None
    data_dir: str = None
    finite_label_prompt: list = field(default_factory=list)
    allowed_prompt_train: list = None
    allowed_prompt_eval: list = None
    banned_prompt_train: list = field(default_factory=list)
    banned_prompt_eval: list = field(default_factory=list)
    split: str = None
    post_fn: dict = field(default_factory=dict)
    answer_post_fn: dict = field(default_factory=dict)
    force_process: list = field(default_factory=list)
    force_process_demo: list = field(default_factory=list)
    train_shot: dict = None
    eval_shot: dict = None
    option_id_space: dict = field(default_factory=dict)
    task_type: str = "cls"
    
    def check_allowed(self, pn, split):
        pn = pn.strip()
        if split == "train":
            return (pn not in self.banned_prompt_train) and (self.allowed_prompt_train is None or pn in self.allowed_prompt_train)
        else:
            return (pn not in self.banned_prompt_eval) and (self.allowed_prompt_eval is None or pn in self.allowed_prompt_eval)

    def finit_label(self, pn):
        pn = pn.strip()
        return self.finite_label_prompt is None or pn in self.finite_label_prompt
    
    def is_force_process(self, pn):
        pn = pn.strip()
        return self.force_process is None or pn in self.force_process
    
    def is_force_process_demo(self, pn):
        pn = pn.strip()
        return self.force_process_demo is None or pn in self.force_process_demo
    
    def get_post_fn(self, pn):
        pn = pn.strip()
        if None in self.post_fn:
            return self.post_fn[None]
        elif pn in self.post_fn:
            return self.post_fn[pn]
        else:
            return lambda x: x
    
    def get_answer_post_fn(self, mn):
        if None in self.answer_post_fn:
            return self.answer_post_fn[None]
        elif mn in self.answer_post_fn:
            return self.answer_post_fn[mn]
        else:
            return lambda x: x
    
    def get_shot(self, pn, split):
        if split == "train":
            if self.train_shot is not None and pn in self.train_shot:
                return self.train_shot[pn]
        else:
            if self.eval_shot is not None and pn in self.eval_shot:
                return self.eval_shot[pn]
        return None


DATA_GROUP_CONFIG = {
    "MCQA": ["cos_e", "dream", "quail", "quartz", "social_i_qa", "wiqa", "cosmos_qa", "qasc", "quarel", "sciq", "wiki_hop"],
    "EXQA": ["adversarial_qa", "quoref", "ropes", "duorc_self", "duorc_para"],
    "CBQA": ["hotpot_qa_distractor", "hotpot_qa_fullwiki", "wiki_qa"],
    "SENT": ["yelp_polarity", "rotten_tomatoes", "imdb", "app_reviews", "amazon_polarity"],
    "TC": ["ag_news", "dbpedia_14", "trec"],
    "S2T": ["wiki_bio", "common_gen"],
    "SUM": ["xsum", "gigaword", "multi_news", "samsum", "cnn_dailymail"],
    "PARA": ["mrpc", "qqp", "paws_labeled_final"],
    "SC": ["copa", "hellaswag", "story_cloze_2016"],
    "NLI": ["rte", "cb", "anli_r1", "anli_r2", "anli_r3"],
    "COREF": ["wsc", "winorande_xl", "winogrande_debiased"],
    "WIC": ["wic"],
    "HR": ["ag_news_o", "amazon_polarity_o", "anli_r1_o", "art_o", "circa_o", "cosmos_qa_o", "dbpedia_14_o", "discovery_o", "emo_o", "emotion_o", "freebase_qa_o", "gigaword_o", "google_wellformed_query_o",
           "hellaswag_o", "imdb_o", "liar_o", "paws_labeled_final_o", "piqa_o", "quail_o", "quoref_o", "ropes_o", "sciq_o", "scitail_o", "social_i_qa_o", "swag_o", "tab_fact_o", "wiki_qa_o", "wiqa_o", "xsum_o",
           "yahoo_answers_topics_o", "yelp_polarity_o", "yelp_review_full_o"],
    "LR": ["ai2_arc_o", "climate_fever_o", "codah_o", "commonsense_qa_o", "dream_o", "financial_phrasebank_o", "medical_questions_pairs_o", "openbookqa_o", "poem_sentiment_o", "qasc_o", "quarel_o", "sick_o"],
    "EASY": ["sst2_eval", "sst5_eval", "subj_eval", "dbpedia_14_eval", "ag_news_eval", "trec_eval"]
}


T0_METRICS = {
    "BLEU": metrics.bleu,
    "ROUGE": metrics.rouge,
    "Span Squad": metrics.span_squad,
    "Squad": metrics.squad,
    "Trivia QA": metrics.trivia_qa,
    "Accuracy": metrics.accuracy,
    "Spearman Correlation": metrics.spearman_corrcoef,
    "Other": metrics.accuracy
}


DATA_CONFIG = {
    "cos_e": DatasetItem(
        name=["cos_e", "v1.11"],
        data_dir=os.path.join(BASE_DATA_DIR, "cos_e/cache/v1.11/")
    ),
    "commonsense_qa": DatasetItem(
        name=["commonsense_qa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "commonsense_qa/cache"),
        allowed_prompt_eval=["most_suitable_answer"],
        post_fn={"most_suitable_answer": lambda x: (x[0], x[1], x[3]["choices"]["text"], x[3])}
    ),
    "commonsense_qa_o": DatasetItem(
        name=["commonsense_qa_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "commonsense_qa/cache"),
        allowed_prompt_eval=["most_suitable"],
        post_fn={"most_suitable": lambda x: (x[0], x[1], x[3]["choices"]["text"], x[3])},
        option_id_space={"most_suitable": [0, 1, 2, 3, 4]}
    ),
    "dream": DatasetItem(
        name=["dream", None],
        data_dir=os.path.join(BASE_DATA_DIR, "dream/cache"),
        allowed_prompt_eval=["read_the_following_conversation_and_answer_the_question"]
    ),
    "dream_o": DatasetItem(
        name=["dream_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "dream/cache"),
        allowed_prompt_eval=["read_the_following_conversation_and_answer_the_question"],
        option_id_space={"read_the_following_conversation_and_answer_the_question": [0, 1, 2]}
    ),
    "quail": DatasetItem(
        name=["quail", None],
        data_dir=os.path.join(BASE_DATA_DIR, "quail/cache")
    ),
    "quail_o": DatasetItem(
        name=["quail_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "quail/cache")
    ),
    "quartz": DatasetItem(
        name=["quartz", None],
        data_dir=os.path.join(BASE_DATA_DIR, "quartz/cache")
    ),
    "social_i_qa": DatasetItem(
        name=["social_i_qa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "social_i_qa/cache")
    ),
    "social_i_qa_o": DatasetItem(
        name=["social_i_qa_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "social_i_qa/cache")
    ),
    "wiqa": DatasetItem(
        name=["wiqa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "wiqa/cache"),
        post_fn={"which_of_the_following_is_the_supposed_perturbation": lambda x:(x[0], x[1], 
            ['directly impacting a step of the process', 'indirectly impacting a step of the process', 'not impacting any step of the process'], x[3])}
    ),
    "wiqa_o": DatasetItem(
        name=["wiqa_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "wiqa/cache"),
        post_fn={"which_of_the_following_is_the_supposed_perturbation": lambda x:(x[0], x[1], 
            ['directly impacting a step of the process', 'indirectly impacting a step of the process', 'not impacting any step of the process'], x[3])}
    ),
    "cosmos_qa": DatasetItem(
        name=["cosmos_qa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "cosmos_qa/cache")
    ),
    "cosmos_qa_o": DatasetItem(
        name=["cosmos_qa_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "cosmos_qa/cache")
    ),
    "qasc": DatasetItem(
        name=["qasc", None],
        data_dir=os.path.join(BASE_DATA_DIR, "qasc/cache"),
        allowed_prompt_eval=["is_correct_1"]
    ),
    "qasc_o": DatasetItem(
        name=["qasc_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "qasc/cache"),
        allowed_prompt_eval=["is_correct_1"],
        finite_label_prompt=["is_correct_1"],
        option_id_space={"is_correct_1": [0, 1]}
    ),
    "qasc_o_1": DatasetItem(
        name=["qasc_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "qasc/cache"),
        allowed_prompt_eval=["qa_with_separated_facts_1"],
    ),
    "quarel": DatasetItem(
        name=["quarel", None],
        data_dir=os.path.join(BASE_DATA_DIR, "quarel/cache"),
        allowed_prompt_eval=["choose_between"]
    ),
    "quarel_o": DatasetItem(
        name=["quarel_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "quarel/cache"),
        allowed_prompt_eval=["choose_between"],
        option_id_space={"choose_between": [0, 1]}
    ),
    "sciq": DatasetItem(
        name=["sciq", None],
        data_dir=os.path.join(BASE_DATA_DIR, "sciq/cache")
    ),
    "sciq_o": DatasetItem(
        name=["sciq_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "sciq/cache")
    ),
    "wiki_hop": DatasetItem(
        name=["wiki_hop", "original"],
        data_dir=os.path.join(BASE_DATA_DIR, "wiki_hop/cache/"),
        split=["train", "validation"],
    ),
    "adversarial_qa": DatasetItem(
        name=["adversarial_qa", "adversarialQA"],
        data_dir=os.path.join(BASE_DATA_DIR, "adversarial_qa/cache/adversarialQA")
    ),
    "quoref": DatasetItem(
        name=["quoref", None],
        data_dir=os.path.join(BASE_DATA_DIR, "quoref/cache"),
        answer_post_fn={"Squad": lambda x: ([[l] for l in x[0]], x[1])}
    ),
    "quoref_o": DatasetItem(
        name=["quoref_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "quoref/cache"),
        answer_post_fn={"Squad": lambda x: ([[l] for l in x[0]], x[1])}
    ),
    "ropes": DatasetItem(
        name=["ropes", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ropes/cache"),
        answer_post_fn={"Squad": lambda x: ([[l] for l in x[0]], x[1])}
    ),
    "ropes_o": DatasetItem(
        name=["ropes_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ropes/cache"),
        answer_post_fn={"Squad": lambda x: ([[l] for l in x[0]], x[1])}
    ),
    "duorc_self": DatasetItem(
        name=["duorc", "SelfRC"],
        data_dir=os.path.join(BASE_DATA_DIR, "duorc/cache/SelfRC")
    ),
    "duorc_para": DatasetItem(
        name=["duorc", "ParaphraseRC"],
        data_dir=os.path.join(BASE_DATA_DIR, "duorc/cache/ParaphraseRC")
    ),
    "hotpot_qa_distractor": DatasetItem(
        name=["hotpot_qa", "distractor"],
        data_dir=os.path.join(BASE_DATA_DIR, "hotpot_qa/cache/distractor")
    ),
    "hotpot_qa_fullwiki": DatasetItem(
        name=["hotpot_qa", "fullwiki"],
        data_dir=os.path.join(BASE_DATA_DIR, "hotpot_qa/cache/fullwiki")
    ),
    "wiki_qa": DatasetItem(
        name=["wiki_qa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "wiki_qa/cache")
    ),
    "wiki_qa_o": DatasetItem(
        name=["wiki_qa_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "wiki_qa/cache")
    ),
    "amazon_polarity": DatasetItem(
        name=["amazon_polarity", None],
        data_dir=os.path.join(BASE_DATA_DIR, "amazon_polarity/cache"),
        # force_process=None
    ),
    "amazon_polarity_o": DatasetItem(
        name=["amazon_polarity_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "amazon_polarity/cache"),
        # force_process=None
    ),
    "app_reviews": DatasetItem(
        name=["app_reviews", None],
        data_dir=os.path.join(BASE_DATA_DIR, "app_reviews/cache"),
        banned_prompt_train=["convert_to_star_rating"]
    ),
    "imdb": DatasetItem(
        name=["imdb", None],
        data_dir=os.path.join(BASE_DATA_DIR, "imdb/cache"),
        post_fn={"Negation template for positive and negative": lambda x: (x[0], x[1], ["negative review.", "positive review."], x[3])}
    ),
    "imdb_o": DatasetItem(
        name=["imdb_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "imdb/cache"),
        post_fn={"Negation template for positive and negative": lambda x: (x[0], x[1], ["negative review.", "positive review."], x[3])}
    ),
    "rotten_tomatoes": DatasetItem(
        name=["rotten_tomatoes", None],
        data_dir=os.path.join(BASE_DATA_DIR, "rotten_tomatoes/cache")
    ),
    "yelp_polarity": DatasetItem(
        name=["yelp_polarity", None],
        data_dir=os.path.join(BASE_DATA_DIR, "yelp_polarity/cache"),
        post_fn={None: lambda x: (x[0], x[1], ["no.", "yes."], x[3])}
    ),
    "yelp_polarity_o": DatasetItem(
        name=["yelp_polarity_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "yelp_polarity/cache"),
        post_fn={None: lambda x: (x[0], x[1], ["no.", "yes."], x[3])}
    ),
    "ag_news": DatasetItem(
        name=["ag_news", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ag_news/cache")
    ),
    "ag_news_o": DatasetItem(
        name=["ag_news_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ag_news/cache")
    ),
    "dbpedia_14": DatasetItem(
        name=["dbpedia_14", None],
        data_dir=os.path.join(BASE_DATA_DIR, "dbpedia_14/cache")
    ),
    "dbpedia_14_o": DatasetItem(
        name=["dbpedia_14_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "dbpedia_14/cache")
    ),
    "trec": DatasetItem(
        name=["trec", None],
        data_dir=os.path.join(BASE_DATA_DIR, "trec/cache")
    ),
    "common_gen": DatasetItem(
        name=["common_gen", None],
        data_dir=os.path.join(BASE_DATA_DIR, "common_gen/cache")
    ),
    "wiki_bio": DatasetItem(
        name=["wiki_bio", None],
        data_dir=os.path.join(BASE_DATA_DIR, "wiki_bio/cache")
    ),
    "cnn_dailymail": DatasetItem(
        name=["cnn_dailymail", "3.0.0"],
        data_dir=os.path.join(BASE_DATA_DIR, "cnn_dailymail/cache/3.0.0")
    ),
    "gigaword": DatasetItem(
        name=["gigaword", None],
        data_dir=os.path.join(BASE_DATA_DIR, "gigaword/cache")
    ),
    "gigaword_o": DatasetItem(
        name=["gigaword_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "gigaword/cache")
    ),
    "multi_news": DatasetItem(
        name=["multi_news", None],
        data_dir=os.path.join(BASE_DATA_DIR, "multi_news/cache")
    ),
    "samsum": DatasetItem(
        name=["samsum", None],
        data_dir=os.path.join(BASE_DATA_DIR, "samsum/cache")
    ),
    "xsum": DatasetItem(
        name=["xsum", None],
        data_dir=os.path.join(BASE_DATA_DIR, "xsum/cache")
    ),
    "xsum_o": DatasetItem(
        name=["xsum_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "xsum/cache")
    ),
    "mrpc": DatasetItem(
        name=["glue", "mrpc"],
        data_dir=os.path.join(BASE_DATA_DIR, "mrpc/cache")
    ),
    "paws_labeled_final": DatasetItem(
        name=["paws", "labeled_final"],
        data_dir=os.path.join(BASE_DATA_DIR, "paws/cache/labeled_final")
    ),
    "paws_labeled_final_o": DatasetItem(
        name=["paws_origin", "labeled_final"],
        data_dir=os.path.join(BASE_DATA_DIR, "paws/cache/labeled_final")
    ),
    "qqp": DatasetItem(
        name=["glue", "qqp"],
        data_dir=os.path.join(BASE_DATA_DIR, "qqp/cache")
    ),
    "copa": DatasetItem(
        name=["super_glue", "copa"],
        data_dir=os.path.join(BASE_DATA_DIR, "copa/cache")
    ),
    "hellaswag": DatasetItem(
        name=["hellaswag", None],
        data_dir=os.path.join(BASE_DATA_DIR, "hellaswag/cache"),
        banned_prompt_eval=[
            "Open-ended completion",
            "Open-ended start",
            "Topic of the context",
            "Reversed appropriate continuation - Yes or No",
            "Appropriate continuation - Yes or No",
            "Topic without the ending answer"
        ]
    ),
    "hellaswag_o": DatasetItem(
        name=["hellaswag_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "hellaswag/cache"),
        banned_prompt_eval=[
            "Open-ended completion",
            "Open-ended start",
            "Topic of the context",
            "Reversed appropriate continuation - Yes or No",
            "Appropriate continuation - Yes or No",
            "Topic without the ending answer"
        ]
    ),
    "story_cloze_2016": DatasetItem(
        name=["story_cloze", "2016"],
        data_dir=os.path.join(BASE_DATA_DIR, "story_cloze/cache/2016"),
        split=["validation"],
        banned_prompt_eval=["Generate Ending"]
    ),
    "story_gen": DatasetItem(
        name=["story_cloze", "2016"],
        data_dir=os.path.join(BASE_DATA_DIR, "story_cloze/cache/2016"),
        split=["validation"]
    ),
    "squad_qg": DatasetItem(
        name=["squad", None],
        data_dir=os.path.join(BASE_DATA_DIR, "squad/cache/"),
        split=["validation"]
    ),
    "anli_r1": DatasetItem(
        name=["anli", None],
        data_dir=os.path.join(BASE_DATA_DIR, "anli/cache/r1")
    ),
    "anli_r1_o": DatasetItem(
        name=["anli_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "anli/cache/r1")
    ),
    "cb": DatasetItem(
        name=["super_glue", "cb"],
        data_dir=os.path.join(BASE_DATA_DIR, "cb/cache"),
    ),
    "rte": DatasetItem(
        name=["super_glue", "rte"],
        data_dir=os.path.join(BASE_DATA_DIR, "rte/cache"),
        finite_label_prompt=None
    ),
    "wsc": DatasetItem(
        name=["super_glue", "wsc.fixed"],
        data_dir=os.path.join(BASE_DATA_DIR, "wsc/cache"),
        finite_label_prompt=None
    ),
    "winogrande_xl": DatasetItem(
        name=["winogrande", "winogrande_xl"],
        data_dir=os.path.join(BASE_DATA_DIR, "winogrande/cache/winogrande_xl"),
        finite_label_prompt=None
    ),
    "winogrande_debiased": DatasetItem(
        name=["winogrande", "winogrande_debiased"],
        data_dir=os.path.join(BASE_DATA_DIR, "winogrande/cache/winogrande_debiased"),
        finite_label_prompt=None
    ),
    "wic": DatasetItem(
        name=["super_glue", "wic"],
        data_dir=os.path.join(BASE_DATA_DIR, "wic/cache"),
        finite_label_prompt=None
    ),
    "sst2": DatasetItem(
        name=["glue", "sst2"],
        data_dir=os.path.join(BASE_DATA_DIR, "sst2/cache"),
        finite_label_prompt=None,
        option_id_space={"Simple Prompt": [0, 1], "Simple Prompt 2": [0, 1]}
    ),
    "qnli": DatasetItem(
        name=["glue", "qnli"],
        data_dir=os.path.join(BASE_DATA_DIR, "qnli/cache")
    ),
    "wnli": DatasetItem(
        name=["glue", "wnli"],
        data_dir=os.path.join(BASE_DATA_DIR, "wnli/cache")
    ),
    "cola": DatasetItem(
        name=["glue", "cola"],
        data_dir=os.path.join(BASE_DATA_DIR, "cola/cache")
    ),
    # HR
    "art_o": DatasetItem(
        name=["art_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "art/cache"),
    ),
    "circa_o": DatasetItem(
        name=["circa_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "circa_fix/cache"),
        banned_prompt_train=["possible_qn", "question_declarative"]
    ),
    "discovery_o": DatasetItem(
        name=["discovery_origin", "discovery"],
        data_dir=os.path.join(BASE_DATA_DIR, "discovery/cache")
    ),
    "emo_o": DatasetItem(
        name=["emo_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "emo/cache")
    ),
    "emotion_o": DatasetItem(
        name=["emotion_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "emotion/cache")
    ),
    "freebase_qa_o": DatasetItem(
        name=["freebase_qa_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "freebase_qa/cache")
    ), 
    "google_wellformed_query_o": DatasetItem(
        name=["google_wellformed_query_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "google_wellformed_query/cache")
    ),
    "liar_o": DatasetItem(
        name=["liar_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "liar/cache")
    ),
    "piqa_o": DatasetItem(
        name=["piqa_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "piqa/cache")
    ), 
    "scitail_o": DatasetItem(
        name=["scitail_origin", "snli_format"],
        data_dir=os.path.join(BASE_DATA_DIR, "scitail/cache/snli_format")
    ),
    "swag_o": DatasetItem(
        name=["swag_origin", "regular"],
        data_dir=os.path.join(BASE_DATA_DIR, "swag/cache/regular")
    ),
    "tab_fact_o": DatasetItem(
        name=["tab_fact_origin", "tab_fact"],
        data_dir=os.path.join(BASE_DATA_DIR, "tab_fact/cache/tab_fact")
    ), 
    "yahoo_answers_topics_o": DatasetItem(
        name=["yahoo_answers_topics_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "yahoo_answers_topics/cache")
    ), 
    "yelp_review_full_o" :DatasetItem(
        name=["yelp_review_full_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "yelp_full/cache")
    ),
    # low
    "ai2_arc_o": DatasetItem(
        name=["ai2_arc_origin", "ARC-Challenge"],
        data_dir=os.path.join(BASE_DATA_DIR, "ai2_arc/cache/ARC-Challenge"),
        allowed_prompt_eval=["multiple_choice"],
        option_id_space={"multiple_choice":[0,1,2,3]}
    ),
    "climate_fever_o": DatasetItem(
        name=["climate_fever_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "climate_fever/cache"),
        allowed_prompt_eval=["claim_and_all_supporting_evidences"],
        finite_label_prompt=["claim_and_all_supporting_evidences"],
        option_id_space={"claim_and_all_supporting_evidences": [0, 1, 2, 3]}
    ), 
    "codah_o": DatasetItem(
        name=["codah_origin", "fold_0"],
        data_dir=os.path.join(BASE_DATA_DIR, "codah/cache/fold_0"),
        allowed_prompt_eval=["answer_with_option"],
        post_fn={"answer_with_option": lambda x: [x[0], x[1], [o.strip() for o in x[3]["candidate_answers"]], x[3]]},
        option_id_space={"answer_with_option": [0, 1, 2, 3]}
    ),
    "financial_phrasebank_o": DatasetItem(
        name=["financial_phrasebank_origin", "sentences_allagree"],
        data_dir=os.path.join(BASE_DATA_DIR, "financial_phrasebank/cache/sentences_allagree"),
        allowed_prompt_eval=["bull_bear"],
        finite_label_prompt=["bull_bear"],
        option_id_space={"bull_bear": [0, 1, 2]}
    ), 
    "medical_questions_pairs_o": DatasetItem(
        name=["medical_questions_pairs_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "medical_questions_pairs/cache"),
        allowed_prompt_eval=["basic_v1"],
        finite_label_prompt=["basic_v1"],
        option_id_space={"basic_v1": [0, 1]}
    ),
    "openbookqa_o": DatasetItem(
        name=["openbookqa_origin", "main"],
        data_dir=os.path.join(BASE_DATA_DIR, "openbookqa/cache/main"),
        allowed_prompt_eval=["choices"],
        option_id_space={"choices": [0, 1, 2, 3]}
    ),
    "poem_sentiment_o": DatasetItem(
        name=["poem_sentiment_origin", None],
        data_dir=os.path.join(BASE_DATA_DIR, "poem_sentiment/cache/"),
        allowed_prompt_eval=["poem_sentiment_1"],
        finite_label_prompt=["poem_sentiment_1"],
        option_id_space={"poem_sentiment_1": [0, 1, 2]}
    ), 
    "sick_o": DatasetItem(
        name=["sick_origin", None],        
        data_dir=os.path.join(BASE_DATA_DIR, "sick/cache/"),
        allowed_prompt_eval=["entailed"],
        post_fn={"entailed": lambda x: (x[0], x[1], ["entailment", "neutral", "contradiction"], x[3])},
        finite_label_prompt=["entailed"],
        option_id_space={"entailed": [0, 1, 2]}
    ),
    "sst2_eval": DatasetItem(
        name=["glue", "sst2"],
        data_dir=os.path.join(BASE_DATA_DIR, "sst2/cache"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space={"MSRA Prompt": [0, 1]}
    ),
    "sst2_eval_all": DatasetItem(
        name=["glue", "sst2"],
        data_dir=os.path.join(BASE_DATA_DIR, "sst2/cache"),
        banned_prompt_train=["Minimal Prompt", "Simple Prompt", "Simple Prompt 2", "MSRA Prompt", "It is"],
        banned_prompt_eval=["Minimal Prompt", "Simple Prompt", "Simple Prompt 2", "MSRA Prompt", "It is"],
        finite_label_prompt=None,
        option_id_space={"ALL": [0, 1]}
    ),
    "sst5_eval": DatasetItem(
        name=["sst5", None],
        data_dir=os.path.join(BASE_DATA_DIR, "sst5/cache/"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space={"MSRA Prompt": [0, 1, 2, 3, 4]}
    ),
    "sst5_eval_2": DatasetItem(
        name=["sst5", None],
        data_dir=os.path.join(BASE_DATA_DIR, "sst5/cache/"),
        allowed_prompt_train=["MSRA Prompt 2"],
        allowed_prompt_eval=["MSRA Prompt 2"],
        finite_label_prompt=["MSRA Prompt 2"],
        option_id_space={"MSRA Prompt 2": [0, 1, 2, 3, 4]}
    ),
    "subj_eval": DatasetItem(
        name=["subj", None],
        data_dir=os.path.join(BASE_DATA_DIR, "subj/cache/"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space={"MSRA Prompt": [0, 1]}
    ),
    "subj_eval_2": DatasetItem(
        name=["subj", None],
        data_dir=os.path.join(BASE_DATA_DIR, "subj/cache/"),
        allowed_prompt_train=["MSRA Prompt 2"],
        allowed_prompt_eval=["MSRA Prompt 2"],
        finite_label_prompt=["MSRA Prompt 2"],
        option_id_space={"MSRA Prompt 2": [0, 1]}
    ),
    "subj_eval_3": DatasetItem(
        name=["subj", None],
        data_dir=os.path.join(BASE_DATA_DIR, "subj/cache/"),
        allowed_prompt_train=["MSRA Prompt 3"],
        allowed_prompt_eval=["MSRA Prompt 3"],
        finite_label_prompt=["MSRA Prompt 3"],
        option_id_space={"MSRA Prompt 3": [0, 1]}
    ),
    "subj_eval_4": DatasetItem(
        name=["subj", None],
        data_dir=os.path.join(BASE_DATA_DIR, "subj/cache/"),
        allowed_prompt_train=["MSRA Prompt 4"],
        allowed_prompt_eval=["MSRA Prompt 4"],
        finite_label_prompt=["MSRA Prompt 4"],
        option_id_space={"MSRA Prompt 4": [0, 1]}
    ),
    "subj_eval_5": DatasetItem(
        name=["subj", None],
        data_dir=os.path.join(BASE_DATA_DIR, "subj/cache/"),
        allowed_prompt_train=["MSRA Prompt 5"],
        allowed_prompt_eval=["MSRA Prompt 5"],
        finite_label_prompt=["MSRA Prompt 5"],
        option_id_space={"MSRA Prompt 5": [0, 1]}
    ),
    "trec_eval": DatasetItem(
        name=["trec", None],
        data_dir=os.path.join(BASE_DATA_DIR, "trec/cache/"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space=None
    ),
    "trec_eval_2": DatasetItem(
        name=["trec", None],
        data_dir=os.path.join(BASE_DATA_DIR, "trec/cache/"),
        allowed_prompt_train=["MSRA Prompt 2"],
        allowed_prompt_eval=["MSRA Prompt 2"],
        finite_label_prompt=["MSRA Prompt 2"],
        option_id_space=None
    ),
    "dbpedia_14_eval": DatasetItem(
        name=["dbpedia_14", None],
        data_dir=os.path.join(BASE_DATA_DIR, "dbpedia_14/cache/"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space={"MSRA Prompt": list(range(14))}
    ),
    "ag_news_eval": DatasetItem(
        name=["ag_news", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ag_news/cache/"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space={"MSRA Prompt": [0, 1, 2, 3]}
    ),
    "ag_news_eval_2": DatasetItem(
        name=["ag_news", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ag_news/cache/"),
        allowed_prompt_train=["MSRA Prompt 2"],
        allowed_prompt_eval=["MSRA Prompt 2"],
        finite_label_prompt=["MSRA Prompt 2"],
        option_id_space={"MSRA Prompt 2": [0, 1, 2, 3]}
    ),
    "ag_news_eval_3": DatasetItem(
        name=["ag_news", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ag_news/cache/"),
        allowed_prompt_train=["MSRA Prompt 3"],
        allowed_prompt_eval=["MSRA Prompt 3"],
        finite_label_prompt=["MSRA Prompt 3"],
        option_id_space={"MSRA Prompt 3": [0, 1, 2, 3]}
    ),
    "ag_news_eval_4": DatasetItem(
        name=["ag_news", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ag_news/cache/"),
        allowed_prompt_train=["MSRA Prompt 4"],
        allowed_prompt_eval=["MSRA Prompt 4"],
        finite_label_prompt=["MSRA Prompt 4"],
        option_id_space={"MSRA Prompt 4": [0, 1, 2, 3]}
    ),
    "ag_news_eval_5": DatasetItem(
        name=["ag_news", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ag_news/cache/"),
        allowed_prompt_train=["MSRA Prompt 5"],
        allowed_prompt_eval=["MSRA Prompt 5"],
        finite_label_prompt=["MSRA Prompt 5"],
        option_id_space={"MSRA Prompt 5": [0, 1, 2, 3]}
    ),
    "ag_news_eval_6": DatasetItem(
        name=["ag_news", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ag_news/cache/"),
        allowed_prompt_train=["MSRA Prompt 6"],
        allowed_prompt_eval=["MSRA Prompt 6"],
        finite_label_prompt=["MSRA Prompt 6"],
        option_id_space={"MSRA Prompt 6": [0, 1, 2, 3]}
    ),
    "ag_news_eval_all": DatasetItem(
        name=["ag_news", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ag_news/cache/"),
        banned_prompt_train=["MSRA Prompt", "MSRA Prompt 2", "MSRA Prompt 3", "MSRA Prompt 4", "MSRA Prompt 5", "MSRA Prompt 6"],
        banned_prompt_eval=["MSRA Prompt", "MSRA Prompt 2", "MSRA Prompt 3", "MSRA Prompt 4", "MSRA Prompt 5", "MSRA Prompt 6"],
        finite_label_prompt=None,
        option_id_space={"ALL": [0, 1, 2, 3]}
    ),
    "rte_eval": DatasetItem(
        name=["super_glue", "rte"],
        data_dir=os.path.join(BASE_DATA_DIR, "rte/cache"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space={"MSRA Prompt": [0, 1]}
    ),
    "rte_eval_all": DatasetItem(
        name=["super_glue", "rte"],
        data_dir=os.path.join(BASE_DATA_DIR, "rte/cache"),
        banned_prompt_train=["MSRA Prompt", "MSRA Prompt 2", "Simple Prompt"],
        banned_prompt_eval=["MSRA Prompt", "MSRA Prompt 2", "Simple Prompt"],
        finite_label_prompt=None,
        option_id_space={"ALL": [0, 1]}
    ),
    "cb_eval": DatasetItem(
        name=["super_glue", "cb"],
        data_dir=os.path.join(BASE_DATA_DIR, "cb/cache"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space=None
    ),
    "cb_eval_all": DatasetItem(
        name=["super_glue", "cb"],
        data_dir=os.path.join(BASE_DATA_DIR, "cb/cache"),
        banned_prompt_train=["MSRA Prompt", "MSRA Prompt 2", "MSRA Prompt 3", "Simple Prompt", "Simple Prompt 2"],
        banned_prompt_eval=["MSRA Prompt", "MSRA Prompt 2", "MSRA Prompt 3", "Simple Prompt", "Simple Prompt 2"],
        finite_label_prompt=None,
        option_id_space=None
    ),
    "mr_eval": DatasetItem(
        name=["rotten_tomatoes", None],
        data_dir=os.path.join(BASE_DATA_DIR, "mr/cache"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space={"MSRA Prompt": [0, 1]}
    ),
    "mr_eval_all": DatasetItem(
        name=["rotten_tomatoes", None],
        data_dir=os.path.join(BASE_DATA_DIR, "mr/cache"),
        banned_prompt_train=["MSRA Prompt", "MSRA Prompt 2"],
        banned_prompt_eval=["MSRA Prompt", "MSRA Prompt 2"],
        finite_label_prompt=None,
        option_id_space={"ALL": [0, 1]}
    ),
    "boolq_eval": DatasetItem(
        name=["super_glue", "boolq"],
        data_dir=os.path.join(BASE_DATA_DIR, "boolq/cache"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space={"MSRA Prompt": [0, 1]}
    ),
    "rte_eval_2": DatasetItem(
        name=["super_glue", "rte"],
        data_dir=os.path.join(BASE_DATA_DIR, "rte/cache"),
        allowed_prompt_train=["MSRA Prompt 2"],
        allowed_prompt_eval=["MSRA Prompt 2"],
        finite_label_prompt=["MSRA Prompt 2"],
        option_id_space={"MSRA Prompt 2": [0, 1]}
    ),
    "cb_eval_2": DatasetItem(
        name=["super_glue", "cb"],
        data_dir=os.path.join(BASE_DATA_DIR, "cb/cache"),
        allowed_prompt_train=["MSRA Prompt 2"],
        allowed_prompt_eval=["MSRA Prompt 2"],
        finite_label_prompt=["MSRA Prompt 2"],
        option_id_space=None
    ),
    "cb_eval_3": DatasetItem(
        name=["super_glue", "cb"],
        data_dir=os.path.join(BASE_DATA_DIR, "cb/cache"),
        allowed_prompt_train=["MSRA Prompt 3"],
        allowed_prompt_eval=["MSRA Prompt 3"],
        finite_label_prompt=["MSRA Prompt 3"],
        option_id_space=None
    ),
    "mr_eval_2": DatasetItem(
        name=["rotten_tomatoes", None],
        data_dir=os.path.join(BASE_DATA_DIR, "mr/cache"),
        allowed_prompt_train=["MSRA Prompt 2"],
        allowed_prompt_eval=["MSRA Prompt 2"],
        finite_label_prompt=["MSRA Prompt 2"],
        option_id_space={"MSRA Prompt 2": [0, 1]}
    ),
    "boolq_eval_2": DatasetItem(
        name=["super_glue", "boolq"],
        data_dir=os.path.join(BASE_DATA_DIR, "boolq/cache"),
        allowed_prompt_train=["MSRA Prompt 2"],
        allowed_prompt_eval=["MSRA Prompt 2"],
        finite_label_prompt=["MSRA Prompt 2"],
        option_id_space={"MSRA Prompt 2": [0, 1]}
    ),
    "boolq_eval_3": DatasetItem(
        name=["super_glue", "boolq"],
        data_dir=os.path.join(BASE_DATA_DIR, "boolq/cache"),
        allowed_prompt_train=["MSRA Prompt 3"],
        allowed_prompt_eval=["MSRA Prompt 3"],
        finite_label_prompt=["MSRA Prompt 3"],
        option_id_space={"MSRA Prompt 3": [0, 1]}
    ),
    "boolq_eval_4": DatasetItem(
        name=["super_glue", "boolq"],
        data_dir=os.path.join(BASE_DATA_DIR, "boolq/cache"),
        allowed_prompt_train=["MSRA Prompt 4"],
        allowed_prompt_eval=["MSRA Prompt 4"],
        finite_label_prompt=["MSRA Prompt 4"],
        option_id_space={"MSRA Prompt 4": [0, 1]}
    ),
    "qqp_eval": DatasetItem(
        name=["glue", "qqp"],
        data_dir=os.path.join(BASE_DATA_DIR, "qqp/cache"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space={"MSRA Prompt": [0, 1]}
    ),
    "mrpc_eval": DatasetItem(
        name=["glue", "mrpc"],
        data_dir=os.path.join(BASE_DATA_DIR, "mrpc/cache"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space={"MSRA Prompt": [0, 1]}
    ),
    "wic_eval": DatasetItem(
        name=["super_glue", "wic"],
        data_dir=os.path.join(BASE_DATA_DIR, "wic/cache"),
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        finite_label_prompt=["MSRA Prompt"],
        option_id_space={"MSRA Prompt": [0, 1]}
    ),
    "story_gen_eval": DatasetItem(
        name=["roc_story", None],
        data_dir=os.path.join(BASE_DATA_DIR, "roc_story/cache/"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        task_type="gen",
        option_id_space=None
    ),
    "story_gen_eval_2": DatasetItem(
        name=["roc_story", None],
        data_dir=os.path.join(BASE_DATA_DIR, "roc_story/cache/"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt 2"],
        allowed_prompt_eval=["MSRA Prompt 2"],
        task_type="gen",
        option_id_space=None
    ),
    "gigaword_eval": DatasetItem(
        name=["gigaword", None],
        data_dir=os.path.join(BASE_DATA_DIR, "gigaword/cache/"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        task_type="gen",
        option_id_space=None
    ),
    "squad_qg_eval": DatasetItem(
        name=["squad", None],
        data_dir=os.path.join(BASE_DATA_DIR, "squad/cache/"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt (Gen)"],
        allowed_prompt_eval=["MSRA Prompt (Gen)"],
        task_type="gen",
        option_id_space=None
    ),
    "squad_eval": DatasetItem(
        name=["squad", None],
        data_dir=os.path.join(BASE_DATA_DIR, "squad/cache/"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        task_type="gen",
        option_id_space=None
    ),
    "openbookqa_eval": DatasetItem(
        name=["openbookqa", "main"],
        data_dir=os.path.join(BASE_DATA_DIR, "openbookqa/cache/main"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        option_id_space=None
        # option_id_space={"MSRA Prompt": [0, 1, 2, 3]}
    ),
    "commonsense_qa_eval": DatasetItem(
        name=["commonsense_qa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "commonsense_qa/cache/"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        option_id_space=None
        # option_id_space={"MSRA Prompt": [0, 1, 2, 3, 4]}
    ),
    "copa_eval": DatasetItem(
        name=["super_glue", "copa"],
        data_dir=os.path.join(BASE_DATA_DIR, "copa/cache/"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        option_id_space=None
        # option_id_space={"MSRA Prompt": [0, 1]}
    ),
    "arc_easy_eval": DatasetItem(
        name=["ai2_arc", "ARC-Easy"],
        data_dir=os.path.join(BASE_DATA_DIR, "ai2_arc/cache/ARC-Easy/"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        option_id_space=None
        # option_id_space={"MSRA Prompt": [0, 1]}
    ),
    "arc_hard_eval": DatasetItem(
        name=["ai2_arc", "ARC-Challenge"],
        data_dir=os.path.join(BASE_DATA_DIR, "ai2_arc/cache/ARC-Challenge/"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        option_id_space=None
        # option_id_space={"MSRA Prompt": [0, 1]}
    ),
    "hellaswag_eval": DatasetItem(
        name=["hellaswag", None],
        data_dir=os.path.join(BASE_DATA_DIR, "hellaswag/cache"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        option_id_space=None
    ),
    "piqa_eval": DatasetItem(
        name=["piqa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "piqa/cache"),
        split=["validation"],
        allowed_prompt_train=["MSRA Prompt"],
        allowed_prompt_eval=["MSRA Prompt"],
        option_id_space=None
    ),
}