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
    "TRAIN": ["cos_e", "dream", "quail", "quartz", "social_i_qa", "wiqa", "cosmos_qa", "qasc", "quarel", "sciq", "wiki_hop", "adversarial_qa", 
            "quoref", "ropes", "duorc_self", "duorc_para", "hotpot_qa_distractor", "hotpot_qa_fullwiki", "wiki_qa", "common_gen", "wiki_bio", "samsum", "xsum", "mrpc",
            "paws_labeled_final", "qqp", "art", "circa", "freebase_qa", "google_wellformed_query", "hellaswag", "liar", "piqa", "scitail", "swag", "tab_fact", 
            "yahoo_answers_topics", "dbpedia_14"],
    "EVAL": ["sst2", "subj", "mr", "rte", "ag_news", "cb", "sst5"]
    
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
    # t0
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
    "dream": DatasetItem(
        name=["dream", None],
        data_dir=os.path.join(BASE_DATA_DIR, "dream/cache"),
        allowed_prompt_eval=["read_the_following_conversation_and_answer_the_question"]
    ),
    "quail": DatasetItem(
        name=["quail", None],
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
    "wiqa": DatasetItem(
        name=["wiqa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "wiqa/cache"),
        post_fn={"which_of_the_following_is_the_supposed_perturbation": lambda x:(x[0], x[1], 
            ['directly impacting a step of the process', 'indirectly impacting a step of the process', 'not impacting any step of the process'], x[3])}
    ),
    "cosmos_qa": DatasetItem(
        name=["cosmos_qa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "cosmos_qa/cache")
    ),
    "qasc": DatasetItem(
        name=["qasc", None],
        data_dir=os.path.join(BASE_DATA_DIR, "qasc/cache"),
        allowed_prompt_eval=["is_correct_1"]
    ),
    "quarel": DatasetItem(
        name=["quarel", None],
        data_dir=os.path.join(BASE_DATA_DIR, "quarel/cache"),
        allowed_prompt_eval=["choose_between"]
    ),
    "sciq": DatasetItem(
        name=["sciq", None],
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
    "ropes": DatasetItem(
        name=["ropes", None],
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
    "amazon_polarity": DatasetItem(
        name=["amazon_polarity", None],
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
    "rotten_tomatoes": DatasetItem(
        name=["rotten_tomatoes", None],
        data_dir=os.path.join(BASE_DATA_DIR, "rotten_tomatoes/cache")
    ),
    "yelp_polarity": DatasetItem(
        name=["yelp_polarity", None],
        data_dir=os.path.join(BASE_DATA_DIR, "yelp_polarity/cache"),
        post_fn={None: lambda x: (x[0], x[1], ["no.", "yes."], x[3])}
    ),
    "ag_news": DatasetItem(
        name=["ag_news", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ag_news/cache")
    ),
    "dbpedia_14": DatasetItem(
        name=["dbpedia_14", None],
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
    "mrpc": DatasetItem(
        name=["glue", "mrpc"],
        data_dir=os.path.join(BASE_DATA_DIR, "mrpc/cache")
    ),
    "paws_labeled_final": DatasetItem(
        name=["paws", "labeled_final"],
        data_dir=os.path.join(BASE_DATA_DIR, "paws/cache/labeled_final")
    ),
    "qqp": DatasetItem(
        name=["glue", "qqp"],
        data_dir=os.path.join(BASE_DATA_DIR, "qqp/cache")
    ),
    "hellaswag": DatasetItem(
        name=["hellaswag", None],
        data_dir=os.path.join(BASE_DATA_DIR, "hellaswag/cache"),
    ),
    "anli_r1": DatasetItem(
        name=["anli", None],
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
        option_id_space={"Easy Prompt": [0, 1], "Easy Prompt 2": [0, 1]}
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
    "art": DatasetItem(
        name=["art", None],
        data_dir=os.path.join(BASE_DATA_DIR, "art/cache"),
    ),
    "circa": DatasetItem(
        name=["circa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "circa_fix/cache"),
        banned_prompt_train=["possible_qn", "question_declarative"]
    ),
    "discovery": DatasetItem(
        name=["discovery", "discovery"],
        data_dir=os.path.join(BASE_DATA_DIR, "discovery/cache")
    ),
    "emo": DatasetItem(
        name=["emo", None],
        data_dir=os.path.join(BASE_DATA_DIR, "emo/cache")
    ),
    "emotion": DatasetItem(
        name=["emotion", None],
        data_dir=os.path.join(BASE_DATA_DIR, "emotion/cache")
    ),
    "freebase_qa": DatasetItem(
        name=["freebase_qa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "freebase_qa/cache")
    ), 
    "google_wellformed_query": DatasetItem(
        name=["google_wellformed_query", None],
        data_dir=os.path.join(BASE_DATA_DIR, "google_wellformed_query/cache")
    ),
    "liar": DatasetItem(
        name=["liar", None],
        data_dir=os.path.join(BASE_DATA_DIR, "liar/cache")
    ),
    "piqa": DatasetItem(
        name=["piqa", None],
        data_dir=os.path.join(BASE_DATA_DIR, "piqa/cache")
    ), 
    "scitail": DatasetItem(
        name=["scitail", "snli_format"],
        data_dir=os.path.join(BASE_DATA_DIR, "scitail/cache/snli_format")
    ),
    "swag": DatasetItem(
        name=["swag", "regular"],
        data_dir=os.path.join(BASE_DATA_DIR, "swag/cache/regular")
    ),
    "tab_fact": DatasetItem(
        name=["tab_fact", "tab_fact"],
        data_dir=os.path.join(BASE_DATA_DIR, "tab_fact/cache/tab_fact")
    ), 
    "yahoo_answers_topics": DatasetItem(
        name=["yahoo_answers_topics", None],
        data_dir=os.path.join(BASE_DATA_DIR, "yahoo_answers_topics/cache")
    ), 
    "yelp_review_full" :DatasetItem(
        name=["yelp_review_full", None],
        data_dir=os.path.join(BASE_DATA_DIR, "yelp_full/cache")
    ),
    # evaluation
    "sst2": DatasetItem(
        name=["glue", "sst2"],
        data_dir=os.path.join(BASE_DATA_DIR, "sst2/cache"),
        allowed_prompt_train=["Easy Prompt"],
        allowed_prompt_eval=["Easy Prompt"],
        finite_label_prompt=["Easy Prompt"],
        option_id_space={"Easy Prompt": [0, 1]}
    ),
    "sst5": DatasetItem(
        name=["sst5", None],
        data_dir=os.path.join(BASE_DATA_DIR, "sst5/cache/"),
        allowed_prompt_train=["Easy Prompt"],
        allowed_prompt_eval=["Easy Prompt"],
        finite_label_prompt=["Easy Prompt"],
        option_id_space={"Easy Prompt": [0, 1, 2, 3, 4]}
    ),
    "subj": DatasetItem(
        name=["subj", None],
        data_dir=os.path.join(BASE_DATA_DIR, "subj/cache/"),
        allowed_prompt_train=["Easy Prompt"],
        allowed_prompt_eval=["Easy Prompt"],
        finite_label_prompt=["Easy Prompt"],
        option_id_space={"Easy Prompt": [0, 1]}
    ),
    "ag_news": DatasetItem(
        name=["ag_news", None],
        data_dir=os.path.join(BASE_DATA_DIR, "ag_news/cache/"),
        allowed_prompt_train=["Easy Prompt"],
        allowed_prompt_eval=["Easy Prompt"],
        finite_label_prompt=["Easy Prompt"],
        option_id_space={"Easy Prompt": [0, 1, 2, 3]}
    ),
    "rte": DatasetItem(
        name=["super_glue", "rte"],
        data_dir=os.path.join(BASE_DATA_DIR, "rte/cache"),
        allowed_prompt_train=["Easy Prompt"],
        allowed_prompt_eval=["Easy Prompt"],
        finite_label_prompt=["Easy Prompt"],
        option_id_space={"Easy Prompt": [0, 1]}
    ),
    "cb": DatasetItem(
        name=["super_glue", "cb"],
        data_dir=os.path.join(BASE_DATA_DIR, "cb/cache"),
        allowed_prompt_train=["Easy Prompt"],
        allowed_prompt_eval=["Easy Prompt"],
        finite_label_prompt=["Easy Prompt"],
        option_id_space=None
    ),
    "mr": DatasetItem(
        name=["rotten_tomatoes", None],
        data_dir=os.path.join(BASE_DATA_DIR, "mr/cache"),
        allowed_prompt_train=["Easy Prompt"],
        allowed_prompt_eval=["Easy Prompt"],
        finite_label_prompt=["Easy Prompt"],
        option_id_space={"Easy Prompt": [0, 1]}
    ),
}