"""Converters from public Hugging Face datasets to Leo decision requests.

None of the held-out benchmark datasets (dair-ai/emotion, tweet_topic, twitter-financial-news-topic,
daily_dialog) or any emotion-labelled dataset is used here, so emotion recognition stays a held-out
task family and the tweet / financial-news domains stay unseen.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from leo.data.augment import (
    about_noul, choice_question, humanize, noul_question, qids, refer, score_question, truncate, wrap_state,
)

Row = dict[str, Any]
Convert = Callable[[Row, list[str], random.Random], dict[str, Any] | None]


def _request(source: str, state: Any, questions: list[dict[str, Any]], rng: random.Random) -> dict[str, Any] | None:
    questions = [q for q in questions if q is not None]
    if not questions:
        return None
    return {"source": source, "state": state, "questions": dict(zip(qids(len(questions), rng), questions))}


def _classify(source: str, text: str, gold: str, names: list[str], rng: random.Random, instrs: list[str],
              descs: dict[str, str] | None = None, subject: str = "text", max_options: int | None = None,
              fields: tuple[str, ...] = ("text", "message", "content"), p_noul: float = 0.5) -> dict[str, Any] | None:
    state, field = wrap_state(text, rng, fields)
    qs: list[dict[str, Any] | None] = []
    if rng.random() < 0.9:
        qs.append(choice_question(refer(field, rng.choice(instrs), rng), names, gold, rng, descs, max_options))
    for _ in range(2):
        if rng.random() < p_noul:
            label = gold if rng.random() < 0.5 else rng.choice(names)
            phrase = (descs or {}).get(label) if descs and rng.random() < 0.5 else humanize(label)
            qs.append(about_noul(phrase, label == gold, rng, subject))
    return _request(source, state, qs, rng)


# ------------------------------------------------------------------------------ topic

AG_DESC = {"World": "world news, politics and international affairs", "Sports": "sports news and results",
           "Business": "business, the economy, companies and markets", "Sci/Tech": "science and technology"}


def ag_news(row: Row, names: list[str], rng: random.Random):
    return _classify("ag_news", row["text"], names[row["label"]], names, rng, [
        "What is the topic of this news article?", "Which newspaper section would this story run in?",
        "Classify the topic of the article.", "Which category best describes this news story?"],
        AG_DESC, subject="article", fields=("article", "text", "headline_and_lead"))


def dbpedia(row: Row, names: list[str], rng: random.Random):
    text = f"{row['title']}. {row['content'].strip()}"
    return _classify("dbpedia_14", truncate(text, 1200), names[row["label"]], names, rng, [
        "What kind of entity does this encyclopedia entry describe?", "Which category does this entry belong to?",
        "What type of thing is the subject of this text?"], subject="entry", max_options=14, fields=("entry", "abstract", "text"))


def yahoo(row: Row, names: list[str], rng: random.Random):
    text = row["question_title"].strip()
    if row.get("question_content") and rng.random() < 0.7:
        text += "\n" + row["question_content"].strip()
    if row.get("best_answer") and rng.random() < 0.3:
        text += "\nAnswer: " + row["best_answer"].strip()
    return _classify("yahoo_answers", truncate(text, 1200), names[row["topic"]], names, rng, [
        "Which category does this question belong to?", "What is this question about?",
        "Which forum section should this post go in?"], subject="question", fields=("question", "post", "text"))


# ------------------------------------------------------------------------------ intent

def banking77(row: Row, names: list[str], rng: random.Random):
    max_opts = 77 if rng.random() < 0.1 else 30
    return _classify("banking77", row["text"], names[row["label"]], names, rng, [
        "What is the customer's intent?", "Which banking request is this?", "What does the customer want to do?",
        "Route this query to the matching intent."], subject="message", max_options=max_opts,
        fields=("message", "customer_message", "query"), p_noul=0.35)


def clinc(row: Row, names: list[str], rng: random.Random):
    gold = names[row["intent"]]
    instr = rng.choice(["What does the user want?", "Which intent does this request express?",
                        "Classify the user's request.", "What is the user asking the assistant to do?"])
    state, field = wrap_state(row["text"], rng, ("utterance", "request", "text"))
    in_scope = [n for n in names if n != "oos"]
    if gold == "oos":
        # Out-of-scope request: offer real intents plus a "none" option, which is the right answer.
        opts = rng.sample(in_scope, rng.randint(4, 25))
        q = choice_question(refer(field, instr, rng), opts, opts[0], rng, p_none=0.0, extra_none_option=True)
        q["label"] = list(q["criteria"])[-1]
    else:
        q = choice_question(refer(field, instr, rng), in_scope, gold, rng, max_options=30, p_none=0.1)
    return _request("clinc_oos", state, [q], rng)


# ------------------------------------------------------------------------------ inference

NLI_KEYS = [
    ({"entailment": "the hypothesis must be true if the premise is true",
      "neutral": "the hypothesis might or might not be true",
      "contradiction": "the hypothesis cannot be true if the premise is true"}, "premise", "hypothesis"),
    ({"supported": "the evidence shows the claim is true", "not enough information": "the evidence neither confirms nor rules out the claim",
      "contradicted": "the evidence shows the claim is false"}, "evidence", "claim"),
    ({"follows": None, "unknown": "cannot be decided from the text", "contradicts": None}, "text", "statement"),
]


def nli(source: str):
    def convert(row: Row, names: list[str], rng: random.Random):
        if row["label"] < 0:
            return None
        crit, a, b = rng.choice(NLI_KEYS)
        p, h = row["premise"].strip(), row["hypothesis"].strip()
        if rng.random() < 0.6:
            state = {a: p, b: h}
            ref_a, ref_b = f"`{a}`", f"`{b}`"
        else:
            state = f"{a.capitalize()}: {p}\n{b.capitalize()}: {h}"
            ref_a, ref_b = f"the {a}", f"the {b}"
        keys = list(crit)
        gold_key = keys[row["label"]]
        qs: list[dict[str, Any] | None] = []
        if rng.random() < 0.75:
            instr = rng.choice([f"How does {ref_b} relate to {ref_a}?", f"Given {ref_a}, is {ref_b} true?",
                                f"Does {ref_a} support {ref_b}?"])
            q = {"type": "choice", "instructions": instr, "criteria": dict(crit) if rng.random() < 0.7 else {k: None for k in keys},
                 "label": gold_key}
            qs.append(q)
        if rng.random() < 0.5:
            qs.append(noul_question(rng.choice([f"Does {ref_a} imply {ref_b}?", f"{ref_b.capitalize()} is true given {ref_a}."]),
                                    row["label"] == 0))
        if rng.random() < 0.3:
            qs.append(noul_question(f"Does {ref_a} contradict {ref_b}?", row["label"] == 2))
        return _request(source, state, qs, rng)

    return convert


def mrpc(row: Row, names: list[str], rng: random.Random):
    a, b = rng.choice([("sentence1", "sentence2"), ("a", "b"), ("original", "rewrite")])
    state = {a: row["sentence1"], b: row["sentence2"]}
    same = row["label"] == 1
    qs = [noul_question(rng.choice([f"Do `{a}` and `{b}` mean the same thing?", f"`{b}` is a paraphrase of `{a}`."]), same)]
    if rng.random() < 0.4:
        qs.append({"type": "choice", "instructions": f"Is `{b}` equivalent in meaning to `{a}`?",
                   "criteria": {"equivalent": "same meaning", "different": "the meaning changed"},
                   "label": "equivalent" if same else "different"})
    return _request("mrpc", state, qs, rng)


# ------------------------------------------------------------------------------ reading / knowledge

def boolq(row: Row, names: list[str], rng: random.Random):
    q = row["question"].strip()
    q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
    state, field = wrap_state(truncate(row["passage"], 1500), rng, ("passage", "document", "context"), p_raw=0.5)
    return _request("boolq", state, [noul_question(refer(field, q, rng), bool(row["answer"]))], rng)


def mcq(source: str):
    def convert(row: Row, names: list[str], rng: random.Random):
        texts, labels = row["choices"]["text"], row["choices"]["label"]
        if row["answerKey"] not in labels or len(set(texts)) != len(texts):
            return None
        gold = labels.index(row["answerKey"])
        state, field = wrap_state(row["question"].strip(), rng, ("question",), p_raw=0.6)
        instr = refer(field, rng.choice(["Which answer is correct?", "Pick the correct answer to the question.",
                                         "Which option best answers the question?"]), rng)
        if rng.random() < 0.5 or max(len(t) for t in texts) > 80:
            keys = labels if rng.random() < 0.7 else [str(i + 1) for i in range(len(texts))]
            crit = dict(zip(keys, texts))
            label = keys[gold]
        else:
            crit = {t: None for t in texts}
            label = texts[gold]
        order = list(crit.items())
        qs: list[dict[str, Any] | None] = [{"type": "choice", "instructions": instr, "criteria": dict(order), "label": label}]
        if rng.random() < 0.3:
            j = rng.randrange(len(texts))
            qs.append(noul_question(f"Is \"{texts[j]}\" the correct answer to the question?", j == gold))
        return _request(source, state, qs, rng)

    return convert


# ------------------------------------------------------------------------------ sentiment / ordinal

def imdb(row: Row, names: list[str], rng: random.Random):
    pos = row["label"] == 1
    state, field = wrap_state(truncate(row["text"].replace("<br />", " "), 1500), rng, ("review",), p_raw=0.6)
    qs: list[dict[str, Any] | None] = []
    r = rng.random()
    if r < 0.5:
        qs.append(noul_question(refer(field, rng.choice(["Is this review positive?", "The reviewer liked the movie.",
                                                          "Does the reviewer recommend the film?"]), rng), pos))
    else:
        keys = rng.choice([("positive", "negative"), ("liked it", "disliked it"), ("favourable", "unfavourable")])
        qs.append({"type": "choice", "instructions": refer(field, "What is the overall sentiment of the review?", rng),
                   "criteria": {keys[0]: None, keys[1]: None}, "label": keys[0] if pos else keys[1]})
    if rng.random() < 0.2:
        qs.append(noul_question("Is this review negative?", not pos))
    return _request("imdb", state, qs, rng)


YELP_LEVELS = [
    ["very negative, a terrible experience", "negative, mostly disappointed", "mixed or neutral",
     "positive, mostly satisfied", "very positive, an excellent experience"],
    ["hated it", "disliked it", "it was okay", "liked it", "loved it"],
    ["would warn others to stay away", "would not go back", "might go back", "would go back", "would strongly recommend it"],
]


def yelp(row: Row, names: list[str], rng: random.Random):
    y = row["label"]
    state, field = wrap_state(truncate(row["text"], 1500), rng, ("review", "text"), p_raw=0.6)
    r = rng.random()
    qs: list[dict[str, Any] | None] = []
    instr = refer(field, rng.choice(["How satisfied is the reviewer?", "Rate the reviewer's overall opinion.",
                                     "How positive is this review?"]), rng)
    if r < 0.55:
        qs.append(score_question(instr, rng.choice(YELP_LEVELS), y))
    elif r < 0.75:
        qs.append(score_question(instr, ["negative", "mixed", "positive"], [0, 0, 1, 2, 2][y]))
    else:
        keys = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
        qs.append({"type": "choice", "instructions": refer(field, "How many stars did the reviewer give?", rng),
                   "criteria": {k: None for k in keys}, "label": keys[y]})
    if y != 2 and rng.random() < 0.3:
        qs.append(noul_question("Was the reviewer satisfied?", y >= 3))
    return _request("yelp", state, qs, rng)


# ------------------------------------------------------------------------------ safety / guardrails

def sms_spam(row: Row, names: list[str], rng: random.Random):
    spam = row["label"] == 1
    state, field = wrap_state(row["sms"].strip(), rng, ("sms", "message"))
    if rng.random() < 0.6:
        q = noul_question(refer(field, rng.choice(["Is this message spam?", "This SMS is unsolicited advertising or a scam."]), rng), spam)
    else:
        q = {"type": "choice", "instructions": refer(field, "Classify this text message.", rng),
             "criteria": rng.choice([{"spam": None, "ham": "a normal personal message"},
                                     {"spam": "advertising, scams, unsolicited offers", "legitimate": None}]),
             "label": None}
        q["label"] = "spam" if spam else next(k for k in q["criteria"] if k != "spam")
    return _request("sms_spam", state, [q], rng)


def prompt_injection(row: Row, names: list[str], rng: random.Random):
    state, field = wrap_state(row["text"], rng, ("user_message", "prompt", "input"), p_raw=0.5)
    instr = rng.choice(["Does this text try to override or inject instructions for an AI system?",
                        "Is this a prompt injection attempt?",
                        "The input tries to make the assistant ignore its instructions."])
    return _request("prompt_injections", state, [noul_question(refer(field, instr, rng), row["label"] == 1)], rng)


def jailbreak(row: Row, names: list[str], rng: random.Random):
    bad = row["type"] == "jailbreak"
    state, field = wrap_state(truncate(row["prompt"], 1500), rng, ("prompt", "user_message"), p_raw=0.5)
    if rng.random() < 0.7:
        q = noul_question(refer(field, rng.choice(["Is this prompt a jailbreak attempt?",
                                                   "Does this prompt try to get an AI to bypass its safety rules?"]), rng), bad)
    else:
        q = {"type": "choice", "instructions": refer(field, "Classify this prompt.", rng),
             "criteria": {"benign": "an ordinary request", "jailbreak": "tries to bypass the model's safety rules"},
             "label": "jailbreak" if bad else "benign"}
    return _request("jailbreak", state, [q], rng)


# ------------------------------------------------------------------------------ rubric judging

HELPSTEER = {
    "helpfulness": ("How helpful is `response` for `prompt`?",
                    ["not helpful at all", "slightly helpful", "partially helpful", "mostly helpful",
                     "extremely helpful, fully addresses the request"]),
    "correctness": ("How factually correct and complete is `response`?",
                    ["mostly incorrect", "several significant errors", "some errors or omissions",
                     "mostly correct", "completely correct and complete"]),
    "coherence": ("How clear and consistent is `response`?",
                  ["incoherent", "hard to follow", "somewhat clear", "mostly clear and consistent", "perfectly clear and consistent"]),
    "complexity": ("How sophisticated is the language in `response`?",
                   ["basic language anyone could write", "simple language", "moderately sophisticated",
                    "advanced vocabulary", "expert, domain-specific language"]),
    "verbosity": ("How long and detailed is `response` relative to what `prompt` asks for?",
                  ["very terse", "brief", "moderate length", "long", "very long and detailed"]),
}


def helpsteer(row: Row, names: list[str], rng: random.Random):
    state = {"prompt": truncate(row["prompt"], 1200), "response": truncate(row["response"], 1800)}
    attrs = rng.sample(list(HELPSTEER), rng.randint(1, 3))
    qs: list[dict[str, Any] | None] = [score_question(HELPSTEER[a][0], HELPSTEER[a][1], int(row[a])) for a in attrs]
    if rng.random() < 0.25 and row["correctness"] != 2:
        qs.append(noul_question("Is `response` factually correct?", row["correctness"] >= 3))
    return _request("helpsteer2", state, qs, rng)


# ------------------------------------------------------------------------------ registry

@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    config: str | None
    train_split: str
    dev_split: str | None  # None: carve dev rows out of train
    n_train: int
    n_dev: int
    label_field: str | None
    convert: Convert


SOURCES: list[Source] = [
    Source("ag_news", "fancyzhx/ag_news", None, "train", "test", 2500, 60, "label", ag_news),
    Source("dbpedia_14", "fancyzhx/dbpedia_14", None, "train", "test", 2000, 60, "label", dbpedia),
    Source("yahoo_answers", "community-datasets/yahoo_answers_topics", None, "train", "test", 2500, 60, "topic", yahoo),
    Source("banking77", "legacy-datasets/banking77", None, "train", "test", 2500, 80, "label", banking77),
    Source("clinc_oos", "clinc/clinc_oos", "plus", "train", "validation", 2500, 80, "intent", clinc),
    Source("snli", "stanfordnlp/snli", None, "train", "validation", 2000, 60, "label", nli("snli")),
    Source("multi_nli", "nyu-mll/multi_nli", None, "train", "validation_matched", 2500, 60, "label", nli("multi_nli")),
    Source("mrpc", "nyu-mll/glue", "mrpc", "train", "validation", 1000, 40, "label", mrpc),
    Source("boolq", "google/boolq", None, "train", "validation", 2000, 60, None, boolq),
    Source("arc_easy", "allenai/ai2_arc", "ARC-Easy", "train", "validation", 1200, 40, None, mcq("arc_easy")),
    Source("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "train", "validation", 800, 40, None, mcq("arc_challenge")),
    Source("commonsense_qa", "tau/commonsense_qa", None, "train", "validation", 2000, 60, None, mcq("commonsense_qa")),
    Source("imdb", "stanfordnlp/imdb", None, "train", "test", 1500, 50, "label", imdb),
    Source("yelp", "Yelp/yelp_review_full", None, "train", "test", 2500, 80, "label", yelp),
    Source("sms_spam", "ucirvine/sms_spam", None, "train", None, 1200, 50, "label", sms_spam),
    Source("prompt_injections", "deepset/prompt-injections", None, "train", "test", 546, 60, None, prompt_injection),
    Source("jailbreak", "jackhhao/jailbreak-classification", None, "train", "test", 900, 60, None, jailbreak),
    Source("helpsteer2", "nvidia/HelpSteer2", None, "train", "validation", 2000, 60, None, helpsteer),
]
