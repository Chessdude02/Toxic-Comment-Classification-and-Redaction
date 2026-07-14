"""
Dataset loading for the toxicity classifier.

Prefers a real Jigsaw-style CSV (`jigsaw-toxic-comment-train.csv`, columns
`comment_text`, `toxic`) if one is present next to where the script runs.
Otherwise falls back to a synthetic dataset built from templates.

The previous synthetic generator (see git history) just repeated ~50 fixed
sentences verbatim 10-15x. A model trained on that memorizes literal strings
instead of learning any real signal, and it never saw the "trap" cases
(non-toxic uses of words like "kill" or "hate") that the evaluation scripts
test for - so it had no chance of getting them right. This version builds a
larger, templated corpus that includes those contrastive examples directly.
"""

import itertools
import random
from pathlib import Path

import pandas as pd

SUBJECTS = ["you", "You", "you honestly", "you seriously"]

INSULTS = [
    "an idiot", "a moron", "stupid", "pathetic", "worthless", "a loser",
    "disgusting", "a waste of space", "an imbecile", "a dumbass",
    "a complete joke", "brain-dead", "an embarrassment", "trash",
    "garbage", "a clown", "insufferable", "repulsive",
]

TOXIC_TEMPLATES = [
    "{subj} are {insult}",
    "{subj} are such {insult}",
    "{subj}'re {insult}",
    "wow, {subj} really are {insult}",
    "everyone thinks {subj} are {insult}",
    "{subj} truly are {insult}",
]

TOXIC_COMMANDS = [
    "shut up, {insult}", "shut your mouth", "go away, nobody likes you",
    "get lost, loser", "nobody asked for your opinion, idiot",
    "just stop talking, you're embarrassing yourself",
]

THREATS = [
    "go kill yourself", "i hope you die", "you should just disappear",
    "i will hurt you", "die in a fire", "go to hell",
    "i hope something bad happens to you", "you deserve to suffer",
    "watch your back", "i'll make you regret this",
]

# Person-directed "hate", and dismissive language aimed at an idea/argument
# rather than a person directly - both toxic, but structurally different
# from the "you are {insult}" templates so the model has to actually learn
# the pattern instead of just keying off "you are".
TOXIC_HATE_AND_DISMISSIVE = [
    "i hate you", "i hate you so much", "i hate everything about you",
    "i really hate you", "i can't stand you, i hate you",
    "that's ridiculous", "this idea is ridiculous", "your argument is ridiculous",
    "that's the dumbest thing i've heard", "this is stupid and wrong",
    "this idea is stupid", "that's a stupid argument", "your plan is ridiculous",
    "this is completely ridiculous", "what a ridiculous thing to say",
    "your opinion is stupid", "that argument is dumb and wrong",
]

PROFANITY_INSULTS = [
    "fuck you", "fucking idiot", "shut the hell up", "piece of shit",
    "go fuck yourself", "what a fucking moron", "screw you, asshole",
    "you're a fucking disgrace",
]

SARCASTIC_TOXIC = [
    "what a 'genius' you are (NOT!)", "oh great, another brilliant idea from you... NOT",
    "you're the smartest person... in your dreams", "i love how clueless you are",
    "thank you for being so helpful... NOT", "you are so... intelligent (NOT!)",
    "what a wonderful job you did ruining everything",
    "i'm so impressed by how little you understand",
]

CLEAN_GREETINGS = [
    "hello, how are you today?", "good morning everyone",
    "hope you're having a great day", "hi there, nice to meet you",
    "thanks for stopping by", "have a wonderful evening",
    "hello, how are you?", "hi, how are you doing?", "hey, how have you been?",
    "how are you feeling today?", "how's it going?", "how are things with you?",
    "good afternoon, how are you?", "hi everyone, how's your day going?",
    "hello there, how are you today?", "good to see you, how are you?",
]

CLEAN_GRATITUDE = [
    "thank you for your help", "i really appreciate this",
    "thanks for explaining that so clearly", "this was very helpful, thank you",
    "i'm grateful for your support", "thanks a lot for the advice",
]

CLEAN_AGREEMENT_DISAGREEMENT = [
    "i disagree with your opinion, but that's okay",
    "i think there might be a better approach here",
    "i'm not sure i agree, could you explain more?",
    "that's an interesting point, i see it differently",
    "i respectfully disagree with this conclusion",
    "good point, i hadn't considered that",
    "i think you might be mistaken about this",
    "that's not quite right, let me explain",
]

CLEAN_NEUTRAL = [
    "the weather is nice today", "this article was very informative",
    "i learned something new today", "great explanation, thanks",
    "this is a well written report", "the meeting starts at noon",
    "please let me know if you have questions", "looking forward to the weekend",
]

# Trap cases: contains a "toxic-sounding" trigger word but is not toxic.
CLEAN_CONTEXT_TRAPS = [
    "you killed it in that presentation", "you totally killed it out there",
    "this movie kills the mood a little", "i'm going to kill some time before the meeting",
    "kill the lights before you leave", "i hate waiting in long lines",
    "i hate brussels sprouts", "i hate mondays", "i hate rainy weather",
    "that's insane how good this cake is", "i could die of embarrassment right now",
    "i'm dying laughing at this joke", "that traffic was killing me today",
    "i hate it when my phone battery dies", "this homework is killing me",
]

CLEAN_MILD_CRITICISM = [
    "that's not quite right", "i think this needs more work",
    "that's not the best approach", "i'm not convinced by this argument",
    "this could use some improvement", "that's a bit confusing to me",
]


def _dedup_shuffled(samples, rng):
    seen = set()
    unique = []
    for s in samples:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    rng.shuffle(unique)
    return unique


def _generate_toxic(rng):
    out = []
    for tmpl, subj, insult in itertools.product(TOXIC_TEMPLATES, SUBJECTS, INSULTS):
        out.append(tmpl.format(subj=subj, insult=insult))
    out.extend(TOXIC_COMMANDS)
    out.extend(THREATS)
    out.extend(PROFANITY_INSULTS)
    out.extend(SARCASTIC_TOXIC)
    return _dedup_shuffled(out, rng)


def _generate_clean(rng):
    out = []
    out.extend(CLEAN_GREETINGS)
    out.extend(CLEAN_GRATITUDE)
    out.extend(CLEAN_AGREEMENT_DISAGREEMENT)
    out.extend(CLEAN_NEUTRAL)
    out.extend(CLEAN_CONTEXT_TRAPS)
    out.extend(CLEAN_MILD_CRITICISM)

    # Templated politeness/gratitude variations to broaden vocabulary use
    # and roughly balance the clean class against the toxic class.
    polite_templates = [
        "thanks for {thing}", "i appreciate {thing}", "great job on {thing}",
        "well done on {thing}", "nice work on {thing}", "thank you so much for {thing}",
        "much appreciated regarding {thing}", "i'm grateful for {thing}",
    ]
    things = ["the update", "your help", "this explanation", "the report",
              "your feedback", "the review", "your patience", "the presentation",
              "your support", "the analysis", "your time", "the summary"]
    for tmpl, thing in itertools.product(polite_templates, things):
        out.append(tmpl.format(thing=thing))

    quality_templates = ["the {noun} was really {adj}", "i found the {noun} quite {adj}",
                          "that {noun} seemed {adj} to me"]
    nouns = ["article", "presentation", "report", "session", "class", "book",
             "film", "discussion", "plan", "idea", "meeting", "proposal"]
    adjs = ["interesting", "informative", "helpful", "clear", "useful", "well organized"]
    for tmpl, noun, adj in itertools.product(quality_templates, nouns, adjs):
        out.append(tmpl.format(noun=noun, adj=adj))

    disagreement_templates = ["i think {opinion} might need reconsideration",
                               "{opinion} seems debatable to me",
                               "i'm not fully convinced that {opinion}",
                               "i see {opinion} a little differently"]
    opinions = ["this plan", "that approach", "your estimate", "this timeline",
                "the proposal", "this strategy", "that conclusion", "the assumption"]
    for tmpl, opinion in itertools.product(disagreement_templates, opinions):
        out.append(tmpl.format(opinion=opinion))

    trap_templates = ["i hate {thing2}", "{thing2} is the worst", "i can't stand {thing2}"]
    things2 = ["mondays", "rainy days", "long lines", "traffic jams", "cold coffee",
               "slow wifi", "spam emails", "waiting rooms", "pop quizzes", "early mornings",
               "loud construction noise", "getting stuck in traffic"]
    for tmpl, thing2 in itertools.product(trap_templates, things2):
        out.append(tmpl.format(thing2=thing2))

    return _dedup_shuffled(out, rng)


def build_synthetic_dataset(seed=42):
    """Build a templated synthetic toxic/clean comment dataset."""
    rng = random.Random(seed)
    toxic = _generate_toxic(rng)
    clean = _generate_clean(rng)

    # Keep classes reasonably balanced.
    n = min(len(toxic), len(clean))
    toxic = toxic[:n]
    clean = clean[:n]

    df = pd.DataFrame({
        "comment_text": toxic + clean,
        "toxic": [1] * len(toxic) + [0] * len(clean),
    })
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def _candidate_csv_paths():
    import os

    candidates = []
    if os.environ.get("JIGSAW_TRAIN_CSV"):
        candidates.append(os.environ["JIGSAW_TRAIN_CSV"])
    candidates += [
        "jigsaw-toxic-comment-train.csv",
        "train.csv",
        str(Path(__file__).resolve().parent.parent / "data" / "train.csv"),
    ]
    return candidates


def load_dataset(csv_path=None, seed=42):
    """
    Load a real Jigsaw CSV if one can be found, else synthesize one.

    Looks for (in order): an explicit `csv_path`, the `JIGSAW_TRAIN_CSV` env
    var, `jigsaw-toxic-comment-train.csv` or `train.csv` in the current
    directory, then `models/data/train.csv`. The Kaggle "Toxic Comment
    Classification Challenge" train.csv has extra label columns
    (severe_toxic, obscene, threat, insult, identity_hate) beyond `toxic` -
    only `comment_text`/`toxic` are used here.
    """
    search_paths = [csv_path] if csv_path else _candidate_csv_paths()

    for path in search_paths:
        try:
            df = pd.read_csv(path)
        except (FileNotFoundError, OSError):
            continue
        if "comment_text" not in df.columns or "toxic" not in df.columns:
            continue
        df = df[["comment_text", "toxic"]].dropna(subset=["comment_text"])
        print(f"Loaded real Jigsaw dataset from '{path}': {len(df)} samples "
              f"({df['toxic'].mean():.1%} toxic)")
        return df, "real"

    df = build_synthetic_dataset(seed=seed)
    print(f"No real dataset found in {search_paths}; using synthetic dataset: {len(df)} samples")
    return df, "synthetic"
