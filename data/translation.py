from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


PAD = 0
BOS = 1
EOS = 2


class TranslationBatch(NamedTuple):
    source: np.ndarray
    decoder_input: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray


@dataclass(frozen=True)
class TranslationDataset:
    train: TranslationBatch
    val: TranslationBatch
    test: TranslationBatch
    source_vocab_size: int
    target_vocab_size: int
    source_vocab: dict[str, int]
    target_vocab: dict[str, int]
    extra_tests: dict[str, TranslationBatch] = field(default_factory=dict)


SUBJECTS = ["i", "you", "we", "they", "cat", "dog", "robot", "teacher"]
OBJECTS = ["book", "apple", "music", "city", "garden", "movie", "river", "letter"]
VERBS = ["see", "like", "carry", "find", "visit", "draw", "watch", "open"]
ADJECTIVES = ["red", "small", "bright", "old", "quiet", "green", "fast", "warm"]
ADVERBS = ["today", "slowly", "again", "outside"]
TENSES = ["past", "future"]
NEGATIONS = ["not"]
PLURALS = ["plural"]

SEMANTIC_COMMON = [
    "please",
    "can",
    "you",
    "need",
    "want",
    "to",
    "for",
    "from",
    "in",
    "on",
    "at",
    "with",
    "and",
    "then",
    "also",
    "book",
    "schedule",
    "set",
    "move",
    "remind",
    "find",
    "reserve",
    "change",
    "cancel",
    "meeting",
    "flight",
    "hotel",
    "restaurant",
    "table",
    "room",
    "call",
    "alarm",
    "me",
    "my",
]
SEMANTIC_PEOPLE = ["alice", "bob", "carol", "dave", "erin", "frank"]
SEMANTIC_CITIES = ["boston", "tokyo", "paris", "berlin", "seattle", "london"]
SEMANTIC_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
SEMANTIC_TIMES = ["morning", "afternoon", "evening", "noon", "nine", "three"]
SEMANTIC_SERVICES = ["calendar", "travel", "dining", "reminder"]
SEMANTIC_PRIORITIES = ["urgent", "normal", "low"]
SEMANTIC_ACTIONS = ["create", "update", "cancel", "search"]

MT_SUBJECTS = ["cat", "dog", "robot", "teacher", "student", "child", "artist", "doctor"]
MT_OBJECTS = ["book", "letter", "apple", "movie", "garden", "river", "music", "city"]
MT_VERBS = ["sees", "likes", "finds", "opens", "visits", "carries", "watches", "draws"]
MT_ADJECTIVES = ["red", "small", "old", "green", "bright", "quiet"]
MT_ADVERBS = ["today", "tomorrow", "yesterday", "slowly", "again"]
MT_V3_SUBJECTS = MT_SUBJECTS + ["engineer", "nurse", "pilot", "writer", "farmer", "chef"]
MT_V3_OBJECTS = MT_OBJECTS + ["bridge", "market", "painting", "machine", "forest", "window"]
MT_V3_VERBS = MT_VERBS + ["repairs", "builds", "cleans", "moves", "studies", "closes"]
MT_V3_ADJECTIVES = MT_ADJECTIVES + ["young", "heavy", "blue", "silver", "difficult", "happy"]
MT_V3_ADVERBS = MT_ADVERBS + ["quickly", "carefully", "inside", "nearby", "politely", "never"]
MT_SUBJECT_DE = {
    "cat": "katze",
    "dog": "hund",
    "robot": "roboter",
    "teacher": "lehrer",
    "student": "student",
    "child": "kind",
    "artist": "kuenstler",
    "doctor": "arzt",
    "engineer": "ingenieur",
    "nurse": "pfleger",
    "pilot": "pilot",
    "writer": "schriftsteller",
    "farmer": "bauer",
    "chef": "koch",
}
MT_OBJECT_DE = {
    "book": "buch",
    "letter": "brief",
    "apple": "apfel",
    "movie": "film",
    "garden": "garten",
    "river": "fluss",
    "music": "musik",
    "city": "stadt",
    "bridge": "bruecke",
    "market": "markt",
    "painting": "gemaelde",
    "machine": "maschine",
    "forest": "wald",
    "window": "fenster",
}
MT_VERB_DE = {
    "sees": "sieht",
    "likes": "mag",
    "finds": "findet",
    "opens": "oeffnet",
    "visits": "besucht",
    "carries": "traegt",
    "watches": "schaut",
    "draws": "zeichnet",
    "repairs": "repariert",
    "builds": "baut",
    "cleans": "reinigt",
    "moves": "bewegt",
    "studies": "studiert",
    "closes": "schliesst",
}
MT_ADJECTIVE_DE = {
    "red": "rote",
    "small": "kleine",
    "old": "alte",
    "green": "gruene",
    "bright": "helle",
    "quiet": "leise",
    "young": "junge",
    "heavy": "schwere",
    "blue": "blaue",
    "silver": "silberne",
    "difficult": "schwierige",
    "happy": "froehliche",
}
MT_ADVERB_DE = {
    "today": "heute",
    "tomorrow": "morgen",
    "yesterday": "gestern",
    "slowly": "langsam",
    "again": "wieder",
    "quickly": "schnell",
    "carefully": "vorsichtig",
    "inside": "drinnen",
    "nearby": "nahe",
    "politely": "hoeflich",
    "never": "nie",
}


def build_synthetic_translation_dataset(config: dict) -> TranslationDataset:
    seed = int(config.get("seed", 0))
    train_size = int(config.get("train_size", 5000))
    val_size = int(config.get("val_size", 1000))
    test_size = int(config.get("test_size", 1000))
    max_source_len = int(config.get("max_source_len", 9))
    max_target_len = int(config.get("max_target_len", 10))
    dataset_name = str(config.get("name", "synthetic_translation")).lower()
    if dataset_name in {"semantic_translation_v2", "translation_v2"}:
        source_vocab, target_vocab = build_semantic_v2_vocabularies()
        sampler = _sample_semantic_v2_pair
    elif dataset_name in {"controlled_bilingual_mt", "controlled_mt", "english_german_mt"}:
        source_vocab, target_vocab = build_controlled_mt_vocabularies()
        sampler = _sample_controlled_mt_pair
    elif dataset_name in {"controlled_bilingual_mt_v3", "translation_v3", "english_german_mt_v3"}:
        source_vocab, target_vocab = build_controlled_mt_v3_vocabularies()
        return _build_controlled_mt_v3_dataset(config, source_vocab, target_vocab)
    else:
        source_vocab, target_vocab = build_vocabularies()
        sampler = _sample_pair
    rng = np.random.default_rng(seed)
    train = _make_split(rng, train_size, max_source_len, max_target_len, source_vocab, target_vocab, sampler)
    val = _make_split(rng, val_size, max_source_len, max_target_len, source_vocab, target_vocab, sampler)
    test = _make_split(rng, test_size, max_source_len, max_target_len, source_vocab, target_vocab, sampler)
    return TranslationDataset(
        train=train,
        val=val,
        test=test,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        source_vocab=source_vocab,
        target_vocab=target_vocab,
    )


def build_vocabularies() -> tuple[dict[str, int], dict[str, int]]:
    source_tokens = (
        ["<pad>", "<bos>", "<eos>"]
        + SUBJECTS
        + OBJECTS
        + VERBS
        + ADJECTIVES
        + ADVERBS
        + TENSES
        + NEGATIONS
        + PLURALS
    )
    target_tokens = (
        ["<pad>", "<bos>", "<eos>"]
        + [f"tr_{token}" for token in SUBJECTS + OBJECTS + VERBS + ADJECTIVES + ADVERBS]
        + ["mk_past", "mk_future", "mk_neg", "mk_plural"]
    )
    return _index(source_tokens), _index(target_tokens)


def build_semantic_v2_vocabularies() -> tuple[dict[str, int], dict[str, int]]:
    source_tokens = (
        ["<pad>", "<bos>", "<eos>"]
        + SEMANTIC_COMMON
        + SEMANTIC_PEOPLE
        + SEMANTIC_CITIES
        + SEMANTIC_DAYS
        + SEMANTIC_TIMES
        + SEMANTIC_SERVICES
        + SEMANTIC_PRIORITIES
        + SEMANTIC_ACTIONS
    )
    target_tokens = (
        ["<pad>", "<bos>", "<eos>"]
        + [f"intent_{service}" for service in SEMANTIC_SERVICES]
        + [f"action_{action}" for action in SEMANTIC_ACTIONS]
        + [f"person_{person}" for person in SEMANTIC_PEOPLE]
        + [f"origin_{city}" for city in SEMANTIC_CITIES]
        + [f"destination_{city}" for city in SEMANTIC_CITIES]
        + [f"location_{city}" for city in SEMANTIC_CITIES]
        + [f"day_{day}" for day in SEMANTIC_DAYS]
        + [f"time_{time}" for time in SEMANTIC_TIMES]
        + [f"priority_{priority}" for priority in SEMANTIC_PRIORITIES]
        + ["needs_hotel", "needs_restaurant", "with_reminder", "has_sequence"]
    )
    return _index(source_tokens), _index(target_tokens)


def build_controlled_mt_vocabularies() -> tuple[dict[str, int], dict[str, int]]:
    source_tokens = (
        ["<pad>", "<bos>", "<eos>", "the", "a", "will", "did", "not", "very"]
        + MT_SUBJECTS
        + MT_OBJECTS
        + MT_VERBS
        + MT_ADJECTIVES
        + MT_ADVERBS
    )
    target_tokens = (
        ["<pad>", "<bos>", "<eos>", "der", "die", "das", "ein", "eine", "wird", "hat", "nicht", "sehr"]
        + list(MT_SUBJECT_DE.values())
        + list(MT_OBJECT_DE.values())
        + list(MT_VERB_DE.values())
        + list(MT_ADJECTIVE_DE.values())
        + list(MT_ADVERB_DE.values())
    )
    return _index(source_tokens), _index(target_tokens)


def build_controlled_mt_v3_vocabularies() -> tuple[dict[str, int], dict[str, int]]:
    source_tokens = (
        ["<pad>", "<bos>", "<eos>", "the", "a", "will", "did", "not", "very", "and", "while", "near"]
        + MT_V3_SUBJECTS
        + MT_V3_OBJECTS
        + MT_V3_VERBS
        + MT_V3_ADJECTIVES
        + MT_V3_ADVERBS
    )
    target_tokens = (
        [
            "<pad>",
            "<bos>",
            "<eos>",
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "wird",
            "hat",
            "nicht",
            "sehr",
            "und",
            "waehrend",
            "bei",
        ]
        + list(MT_SUBJECT_DE.values())
        + list(MT_OBJECT_DE.values())
        + list(MT_VERB_DE.values())
        + list(MT_ADJECTIVE_DE.values())
        + list(MT_ADVERB_DE.values())
    )
    return _index(source_tokens), _index(target_tokens)


def decode_target(ids: np.ndarray, vocab: dict[str, int]) -> list[str]:
    inverse = {idx: token for token, idx in vocab.items()}
    tokens = []
    for item in ids.tolist():
        if item in (PAD, BOS):
            continue
        if item == EOS:
            break
        tokens.append(inverse[int(item)])
    return tokens


def _make_split(
    rng: np.random.Generator,
    size: int,
    max_source_len: int,
    max_target_len: int,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    sampler,
) -> TranslationBatch:
    source = np.zeros((size, max_source_len), dtype=np.int32)
    decoder_input = np.zeros((size, max_target_len), dtype=np.int32)
    target = np.zeros((size, max_target_len), dtype=np.int32)
    target_mask = np.zeros((size, max_target_len), dtype=np.float32)
    for i in range(size):
        source_tokens, target_tokens = sampler(rng)
        source[i] = _encode(source_tokens, source_vocab, max_source_len, add_bos=False)
        target_with_eos = _encode(target_tokens, target_vocab, max_target_len, add_bos=False)
        decoder_tokens = ["<bos>"] + target_tokens
        decoder_input[i] = _encode(decoder_tokens, target_vocab, max_target_len, add_bos=False)
        target[i] = target_with_eos
        target_mask[i] = (target_with_eos != PAD).astype(np.float32)
    return TranslationBatch(source, decoder_input, target, target_mask)


def _build_controlled_mt_v3_dataset(
    config: dict,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> TranslationDataset:
    seed = int(config.get("seed", 0))
    train_size = int(config.get("train_size", 8000))
    val_size = int(config.get("val_size", 1200))
    test_size = int(config.get("test_size", 1200))
    composition_test_size = int(config.get("composition_test_size", test_size))
    long_test_size = int(config.get("long_test_size", test_size))
    max_source_len = int(config.get("max_source_len", 28))
    max_target_len = int(config.get("max_target_len", 30))

    rng = np.random.default_rng(seed)
    train = _make_split(
        rng,
        train_size,
        max_source_len,
        max_target_len,
        source_vocab,
        target_vocab,
        lambda item_rng: _sample_controlled_mt_v3_pair(item_rng, split="train"),
    )
    val = _make_split(
        rng,
        val_size,
        max_source_len,
        max_target_len,
        source_vocab,
        target_vocab,
        lambda item_rng: _sample_controlled_mt_v3_pair(item_rng, split="iid"),
    )
    test = _make_split(
        rng,
        test_size,
        max_source_len,
        max_target_len,
        source_vocab,
        target_vocab,
        lambda item_rng: _sample_controlled_mt_v3_pair(item_rng, split="iid"),
    )
    composition = _make_split(
        rng,
        composition_test_size,
        max_source_len,
        max_target_len,
        source_vocab,
        target_vocab,
        lambda item_rng: _sample_controlled_mt_v3_pair(item_rng, split="composition"),
    )
    long = _make_split(
        rng,
        long_test_size,
        max_source_len,
        max_target_len,
        source_vocab,
        target_vocab,
        lambda item_rng: _sample_controlled_mt_v3_pair(item_rng, split="long"),
    )
    return TranslationDataset(
        train=train,
        val=val,
        test=test,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        extra_tests={"composition": composition, "long": long},
    )


def _sample_pair(rng: np.random.Generator) -> tuple[list[str], list[str]]:
    subject = _choice(rng, SUBJECTS)
    verb = _choice(rng, VERBS)
    obj = _choice(rng, OBJECTS)
    adj = _choice(rng, ADJECTIVES)
    tense = _choice(rng, TENSES)
    use_neg = bool(rng.random() < 0.35)
    use_plural = bool(rng.random() < 0.30)
    use_adverb = bool(rng.random() < 0.45)
    adverb = _choice(rng, ADVERBS)

    source = [subject, adj, obj]
    if use_plural:
        source.append("plural")
    source.append(verb)
    source.append(tense)
    if use_neg:
        source.append("not")
    if use_adverb:
        source.append(adverb)

    target = [f"tr_{subject}", f"tr_{verb}"]
    if use_neg:
        target.append("mk_neg")
    target.extend([f"tr_{obj}", f"tr_{adj}"])
    if use_plural:
        target.append("mk_plural")
    target.append("mk_past" if tense == "past" else "mk_future")
    if use_adverb:
        target.append(f"tr_{adverb}")
    return source, target


def _sample_semantic_v2_pair(rng: np.random.Generator) -> tuple[list[str], list[str]]:
    intent = _choice(rng, SEMANTIC_SERVICES)
    action = _choice(rng, SEMANTIC_ACTIONS)
    person = _choice(rng, SEMANTIC_PEOPLE)
    day = _choice(rng, SEMANTIC_DAYS)
    time = _choice(rng, SEMANTIC_TIMES)
    priority = _choice(rng, SEMANTIC_PRIORITIES)
    origin = _choice(rng, SEMANTIC_CITIES)
    destination = _different_choice(rng, SEMANTIC_CITIES, origin)
    location = _choice(rng, SEMANTIC_CITIES)
    with_reminder = bool(rng.random() < 0.45)
    has_sequence = bool(rng.random() < 0.35)

    if intent == "travel":
        needs_hotel = bool(rng.random() < 0.55)
        source = [
            "please",
            "book" if action in {"create", "search"} else "change",
            "travel",
            "from",
            origin,
            "to",
            destination,
            "on",
            day,
            "in",
            time,
            "for",
            person,
        ]
        if needs_hotel:
            source.extend(["and", "hotel"])
        if with_reminder:
            source.extend(["then", "remind", "me"])
        target = [
            "intent_travel",
            f"action_{action}",
            f"person_{person}",
            f"origin_{origin}",
            f"destination_{destination}",
            f"day_{day}",
            f"time_{time}",
            f"priority_{priority}",
        ]
        if needs_hotel:
            target.append("needs_hotel")
    elif intent == "calendar":
        source = [
            "can",
            "you",
            "schedule" if action != "cancel" else "cancel",
            "meeting",
            "with",
            person,
            "on",
            day,
            "at",
            time,
            "in",
            location,
        ]
        if priority == "urgent":
            source.insert(0, "urgent")
        if with_reminder:
            source.extend(["and", "set", "remind"])
        target = [
            "intent_calendar",
            f"action_{action}",
            f"person_{person}",
            f"location_{location}",
            f"day_{day}",
            f"time_{time}",
            f"priority_{priority}",
        ]
    elif intent == "dining":
        source = [
            "reserve" if action in {"create", "search"} else "change",
            "restaurant",
            "table",
            "for",
            person,
            "in",
            location,
            "on",
            day,
            "at",
            time,
        ]
        target = [
            "intent_dining",
            f"action_{action}",
            f"person_{person}",
            f"location_{location}",
            f"day_{day}",
            f"time_{time}",
            f"priority_{priority}",
            "needs_restaurant",
        ]
    else:
        source = [
            "set" if action != "cancel" else "cancel",
            "remind",
            "me",
            "to",
            "call",
            person,
            "on",
            day,
            "at",
            time,
        ]
        target = [
            "intent_reminder",
            f"action_{action}",
            f"person_{person}",
            f"day_{day}",
            f"time_{time}",
            f"priority_{priority}",
            "with_reminder",
        ]

    if priority != "normal" and priority not in source:
        source.append(priority)
    if with_reminder and "with_reminder" not in target:
        target.append("with_reminder")
    if has_sequence:
        source.extend(["also", "then"])
        target.append("has_sequence")
    return source, target


def _sample_controlled_mt_pair(rng: np.random.Generator) -> tuple[list[str], list[str]]:
    subject = _choice(rng, MT_SUBJECTS)
    verb = _choice(rng, MT_VERBS)
    obj = _choice(rng, MT_OBJECTS)
    subject_adj = _choice(rng, MT_ADJECTIVES)
    object_adj = _choice(rng, MT_ADJECTIVES)
    adverb = _choice(rng, MT_ADVERBS)
    tense = _choice(rng, ["present", "future", "past"])
    use_neg = bool(rng.random() < 0.30)
    use_subject_adj = bool(rng.random() < 0.55)
    use_object_adj = bool(rng.random() < 0.55)
    use_adverb = bool(rng.random() < 0.60)

    source = ["the"]
    if use_subject_adj:
        source.append(subject_adj)
    source.append(subject)
    if tense == "future":
        source.append("will")
    elif tense == "past":
        source.append("did")
    if use_neg:
        source.append("not")
    source.append(verb)
    source.append("a")
    if use_object_adj:
        source.append(object_adj)
    source.append(obj)
    if use_adverb:
        source.append(adverb)

    target = ["der"]
    if use_subject_adj:
        target.append(MT_ADJECTIVE_DE[subject_adj])
    target.append(MT_SUBJECT_DE[subject])
    if use_adverb:
        target.append(MT_ADVERB_DE[adverb])
    if tense == "future":
        target.append("wird")
    elif tense == "past":
        target.append("hat")
    if use_neg:
        target.append("nicht")
    target.append(MT_VERB_DE[verb])
    target.append("ein")
    if use_object_adj:
        target.append(MT_ADJECTIVE_DE[object_adj])
    target.append(MT_OBJECT_DE[obj])
    return source, target


def _sample_controlled_mt_v3_pair(
    rng: np.random.Generator,
    *,
    split: str,
) -> tuple[list[str], list[str]]:
    if split == "long":
        first = _sample_controlled_mt_v3_clause(
            rng,
            tense=_choice(rng, ["present", "future", "past"]),
            use_neg=bool(rng.random() < 0.45),
            use_adverb=True,
            use_subject_adj=True,
            use_object_adj=True,
        )
        second = _sample_controlled_mt_v3_clause(
            rng,
            tense=_choice(rng, ["present", "future", "past"]),
            use_neg=bool(rng.random() < 0.45),
            use_adverb=True,
            use_subject_adj=bool(rng.random() < 0.70),
            use_object_adj=True,
        )
        source = first[0] + ["and", "while"] + second[0]
        target = first[1] + ["und", "waehrend"] + second[1]
        return source, target

    if split == "composition":
        return _sample_controlled_mt_v3_clause(
            rng,
            tense="future",
            use_neg=True,
            use_adverb=True,
            use_subject_adj=bool(rng.random() < 0.75),
            use_object_adj=True,
        )

    while True:
        tense = _choice(rng, ["present", "future", "past"])
        use_neg = bool(rng.random() < 0.25)
        use_adverb = bool(rng.random() < 0.55)
        if split == "train" and tense == "future" and use_neg and use_adverb:
            continue
        return _sample_controlled_mt_v3_clause(
            rng,
            tense=tense,
            use_neg=use_neg,
            use_adverb=use_adverb,
            use_subject_adj=bool(rng.random() < 0.60),
            use_object_adj=bool(rng.random() < 0.60),
        )


def _sample_controlled_mt_v3_clause(
    rng: np.random.Generator,
    *,
    tense: str,
    use_neg: bool,
    use_adverb: bool,
    use_subject_adj: bool,
    use_object_adj: bool,
) -> tuple[list[str], list[str]]:
    subject = _choice(rng, MT_V3_SUBJECTS)
    verb = _choice(rng, MT_V3_VERBS)
    obj = _choice(rng, MT_V3_OBJECTS)
    subject_adj = _choice(rng, MT_V3_ADJECTIVES)
    object_adj = _choice(rng, MT_V3_ADJECTIVES)
    adverb = _choice(rng, MT_V3_ADVERBS)
    intensify_object = bool(rng.random() < 0.35)

    source = ["the"]
    if use_subject_adj:
        source.append(subject_adj)
    source.append(subject)
    if tense == "future":
        source.append("will")
    elif tense == "past":
        source.append("did")
    if use_neg:
        source.append("not")
    source.append(verb)
    source.append("a")
    if intensify_object and use_object_adj:
        source.append("very")
    if use_object_adj:
        source.append(object_adj)
    source.append(obj)
    if use_adverb:
        source.append(adverb)

    target = ["der"]
    if use_subject_adj:
        target.append(MT_ADJECTIVE_DE[subject_adj])
    target.append(MT_SUBJECT_DE[subject])
    if use_adverb:
        target.append(MT_ADVERB_DE[adverb])
    if tense == "future":
        target.append("wird")
    elif tense == "past":
        target.append("hat")
    if use_neg:
        target.append("nicht")
    target.append(MT_VERB_DE[verb])
    target.append("ein")
    if intensify_object and use_object_adj:
        target.append("sehr")
    if use_object_adj:
        target.append(MT_ADJECTIVE_DE[object_adj])
    target.append(MT_OBJECT_DE[obj])
    return source, target


def _choice(rng: np.random.Generator, values: list[str]) -> str:
    return values[int(rng.integers(0, len(values)))]


def _different_choice(rng: np.random.Generator, values: list[str], excluded: str) -> str:
    candidates = [value for value in values if value != excluded]
    return _choice(rng, candidates)


def _encode(tokens: list[str], vocab: dict[str, int], max_len: int, add_bos: bool) -> np.ndarray:
    full = (["<bos>"] if add_bos else []) + tokens + ["<eos>"]
    if len(full) > max_len:
        raise ValueError(f"Sequence length {len(full)} exceeds max_len={max_len}: {full}")
    ids = np.zeros((max_len,), dtype=np.int32)
    ids[: len(full)] = np.asarray([vocab[token] for token in full], dtype=np.int32)
    return ids


def _index(tokens: list[str]) -> dict[str, int]:
    return {token: index for index, token in enumerate(tokens)}
