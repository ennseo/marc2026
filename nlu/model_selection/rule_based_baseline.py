"""Rule-based (non-LLM) baseline for comparison against parser.py's LLM approach.

Uses spaCy dependency parsing (not just keyword regex) so this is a fair, serious
baseline -- not a strawman. Same output schema as parser.NLUParser.parse():
{target_type, landmark, relation, situation}.

Purpose: quantify how much the LLM actually buys us over a classical NLP approach,
for the paper's ablation/comparison section. See nlu/README.md for the numbers.
"""

import re

import spacy

_REL_WORDS = {
    "on": "on", "onto": "on",
    "near": "near", "by": "near", "close": "near",
    "beside": "beside", "against": "beside",
    "next": "near",  # "next to" handled specially below
    "under": "near",  # not a real category (see README); nearest fit
}

_FIND_VERBS = {"find", "locate", "spot", "notice", "see", "check"}
_LEFT_VERBS = {"leave", "drop", "forget", "lose"}

_EMERGENCY_WORDS = ["collapsed", "motionless", "unresponsive", "unconscious", "not moving", "needs help"]
_ACCIDENT_WORDS = ["accident", "fell", "fallen", "toppled", "tripped", "down next to"]
_ABNORMAL_WORDS = ["climbed", "on top of", "should not be", "odd", "oddly"]

_PERSON_WORDS = {"person", "someone", "people", "one"}


class RuleBasedParser:
    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm")

    def _noun_phrase(self, tok) -> str:
        """tok + any compound/amod children, e.g. 'can' + compound('trash') -> 'trash_can'."""
        parts = [c.text for c in tok.children if c.dep_ in ("compound", "amod")]
        parts.append(tok.lemma_ if tok.pos_ == "NOUN" else tok.text)
        return "_".join(p.lower() for p in parts)

    def _situation(self, sentence: str) -> str:
        s = sentence.lower()
        if any(w in s for w in _EMERGENCY_WORDS):
            return "emergency"
        if any(w in s for w in _ACCIDENT_WORDS):
            return "accident"
        if any(w in s for w in _ABNORMAL_WORDS):
            return "abnormal"
        return "normal"

    def parse(self, sentence: str) -> dict:
        doc = self._nlp(sentence)

        # -- target_type --
        target_type = None
        if any(t.lemma_.lower() in _PERSON_WORDS for t in doc if t.pos_ in ("NOUN", "PRON")):
            target_type = "person"
        else:
            for tok in doc:
                if tok.dep_ == "dobj" and tok.head.lemma_.lower() in (_FIND_VERBS | _LEFT_VERBS):
                    target_type = self._noun_phrase(tok)
                    break
            if target_type is None:
                for tok in doc:
                    if tok.dep_ == "dobj" and tok.pos_ == "NOUN":
                        target_type = self._noun_phrase(tok)
                        break

        # -- landmark + relation --
        landmark, relation = None, None
        for tok in doc:
            if tok.pos_ != "ADP":
                continue
            lemma = tok.lemma_.lower()
            # "next to" / "close to": the real preposition is "to", cued by adjacent next/close
            if lemma == "to" and tok.head.lemma_.lower() in ("next", "close"):
                rel_word = tok.head.lemma_.lower()
            elif lemma in _REL_WORDS:
                rel_word = lemma
            else:
                continue
            pobj = next((c for c in tok.children if c.dep_ == "pobj"), None)
            if pobj is None:
                continue
            # "on top of X" / "in front of X": pobj is a relational noun (top/front/side) that
            # itself takes its own "of X" -- descend one more level to reach the real landmark.
            if pobj.lemma_.lower() in ("top", "front", "side", "back"):
                inner = next((c for c in pobj.children if c.dep_ == "prep" and c.lemma_.lower() == "of"), None)
                inner_pobj = next((c for c in inner.children if c.dep_ == "pobj"), None) if inner else None
                if inner_pobj is not None:
                    pobj = inner_pobj
            landmark = self._noun_phrase(pobj)
            relation = _REL_WORDS[rel_word]
            break  # first spatial preposition found wins

        situation = self._situation(sentence) if target_type == "person" else None

        return {
            "target_type": target_type,
            "landmark": landmark,
            "relation": relation,
            "situation": situation,
        }
