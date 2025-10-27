# username : judealphonsesavio_db_user
# password : CbGqKjIikb7Njgqp
# connection_string : mongodb+srv://judealphonsesavio_db_user:CbGqKjIikb7Njgqp@readytech-cluster.qjj8qmm.mongodb.net/?appName=ReadyTech-Cluster 

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.operations import IndexModel
from bson import ObjectId as BsonObjectId
from bson import ObjectId
from dotenv import load_dotenv
import os
import json
import re
from groq import Groq

load_dotenv()

# ---------- similarity helper ----------
import math
from collections import Counter

def _normalize_and_tokenize(text: str):
    if not text:
        return []
    s = re.sub(r'[^0-9a-zA-Z\s]', ' ', text.lower())
    tokens = [t for t in s.split() if t]
    return tokens

def compute_cosine_similarity(a: str, b: str) -> float:
    """
    Lightweight cosine similarity based on token counts (bag-of-words).
    Returns float in [0.0, 1.0]. Rounds to 4 decimals.
    This function should only be called if an expected answer exists.
    """
    ta = _normalize_and_tokenize(a)
    tb = _normalize_and_tokenize(b)
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    intersection = set(ca.keys()) & set(cb.keys())
    dot = sum(ca[k] * cb[k] for k in intersection)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    sim = dot / (na * nb)
    return float(round(sim, 4))

# ========== CONFIG ==========
USE_MOCK_DATA = False
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL")
MONGO_URI = os.getenv("MONGO_URI")

# ========== APP & DB ==========
app = FastAPI(title="Survey Chat Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])

client = MongoClient(MONGO_URI)
db = client.survey_chat_app

# ========== Pydantic Models ==========
class UserMessage(BaseModel):
    user_id: str
    survey_id: str
    message: str

class SurveyResponse(BaseModel):
    response: str
    status: str
    current_question: Optional[str] = None
    current_question_index: Optional[int] = None
    progress: float = 0.0
    completed: bool = False

class RecommendRequest(BaseModel):
    title: str
    description: str
    num_questions: Optional[int] = 3

class RecommendedQuestion(BaseModel):
    question_title: str
    question_description: str
    expected_answer: str

class RecommendResponse(BaseModel):
    questions: List[RecommendedQuestion]

# ========== Groq client wrapper ==========
groq_client = Groq(api_key=GROQ_API_KEY)

def extract_json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

# ========== user_survey_state helpers ==========
def get_or_create_user_survey_state(user_id: str, survey_id: str) -> dict:
    # normalize survey_id as string for user_survey_state documents
    survey_id = str(survey_id)
    doc = db.user_survey_state.find_one({"user_id": user_id, "survey_id": survey_id})
    if doc:
        if "answers" not in doc:
            doc["answers"] = {}
        if "user_conversations" not in doc:
            doc["user_conversations"] = []
        if "expecting_revision" not in doc:
            doc["expecting_revision"] = False
        if "pending_edit_q" not in doc:
            doc["pending_edit_q"] = None
        if "answer_similarities" not in doc:
            doc["answer_similarities"] = {}
        return doc

    user = db.users.find_one({"_id": user_id})
    assigned = None
    if user:
        for a in user.get("assignedSurveys", []):
            if str(a.get("surveyId")) == str(survey_id):
                assigned = a
                break

    new_doc = {
        "user_id": user_id,
        "survey_id": survey_id,
        "status": "not_started",
        "current_question_index": 0,
        "answers": assigned.get("answers", {}) if assigned else {},
        "user_conversations": assigned.get("conversationTurns", []) if assigned else [],
        "startedAt": assigned.get("startedAt") if assigned else None,
        "completedAt": assigned.get("completedAt") if assigned else None,
        "expecting_revision": False,
        "pending_edit_q": None,
        "answer_similarities": assigned.get("answer_similarities", {}) if assigned else {},
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow()
    }
    db.user_survey_state.insert_one(new_doc)
    return db.user_survey_state.find_one({"user_id": user_id, "survey_id": survey_id})

def append_user_conversation(state_doc: dict, role: str, content: str) -> list:
    turns = state_doc.get("user_conversations", []) or []
    turns.append({"role": role, "content": content, "timestamp": datetime.utcnow()})
    turns = turns[-500:]  # keep a reasonable history (up to 500 turns)
    db.user_survey_state.update_one(
        {"_id": state_doc["_id"]},
        {"$set": {"user_conversations": turns, "updatedAt": datetime.utcnow()}}
    )
    return turns

def update_user_survey_state(state_doc: dict, updates: dict):
    updates["updatedAt"] = datetime.utcnow()
    db.user_survey_state.update_one({"_id": state_doc["_id"]}, {"$set": updates})
    return db.user_survey_state.find_one({"_id": state_doc["_id"]})

def mirror_to_user_assignment(user_id: str, survey_id: str, state_doc: dict):
    try:
        # attempt to update backward compatible assignment; ignore failures
        # survey_id stored in state_doc is string — convert to ObjectId for matching assignedSurveys.surveyId
        db.users.update_one(
            {"_id": user_id, "assignedSurveys.surveyId": survey_id},
            {"$set": {
                "assignedSurveys.$.answers": state_doc.get("answers", {}),
                "assignedSurveys.$.answer_similarities": state_doc.get("answer_similarities", {}),
                "assignedSurveys.$.currentQuestionIndex": state_doc.get("current_question_index", 0),
                "assignedSurveys.$.status": state_doc.get("status", "active"),
                "assignedSurveys.$.conversationTurns": state_doc.get("user_conversations", [])
            }}
        )
    except Exception:
        pass

# ========== Survey Agent (graph-like) with LLM intent classifier and judge ==========
class SurveyAgentGraph:
    def __init__(self, db):
        self.db = db
        self.client = groq_client
        self.model = GROQ_MODEL
        self.classifier_model = CLASSIFIER_MODEL

    def _get_survey(self, survey_id: str) -> dict:
        try:
            return self.db.surveys.find_one({"_id": survey_id})
        except Exception:
            return None

    def greet(self, state: dict) -> dict:
        survey = state["survey"]
        answers = state["answers"]
        total_q = len(survey["questions"])
        answered = len(answers)
        if answered == 0:
            greeting = f"👋 Welcome to the '{survey['title']}' survey!\n\n{survey['description']}\n\nLet's get started."
        else:
            percent = (answered / total_q) * 100
            greeting = f"Welcome back — you've answered {answered} of {total_q} questions ({percent:.0f}%). Let's continue."
        state["response"] = greeting
        return state

    def ask_question(self, state: dict) -> dict:
        survey = state["survey"]
        idx = state["current_index"]
        if idx >= len(survey["questions"]):
            state["next_node"] = "complete"
            return state
        q = survey["questions"][idx]
        state["current_question"] = q
        nav = "\n\n(You can 'skip', 'back', 'review', 'revise', or 'go to question <n>'.)"
        state["response"] = f"Q{idx+1}: {q['text']}{nav}"
        return state

    def _validate_with_groq(self, question: dict, answer: str) -> Tuple[bool, str, float]:
        expected = question.get("expectedAnswer", "")
        if not expected:
            return (False, "Looks okay.", 1.0)
        prompt = (
            f"Question: {question['text']}\n"
            f"Expected Answer Pattern: {expected}\n"
            f"User's Answer: {answer}\n\n"
            "Return JSON {\"needs_correction\": boolean, \"feedback\": string, \"confidence\": float}.\n"
            "Be concise."
        )
        json_schema = {
            "name": "validation_result",
            "schema": {
                "type": "object",
                "properties": {
                    "needs_correction": {"type": "boolean"},
                    "feedback": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["needs_correction", "feedback", "confidence"]
            }
        }
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a concise survey answer validator."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": json_schema},
            )
            raw = resp.choices[0].message.content or ""
            parsed = extract_json_from_text(raw) or {}
            needs = bool(parsed.get("needs_correction", False))
            feedback = parsed.get("feedback", "").strip() or "Looks okay."
            confidence = float(parsed.get("confidence", 0.0))
            if confidence < 0.4:
                return (False, feedback, confidence)
            return (needs, feedback, confidence)
        except Exception:
            exp_tokens = set(re.findall(r"\w+", expected.lower()))
            ans_tokens = set(re.findall(r"\w+", answer.lower()))
            overlap = exp_tokens.intersection(ans_tokens)
            if len(overlap) >= 2 or (len(overlap) / max(1, len(exp_tokens)) > 0.25):
                return (False, "Partially matches expected answer.", 0.5)
            else:
                return (False, "Answer accepted (did not match expected pattern closely, but recorded).", 0.2)

    def _next_unanswered_index(self, state: dict) -> int:
        """
        Return the index of the next unanswered question.
        If all answered, return len(questions) as sentinel (meaning complete).
        Search from current_index forward then wrap to beginning.
        """
        survey = state["survey"]
        total = len(survey["questions"])
        answers = state.get("answers", {}) or {}

        start = max(0, int(state.get("current_index", 0)))
        for i in range(start, total):
            qid = str(survey["questions"][i]["id"])
            if qid not in answers:
                return i
        for i in range(0, start):
            qid = str(survey["questions"][i]["id"])
            if qid not in answers:
                return i
        return total

    def _classify_intent_with_groq(self, state: dict, message: str) -> dict:
        """
        Use Groq-based LLM (CLASSIFIER_MODEL) to classify intent.
        Returns a dict:
          {
            "intent": one of ["answer","skip","goto","back","review","revise","start","thanks","unknown"],
            "question_number": Optional[int],
            "revision_text": Optional[str],
            "confidence": float
          }
        """
        turns = state["state_doc"].get("user_conversations", [])[-6:]
        convo_text = ""
        for t in turns:
            role = t.get("role", "user")
            content = t.get("content", "")
            convo_text += f"{role}: {content}\n"
        current_q_text = state.get("current_question", {}).get("text") if state.get("current_question") else ""
        prompt = (
            "You are an intent classification assistant. Given the recent conversation and the user's latest message, "
            "classify the user's intent into one of: answer, skip, goto, back, review, revise, start, thanks, unknown.\n\n"
            "Return JSON with keys:\n"
            "- intent: string (one of the above)\n"
            "- question_number: integer or null (if the user specified a target question)\n"
            "- revision_text: string or null (if the user provided the new answer in same message)\n"
            "- confidence: number between 0 and 1\n\n"
            "Be concise and only output valid JSON. Use null for missing fields.\n\n"
            "Context:\n{convo}\nCurrent question: {cq}\nUser message: {msg}\n\n"
            "If the user message appears to be a direct answer to the current question, classify as 'answer'. "
            "If the user asks to 'go to' or 'skip to' a numbered question, classify as 'goto' (extract number). "
            "If user says 'skip' or 'next' without number classify as 'skip'. "
            "If user asks 'review' without number classify 'review', if with number include question_number. "
            "If user asks to 'revise' or 'edit' and provides a number, set intent 'revise' and question_number. "
            "If user provides the new answer text along with the revise command, place it in revision_text. "
            "If unsure, set intent 'unknown'."
        ).format(convo=convo_text, cq=current_q_text, msg=message)

        json_schema = {
            "name": "intent_result",
            "schema": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "question_number": {"type": ["integer", "null"]},
                    "revision_text": {"type": ["string", "null"]},
                    "confidence": {"type": "number"}
                },
                "required": ["intent", "question_number", "revision_text", "confidence"]
            }
        }

        try:
            resp = self.client.chat.completions.create(
                model=self.classifier_model,
                messages=[
                    {"role": "system", "content": "You are a clear intent classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_schema", "json_schema": json_schema},
            )
            raw = resp.choices[0].message.content or ""
            parsed = extract_json_from_text(raw) or {}
            intent = parsed.get("intent", "unknown")
            qn = parsed.get("question_number", None)
            rev_text = parsed.get("revision_text", None)
            confidence = float(parsed.get("confidence", 0.0))
            return {
                "intent": intent if isinstance(intent, str) else "unknown",
                "question_number": int(qn) if isinstance(qn, (int, float)) else (None if qn is None else None),
                "revision_text": rev_text if isinstance(rev_text, str) and rev_text.strip() else None,
                "confidence": confidence
            }
        except Exception:
            lower = (message or "").lower()

            # Strong pattern: user asking "what was my answer" => REVIEW (single question)
            if re.search(r"\bwhat (?:was|did) i (?:answer|say)\b", lower) or re.search(r"\bwhat was my answer\b", lower):
                m = re.search(r"(\d{1,3})", lower)
                return {"intent": "review", "question_number": int(m.group(1)) if m else None, "revision_text": None, "confidence": 0.6}

            # "what is/was my answer to question 5" pattern
            if re.search(r"answer to question", lower) or re.search(r"what was my answer to question", lower):
                m = re.search(r"(\d{1,3})", lower)
                return {"intent": "review", "question_number": int(m.group(1)) if m else None, "revision_text": None, "confidence": 0.6}

            # support 'last' mapping
            if "last" in lower or "final" in lower:
                total = len(state.get("survey", {}).get("questions", []))
                if total:
                    return {"intent": "goto", "question_number": total, "revision_text": None, "confidence": 0.4}
            if any(kw in lower for kw in ["skip to", "go to", "goto", "go to question", "jump to"]):
                m = re.search(r"(\d{1,3})", lower)
                return {"intent": "goto", "question_number": int(m.group(1)) if m else None, "revision_text": None, "confidence": 0.2}
            if "skip" in lower and "question" not in lower or re.search(r"\bskip\b|\bnext\b", lower):
                return {"intent": "skip", "question_number": None, "revision_text": None, "confidence": 0.2}
            if "back" in lower or "previous" in lower:
                return {"intent": "back", "question_number": None, "revision_text": None, "confidence": 0.2}
            if "review" in lower:
                m = re.search(r"(\d{1,3})", lower)
                return {"intent": "review", "question_number": int(m.group(1)) if m else None, "revision_text": None, "confidence": 0.2}
            # detect 'revise' or 'change' but ensure we don't conflate review phrases
            if any(w in lower for w in ["revise", "edit", "change", "update", "correct"]) and "what was my answer" not in lower:
                m = re.search(r"(\d{1,3})", lower)
                rev_text = None
                split_by = re.split(r":|\"|'|- ", message, 1)
                if len(split_by) > 1:
                    rev_text = split_by[1].strip()
                return {"intent": "revise", "question_number": int(m.group(1)) if m else None, "revision_text": rev_text, "confidence": 0.2}
            if lower.strip() in ("start", "begin", "hi", "hello", ""):
                return {"intent": "start", "question_number": None, "revision_text": None, "confidence": 0.3}
            if any(w in lower for w in ["thank", "thanks", "thx"]):
                return {"intent": "thanks", "question_number": None, "revision_text": None, "confidence": 0.9}
            return {"intent": "answer", "question_number": None, "revision_text": None, "confidence": 0.2}

    def _judge_answer_with_groq(self, question: dict, answer: str) -> dict:
        """
        Produce an LLM-based judge for the answer's closeness to the expected answer.
        Returns dict: {"comment": str, "confidence_score": int} where confidence_score in 0..100.
        Uses CLASSIFIER_MODEL via groq_client. Falls back to heuristic if LLM fails.
        """
        expected = question.get("expectedAnswer", "")
        # Build a concise prompt instructing the LLM to output strict JSON
        prompt = (
            "You are a concise reviewer. Given a survey question, the expected answer pattern, and a respondent's answer, "
            "produce JSON: {\"comment\": string, \"confidence_score\": integer} where confidence_score ranges 0-100 "
            "and represents how closely the respondent's answer matches the expected answer. Be short and directly evaluative.\n\n"
            f"Question: {question.get('text')}\n"
            f"Expected Answer Pattern: {expected}\n"
            f"Respondent Answer: {answer}\n\n"
            "Return only valid JSON."
        )
        json_schema = {
            "name": "llm_judge",
            "schema": {
                "type": "object",
                "properties": {
                    "comment": {"type": "string"},
                    "confidence_score": {"type": "integer"}
                },
                "required": ["comment", "confidence_score"]
            }
        }
        try:
            resp = self.client.chat.completions.create(
                model=self.classifier_model,
                messages=[
                    {"role": "system", "content": "You are a concise answer judge."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_schema", "json_schema": json_schema},
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
            parsed = extract_json_from_text(raw) or {}
            comment = parsed.get("comment", "").strip()
            confidence_score = int(parsed.get("confidence_score", 0))
            # clamp
            confidence_score = max(0, min(100, confidence_score))
            if not comment:
                comment = "No comment provided."
            return {"comment": comment, "confidence_score": confidence_score}
        except Exception:
            # Fallback heuristic: use text overlap / similarity signal
            try:
                sim = compute_cosine_similarity(expected, answer) if expected else 0.0
            except Exception:
                sim = 0.0
            # Map similarity 0..1 to 0..100 roughly, but nudge for minimal one-word acceptances
            score = int(round(sim * 100))
            if score > 95:
                comment = "Excellent — closely matches expected answer."
            elif score > 75:
                comment = "Good match; most key points present."
            elif score > 50:
                comment = "Partial match; some parts missing."
            elif score > 25:
                comment = "Weak match; several expected elements missing."
            else:
                # If answer is short but seems to answer directly, give a small boost
                if len(_normalize_and_tokenize(answer)) <= 3 and len(answer.strip()) > 0:
                    score = max(score, 35)
                    comment = "Short direct answer; may be acceptable depending on expected detail."
                else:
                    comment = "Does not match expected answer closely."
            score = max(0, min(100, score))
            return {"comment": comment, "confidence_score": score}

    def process_answer(self, state: dict) -> dict:
        msg = state.get("user_message", "").strip()
        state_doc = state["state_doc"]

        classifier = self._classify_intent_with_groq(state, msg)
        intent = classifier.get("intent", "unknown")
        qn = classifier.get("question_number")
        rev_text = classifier.get("revision_text")
        conf = classifier.get("confidence", 0.0)

        # Revised logic for 'revise' intent:
        if intent == "revise":
            if qn:
                q_index = max(0, int(qn) - 1)
                if q_index >= len(state["survey"]["questions"]):
                    state["response"] = f"I couldn't find question {qn} to update. Please provide a valid question number."
                    state["next_node"] = "ask_question"
                    append_user_conversation(state_doc, "user", msg)
                    return state
                # If revision text is provided inline, apply update immediately (validate then save)
                if rev_text:
                    q = state["survey"]["questions"][q_index]
                    needs_correction, feedback, confidence = self._validate_with_groq(q, rev_text)
                    answers = state.get("answers", {}) or {}
                    answers[str(q["id"])] = rev_text
                    state["answers"] = answers

                    # compute similarity and judge only if expectedAnswer exists
                    answer_similarities = state.get("answer_similarities", {}) or {}
                    expected = q.get("expectedAnswer", "")
                    if expected:
                        sim = compute_cosine_similarity(expected, rev_text)
                        judge = self._judge_answer_with_groq(q, rev_text)
                        answer_similarities[str(q["id"])] = {
                            "similarity": sim,
                            "llm_judge": judge
                        }
                        state["answer_similarities"] = answer_similarities

                    new_doc = update_user_survey_state(state_doc, {
                        "answers": answers,
                        "answer_similarities": state.get("answer_similarities", {}),
                        "expecting_revision": False,
                        "pending_edit_q": None
                    })
                    state["state_doc"] = new_doc
                    append_user_conversation(new_doc, "user", msg)
                    append_user_conversation(new_doc, "agent", f"✅ Updated answer for Q{q_index+1}: {feedback if confidence < 0.5 else 'Updated.'}")
                    mirror_to_user_assignment(state["user"]["_id"], state["survey"]["_id"], new_doc)
                    state["response"] = f"✅ Updated answer for Q{q_index+1}."
                    state["next_node"] = "ask_question"
                    return state
                # Otherwise, prompt the user with current answer and request new text
                q = state["survey"]["questions"][q_index]
                existing_ans = state.get("answers", {}).get(str(q["id"]), None)
                if existing_ans:
                    # If answered already, show existing and ask for new answer
                    new_doc = update_user_survey_state(state_doc, {"expecting_revision": True, "pending_edit_q": int(qn)})
                    state["state_doc"] = new_doc
                    append_user_conversation(new_doc, "user", msg)
                    state["response"] = f"Current answer for Q{qn}: {existing_ans}\n\nPlease provide the new answer you'd like to save."
                    state["next_node"] = "ask_question"
                    return state
                else:
                    # If not answered, ask for the new answer and mark expecting_revision
                    new_doc = update_user_survey_state(state_doc, {"expecting_revision": True, "pending_edit_q": int(qn)})
                    state["state_doc"] = new_doc
                    append_user_conversation(new_doc, "user", msg)
                    state["response"] = f"Q{qn} hasn't been answered yet. Please provide your answer now."
                    state["next_node"] = "ask_question"
                    return state

        if intent == "goto":
            if not qn:
                state["response"] = "Which question number would you like to go to?"
                state["next_node"] = "ask_question"
                append_user_conversation(state_doc, "user", msg)
                return state
            qidx = max(0, int(qn) - 1)
            if qidx >= len(state["survey"]["questions"]):
                state["response"] = f"There are only {len(state['survey']['questions'])} questions. Please pick a number between 1 and {len(state['survey']['questions'])}."
                state["next_node"] = "ask_question"
                append_user_conversation(state_doc, "user", msg)
                return state
            state["current_index"] = qidx
            new_doc = update_user_survey_state(state_doc, {"current_question_index": state["current_index"], "status": "active"})
            state["state_doc"] = new_doc
            append_user_conversation(new_doc, "user", msg)
            state["response"] = f"Jumping to question {qn}."
            state["next_node"] = "ask_question"
            mirror_to_user_assignment(state["user"]["_id"], state["survey"]["_id"], new_doc)
            return state

        if intent == "skip":
            # move to next unanswered rather than blindly incrementing
            # set a temporary current_index to current+1 and then let _next_unanswered_index find next unanswered
            state["current_index"] = min(state.get("current_index", 0) + 1, len(state["survey"]["questions"]))
            next_idx = self._next_unanswered_index(state)
            if next_idx < len(state["survey"]["questions"]):
                state["current_index"] = next_idx
            else:
                # if none unanswered, set to len (complete sentinel)
                state["current_index"] = len(state["survey"]["questions"])
            new_doc = update_user_survey_state(state_doc, {"current_question_index": state["current_index"], "status": "active"})
            state["state_doc"] = new_doc
            append_user_conversation(new_doc, "user", msg)
            state["response"] = "Skipping this question — moving on."
            state["next_node"] = "ask_question" if state["current_index"] < len(state["survey"]["questions"]) else "complete"
            mirror_to_user_assignment(state["user"]["_id"], state["survey"]["_id"], new_doc)
            return state

        if intent == "back":
            # move to previous question (but prefer previous unanswered if relevant)
            state["current_index"] = max(0, state.get("current_index", 0) - 1)
            new_doc = update_user_survey_state(state_doc, {"current_question_index": state["current_index"], "status": "active"})
            state["state_doc"] = new_doc
            append_user_conversation(new_doc, "user", msg)
            state["response"] = "Going back to the previous question."
            state["next_node"] = "ask_question"
            mirror_to_user_assignment(state["user"]["_id"], state["survey"]["_id"], new_doc)
            return state

        if intent == "review":
            if qn:
                qidx = max(0, int(qn) - 1)
                if qidx >= len(state["survey"]["questions"]):
                    state["response"] = f"There are only {len(state['survey']['questions'])} questions. I can't review question {qn}."
                    state["next_node"] = "ask_question"
                    append_user_conversation(state_doc, "user", msg)
                    return state
                q = state["survey"]["questions"][qidx]
                ans = state.get("answers", {}).get(str(q["id"]), "[Not answered]")
                state["response"] = f"Q{qn}: {q['text']}\nA: {ans}"
                # include similarity and judge if available
                simmap = state.get("answer_similarities", {}) or {}
                sim_entry = simmap.get(str(q["id"]))
                if sim_entry is not None:
                    # handle older float entries gracefully
                    if isinstance(sim_entry, dict):
                        sim = sim_entry.get("similarity")
                        llm_j = sim_entry.get("llm_judge")
                        state["response"] += f"\nSimilarity: {sim}\nJudge: {llm_j}"
                    else:
                        state["response"] += f"\nSimilarity: {sim_entry}"
                state["next_node"] = "ask_question"
                append_user_conversation(state_doc, "user", msg)
                append_user_conversation(state_doc, "agent", state["response"])
                return state
            review = "Here are your answers so far:\n"
            for i, q in enumerate(state["survey"]["questions"]):
                qid = str(q["id"])
                ans = state.get("answers", {}).get(qid, "[Not answered]")
                review += f"\nQ{i+1}: {q['text']}\nA: {ans}\n"
                sim_entry = (state.get("answer_similarities", {}) or {}).get(qid)
                if sim_entry is not None:
                    if isinstance(sim_entry, dict):
                        review += f"Similarity: {sim_entry.get('similarity')}\nJudge: {sim_entry.get('llm_judge')}\n"
                    else:
                        review += f"Similarity: {sim_entry}\n"
            state["response"] = review
            append_user_conversation(state_doc, "user", msg)
            append_user_conversation(state_doc, "agent", review)
            state["next_node"] = "ask_question"
            return state

        if intent == "start":
            state["status"] = "active"
            new_doc = update_user_survey_state(state_doc, {"status": "active", "startedAt": datetime.utcnow()})
            state["state_doc"] = new_doc
            append_user_conversation(new_doc, "user", msg)
            state = self.greet(state)
            append_user_conversation(new_doc, "agent", state["response"])
            state = self.ask_question(state)
            append_user_conversation(new_doc, "agent", state["response"])
            state["next_node"] = "ask_question"
            return state

        if intent == "thanks":
            state["response"] = "You're welcome — let me know if you'd like to review or change any answers."
            append_user_conversation(state_doc, "user", msg)
            append_user_conversation(state_doc, "agent", state["response"])
            state["next_node"] = "ask_question"
            return state

        # default -> treat as answer
        # handle case where we were expecting revision and user supplied the new answer now
        if state_doc.get("expecting_revision", False) and state_doc.get("pending_edit_q"):
            edit_q = state_doc.get("pending_edit_q")
            q_index = max(0, int(edit_q) - 1)
            if q_index < len(state["survey"]["questions"]):
                q = state["survey"]["questions"][q_index]
                needs_correction, feedback, confidence = self._validate_with_groq(q, msg)
                answers = state.get("answers", {}) or {}
                answers[str(q["id"])] = msg
                state["answers"] = answers

                # compute similarity and judge only if expectedAnswer exists
                answer_similarities = state.get("answer_similarities", {}) or {}
                expected = q.get("expectedAnswer", "")
                if expected:
                    sim = compute_cosine_similarity(expected, msg)
                    judge = self._judge_answer_with_groq(q, msg)
                    answer_similarities[str(q["id"])] = {
                        "similarity": sim,
                        "llm_judge": judge
                    }
                    state["answer_similarities"] = answer_similarities

                new_doc = update_user_survey_state(state_doc, {
                    "answers": answers,
                    "answer_similarities": state.get("answer_similarities", {}),
                    "expecting_revision": False,
                    "pending_edit_q": None
                })
                state["state_doc"] = new_doc
                append_user_conversation(new_doc, "user", msg)
                append_user_conversation(new_doc, "agent", f"✅ Updated answer for Q{q_index+1}: {feedback if confidence < 0.5 else 'Updated.'}")
                mirror_to_user_assignment(state["user"]["_id"], state["survey"]["_id"], new_doc)
                state["response"] = f"✅ Updated answer for Q{q_index+1}."
                state["next_node"] = "ask_question"
                return state

        cur_q = state.get("current_question")
        if not cur_q:
            idx = state.get("current_index", 0)
            if idx < len(state["survey"]["questions"]):
                state["current_question"] = state["survey"]["questions"][idx]
                cur_q = state["current_question"]
            else:
                state["response"] = "No more questions left."
                state["next_node"] = "complete"
                return state

        state["candidate_answer"] = msg
        state["candidate_qid"] = str(cur_q["id"])
        state["next_node"] = "validate_answer"
        append_user_conversation(state_doc, "user", msg)
        return state

    def validate_answer(self, state: dict) -> dict:
        question = state.get("current_question")
        answer = state.get("candidate_answer", "")
        if not question:
            state["response"] = "No question to validate."
            state["next_node"] = "ask_question"
            return state

        needs_correction, feedback, confidence = self._validate_with_groq(question, answer)

        if needs_correction and confidence > 0.7:
            state["response"] = f"🤔 {feedback}\n\nCould you please revise your answer?"
            state["next_node"] = "ask_question"
            return state
        else:
            qid = state.get("candidate_qid")
            if qid is not None:
                answers = state.get("answers", {}) or {}
                answers[qid] = answer
                state["answers"] = answers

                # compute similarity and judge only if expectedAnswer exists
                answer_similarities = state.get("answer_similarities", {}) or {}
                expected = question.get("expectedAnswer", "")
                if expected:
                    sim = compute_cosine_similarity(expected, answer)
                    judge = self._judge_answer_with_groq(question, answer)
                    answer_similarities[qid] = {
                        "similarity": sim,
                        "llm_judge": judge
                    }
                    state["answer_similarities"] = answer_similarities

                # Find next unanswered index (if any)
                next_idx = self._next_unanswered_index(state)
                total_q = len(state["survey"]["questions"])
                state_doc = state["state_doc"]

                if next_idx >= total_q:
                    # all answered -> complete
                    state["current_index"] = total_q
                    new_doc = update_user_survey_state(state_doc, {
                        "answers": answers,
                        "answer_similarities": state.get("answer_similarities", {}),
                        "current_question_index": state["current_index"],
                        "status": "completed",
                        "expecting_revision": False,
                        "pending_edit_q": None,
                        "completedAt": datetime.utcnow()
                    })
                    state["state_doc"] = new_doc
                    append_user_conversation(new_doc, "agent", f"✅ Answer recorded! {'' if confidence >= 0.5 else '(Note: partial match)'}")
                    mirror_to_user_assignment(state["user"]["_id"], state["survey"]["_id"], new_doc)
                    state["response"] = "✅ Answer recorded!"
                    state["next_node"] = "complete"
                    return state
                else:
                    # continue at next unanswered question
                    state["current_index"] = next_idx
                    new_doc = update_user_survey_state(state_doc, {
                        "answers": answers,
                        "answer_similarities": state.get("answer_similarities", {}),
                        "current_question_index": state["current_index"],
                        "status": "active",
                        "expecting_revision": False,
                        "pending_edit_q": None
                    })
                    state["state_doc"] = new_doc
                    append_user_conversation(new_doc, "agent", f"✅ Answer recorded! {'' if confidence >= 0.5 else '(Note: partial match)'}")
                    mirror_to_user_assignment(state["user"]["_id"], state["survey"]["_id"], new_doc)
                    state["response"] = "✅ Answer recorded!"
                    state["next_node"] = "ask_question"
                    return state

        # fallback
        state["response"] = "✅ Answer recorded!"
        state["next_node"] = "ask_question"
        return state

    def complete(self, state: dict) -> dict:
        state["status"] = "completed"
        state["response"] = "🎉 Thank you for completing the survey! All your responses have been recorded."
        new_doc = update_user_survey_state(state["state_doc"], {
            "status": "completed",
            "completedAt": datetime.utcnow(),
            "expecting_revision": False,
            "pending_edit_q": None
        })
        append_user_conversation(new_doc, "agent", state["response"])
        mirror_to_user_assignment(state["user"]["_id"], state["survey"]["_id"], new_doc)
        state["state_doc"] = new_doc
        state["next_node"] = None
        return state

    def invoke(self, user_id: str, survey_id: str, user_message: str) -> dict:
        survey = self._get_survey(survey_id)
        if not survey:
            raise HTTPException(404, "Survey not found")

        state_doc = get_or_create_user_survey_state(user_id, survey_id)

        state = {
            "user": db.users.find_one({"_id": user_id}) if db.users.find_one({"_id": user_id}) else {"_id": user_id},
            "user_id": user_id,
            "survey": survey,
            "survey_id": survey_id,
            "state_doc": state_doc,
            "status": state_doc.get("status", "not_started"),
            "current_index": int(state_doc.get("current_question_index", 0) or 0),
            "answers": state_doc.get("answers", {}) or {},
            "answer_similarities": state_doc.get("answer_similarities", {}) or {},
            "user_message": (user_message or "").strip(),
            "next_node": None,
            "_invoked_at": datetime.utcnow()
        }

        incoming = state["user_message"]
        lower_in = incoming.lower()
        # removed empty string trigger to avoid accidental start on empty input
        start_triggers = {"start", "begin", "hi", "hello", "hey"}

        if incoming:
            append_user_conversation(state_doc, "user", incoming)
            state_doc = db.user_survey_state.find_one({"_id": state_doc["_id"]})
            state["state_doc"] = state_doc

        # If survey not started and user explicitly sends a start-like message, return greeting first then question
        if state["status"] == "not_started" and lower_in in start_triggers:
            state["status"] = "active"
            state_doc = update_user_survey_state(state_doc, {"status": "active", "startedAt": datetime.utcnow()})
            state["state_doc"] = state_doc

            # build greeting and first question deterministically and append both (greeting first)
            greeting_state = self.greet(state)
            greeting_text = greeting_state.get("response", "")
            append_user_conversation(state_doc, "agent", greeting_text)

            # ensure we set current_index properly before asking question
            question_state = self.ask_question(state)
            question_text = question_state.get("response", "")
            append_user_conversation(state_doc, "agent", question_text)

            # persist the index/status
            state_doc = update_user_survey_state(state_doc, {"current_question_index": state.get("current_index", 0), "status": "active"})
            mirror_to_user_assignment(user_id, survey_id, state_doc)

            combined = f"{greeting_text}\n\n{question_text}"
            # Return both greeting and question (UI can render as two messages)
            return {
                "response": combined,
                "status": state.get("status", "active"),
                "current_question": state.get("current_question", {}).get("text") if state.get("current_question") else None,
                "current_question_index": state.get("current_index"),
                "progress": (len(state.get("answers", {})) / max(len(survey["questions"]), 1)) * 100,
                "completed": state.get("status") == "completed"
            }

        if state["status"] == "not_started":
            state_doc = update_user_survey_state(state_doc, {"status": "active", "startedAt": datetime.utcnow()})
            state["state_doc"] = state_doc
            state["status"] = "active"

        if not state.get("current_question"):
            qidx = state.get("current_index", 0)
            if qidx < len(survey["questions"]):
                state["current_question"] = survey["questions"][qidx]

        state["next_node"] = "process_answer"

        for _ in range(12):
            node = state.get("next_node")
            if node == "process_answer":
                state = self.process_answer(state)
                continue
            if node == "validate_answer":
                state = self.validate_answer(state)
                continue
            if node == "ask_question":
                state = self.ask_question(state)
                append_user_conversation(state["state_doc"], "agent", state["response"])
                state_doc = update_user_survey_state(state["state_doc"], {
                    "current_question_index": state["current_index"],
                    "status": state.get("status", "active")
                })
                mirror_to_user_assignment(user_id, survey_id, state_doc)
                return {
                    "response": state["response"],
                    "status": state.get("status", "active"),
                    "current_question": state.get("current_question", {}).get("text") if state.get("current_question") else None,
                    "current_question_index": state.get("current_index"),
                    "progress": (len(state.get("answers", {})) / max(len(survey["questions"]), 1)) * 100,
                    "completed": state.get("status") == "completed"
                }
            if node == "complete":
                state = self.complete(state)
                return {
                    "response": state["response"],
                    "status": "completed",
                    "current_question": None,
                    "current_question_index": len(state["survey"]["questions"]),
                    "progress": 100.0,
                    "completed": True
                }
            state["next_node"] = "ask_question"

        return {
            "response": "Sorry — something went wrong in the conversation flow.",
            "status": "error",
            "current_question": None,
            "current_question_index": None,
            "progress": 0.0,
            "completed": False
        }

# instantiate
survey_agent = SurveyAgentGraph(db)

# ========== Mock data ==========
def initialize_mock_data():
    db.users.delete_many({})
    db.surveys.delete_many({})
    db.user_survey_state.delete_many({})

    db.users.create_indexes([IndexModel([("_id", ASCENDING)]), IndexModel([("assignedSurveys.surveyId", ASCENDING)]), IndexModel([("email", ASCENDING)], unique=True)])
    db.surveys.create_indexes([IndexModel([("_id", ASCENDING)]), IndexModel([("title", ASCENDING)])])
    db.user_survey_state.create_indexes([IndexModel([("user_id", ASCENDING), ("survey_id", ASCENDING)])])

    survey_data = {
        "_id": ObjectId("68fcb4979ca9e49998b18cb3"),
        "title": "Prospective Students - Admission Interest Survey",
        "description": "Understand motivations, expectations, and decision factors of applicant",
        "questions": [
            {"id": 1, "text": "What inspired you to consider applying to our university?", "expectedAnswer": "Your strong reputation in engineering and innovation, plus the vibrant campus life"},
            {"id": 2, "text": "Which academic programs or departments interest you most, and why?", "expectedAnswer": "Computer Science and AI - I'm passionate about technology and your department has excellent research facilities"},
            {"id": 3, "text": "How did you first hear about our university?", "expectedAnswer": "Through a college fair at my school and later via your Instagram ads showcasing campus events"},
            {"id": 4, "text": "What factors are most important to you when choosing a university?", "expectedAnswer": "Scholarship opportunities, faculty expertise, internship programs, and campus facilities"},
            {"id": 5, "text": "Do you have any concerns or questions about the admission process?", "expectedAnswer": "I'm unsure about the timeline for scholarship decisions and whether international students have different deadlines"}
        ],
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow()
    }
    db.surveys.insert_one(survey_data)

    user_data = {
        "_id": "user123",
        "name": "Test User",
        "email": "test@example.com",
        "assignedSurveys": [
            {
                "surveyId": ObjectId("68fcb4979ca9e49998b18cb3"),
                "status": "not_started",
                "startedAt": None,
                "completedAt": None,
                "currentQuestionIndex": 0,
                "answers": {},
                # now mapped qid -> { similarity: float, llm_judge: {comment, confidence_score} }
                "answer_similarities": {},
                "conversationTurns": []
            }
        ]
    }
    db.users.insert_one(user_data)
    print("✅ Mock data initialized!")

if USE_MOCK_DATA:
    initialize_mock_data()

# ========== Routes ==========
@app.get("/")
async def root():
    return {"message": "Survey Chat Agent API", "version": "1.0.0", "endpoints": {"chat": "POST /survey/chat", "user_surveys": "GET /users/{user_id}/surveys", "survey_state": "GET /users/{user_id}/surveys/{survey_id}/state", "reset_survey": "POST /users/{user_id}/surveys/{survey_id}/reset"}}

@app.post("/survey/chat", response_model=SurveyResponse)
async def chat_with_survey(message: UserMessage):
    try:
        print(message)
        result = survey_agent.invoke(message.user_id, message.survey_id, message.message)
        return SurveyResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error processing survey: {str(e)}")

@app.get("/users/{user_id}/surveys")
async def get_user_surveys(user_id: str):
    user = db.users.find_one({"_id": user_id})
    if not user:
        raise HTTPException(404, "User not found")
    surveys = []
    for assignment in user.get("assignedSurveys", []):
        survey = db.surveys.find_one({"_id": assignment["surveyId"]})
        if survey:
            progress = (len(assignment.get("answers", {})) / len(survey["questions"])) * 100
            surveys.append({
                "survey_id": str(survey["_id"]),
                "title": survey["title"],
                "description": survey["description"],
                "status": assignment["status"],
                "progress": round(progress, 1),
                "current_question_index": assignment.get("currentQuestionIndex", 0),
                "total_questions": len(survey["questions"]),
                "answered_questions": len(assignment.get("answers", {})),
                "started_at": assignment.get("startedAt"),
                "completed_at": assignment.get("completedAt")
            })
    return surveys

@app.post("/users/{user_id}/surveys/{survey_id}/start", response_model=SurveyResponse)
async def start_survey(user_id: str, survey_id: str):
    survey = db.surveys.find_one({"_id": survey_id})
    if not survey:
        raise HTTPException(404, "Survey not found")

    state_doc = get_or_create_user_survey_state(user_id, survey_id)

    if state_doc.get("status") == "completed":
        return SurveyResponse(
            response="Survey already completed.",
            status="completed",
            current_question=None,
            progress=100.0,
            completed=True
        )

    updates = {"status": "active"}
    if not state_doc.get("startedAt"):
        updates["startedAt"] = datetime.utcnow()
    state_doc = update_user_survey_state(state_doc, updates)

    runtime = {
        "user": db.users.find_one({"_id": user_id}) if db.users.find_one({"_id": user_id}) else {"_id": user_id},
        "user_id": user_id,
        "survey": survey,
        "survey_id": survey_id,
        "state_doc": state_doc,
        "status": state_doc.get("status", "active"),
        "current_index": state_doc.get("current_question_index", 0),
        "answers": state_doc.get("answers", {}) or {},
        "answer_similarities": state_doc.get("answer_similarities", {}) or {}
    }

    agent = survey_agent
    # produce greeting and first question and append both
    runtime = agent.greet(runtime)
    append_user_conversation(state_doc, "agent", runtime["response"])
    runtime = agent.ask_question(runtime)
    append_user_conversation(state_doc, "agent", runtime["response"])

    # persist the index/status
    state_doc = update_user_survey_state(state_doc, {
        "current_question_index": runtime.get("current_index", 0),
        "status": "active"
    })
    mirror_to_user_assignment(user_id, survey_id, state_doc)

    # Return the question text (greeting already appended to conversation, UI can show full history)
    return SurveyResponse(
        response=runtime["response"],
        status="active",
        current_question=runtime.get("current_question", {}).get("text") if runtime.get("current_question") else None,
        progress=(len(runtime.get("answers", {})) / max(len(survey.get("questions", [])), 1)) * 100,
        completed=False
    )

@app.get("/users/{user_id}/surveys/{survey_id}/state")
async def get_survey_state(user_id: str, survey_id: str):
    doc = db.user_survey_state.find_one({"user_id": user_id, "survey_id": survey_id})
    if doc:
        survey = db.surveys.find_one({"_id": survey_id})
        total = len(survey["questions"]) if survey else 1
        answered = len(doc.get("answers", {}) or {})
        idx = int(doc.get("current_question_index", 0) or 0)
        if not survey or idx < 0 or idx >= len(survey.get("questions", [])):
            current_q_text = None
        else:
            current_q_text = survey["questions"][idx]["text"]
        return {
            "user_id": user_id,
            "survey_id": survey_id,
            "survey_title": survey["title"] if survey else None,
            "status": doc.get("status", "not_started"),
            "current_question_index": doc.get("current_question_index", 0),
            "current_question": current_q_text,
            "answers": doc.get("answers", {}),
            "answer_similarities": doc.get("answer_similarities", {}),
            "progress": round((answered / total) * 100, 1),
            "total_questions": total,
            "answered_questions": answered,
            "conversation_turns": doc.get("user_conversations", []),
            "started_at": doc.get("startedAt"),
            "completed_at": doc.get("completedAt")
        }

    user = db.users.find_one({"_id": user_id})
    if not user:
        raise HTTPException(404, "User not found")
    for assignment in user.get("assignedSurveys", []):
        if str(assignment["surveyId"]) == survey_id:
            survey = db.surveys.find_one({"_id": assignment["surveyId"]})
            total = len(survey["questions"]) if survey else 1
            answered = len(assignment.get("answers", {}) or {})
            return {
                "user_id": user_id,
                "survey_id": survey_id,
                "survey_title": survey["title"] if survey else None,
                "status": assignment.get("status", "not_started"),
                "current_question_index": assignment.get("currentQuestionIndex", 0),
                "current_question": survey["questions"][assignment.get("currentQuestionIndex", 0)]["text"] if survey and survey.get("questions") and 0 <= assignment.get("currentQuestionIndex", 0) < len(survey.get("questions", [])) else None,
                "answers": assignment.get("answers", {}),
                "answer_similarities": assignment.get("answer_similarities", {}),
                "progress": round((answered / total) * 100, 1),
                "total_questions": total,
                "answered_questions": answered,
                "conversation_turns": assignment.get("conversationTurns", [])[-6:],
                "started_at": assignment.get("startedAt"),
                "completed_at": assignment.get("completedAt")
            }
    raise HTTPException(404, "Survey not found")

@app.post("/users/{user_id}/surveys/{survey_id}/reset")
async def reset_survey(user_id: str, survey_id: str):
    res = db.user_survey_state.update_one(
        {"user_id": user_id, "survey_id": survey_id},
        {"$set": {"status": "not_started", "current_question_index": 0, "answers": {}, "user_conversations": [], "startedAt": None, "completedAt": None, "expecting_revision": False, "pending_edit_q": None, "answer_similarities": {}, "updatedAt": datetime.utcnow()}}
    )
    db.users.update_one({"_id": user_id, "assignedSurveys.surveyId": survey_id},
                        {"$set": {"assignedSurveys.$.status": "not_started", "assignedSurveys.$.currentQuestionIndex": 0, "assignedSurveys.$.answers": {}, "assignedSurveys.$.answer_similarities": {}, "assignedSurveys.$.conversationTurns": [], "assignedSurveys.$.startedAt": None, "assignedSurveys.$.completedAt": None}})
    if res.matched_count == 0:
        raise HTTPException(404, "Survey not found or already reset")
    return {"message": "Survey reset successfully"}

@app.post("/recommend_questions", response_model=RecommendResponse)
async def recommend_questions(req: RecommendRequest):
    json_schema = {
        "name": "recommend_questions_result",
        "schema": {
            "type": "array",
            "minItems": req.num_questions,
            "maxItems": req.num_questions,
            "items": {
                "type": "object",
                "properties": {
                    "question_title": {"type": "string"},
                    "question_description": {"type": "string"},
                    "expected_answer": {"type": "string"}
                },
                "required": ["question_title", "question_description", "expected_answer"]
            }
        }
    }

    prompt = f"""
        You are an expert survey questionnaire producer. Follow this EXACT ordered plan and return ONLY the final JSON object described under OUTPUT. Do not output any additional text.

        PLAN (must follow exactly):
        1) Read the survey title and description carefully.
        2) Extract {req.num_questions} concise theme phrases that capture the description (each theme 2–6 words).
        3) For each theme, produce:
        - question_title : one-line concise summary of the question's intent (no 'Suggested' prefix).
        - question_description : the actual survey question text ONLY. Start directly with the question and end with a question mark. Do NOT add lead-ins like "Please answer" or any extra wording.
        - expected_answer : a short, model-style sample answer (1–2 sentences) written as a realistic human respondent (first-person when appropriate). This must be an actual sample answer (not meta-instructions).
        4) Enforce these negative constraints: do NOT use the words/phrases "Suggested", "Short free-text", "Please answer", "Please describe", or any instruction-like commentary in any field. Do NOT output meta-comments (e.g., "This is a sample") or placeholders.
        5) Ensure each question_description ends with a single question mark and is grammatically correct.
        6) Output exactly {req.num_questions} question objects. No more, no less.

        OUTPUT (must be the only output — valid JSON object):
        {{
        "themes": ["theme1", "theme2", "theme3", ...],
        "questions": [
            {{
            "question_title": "...",
            "question_description": "...?",
            "expected_answer": "..."
            }},
            ...
        ]
        }}

        FEW-SHOT EXAMPLES (follow these styles exactly):

        Example 1 — Academic / career:
        Input Title: "Student Career Intentions"
        Input Description: "Understand students' plans after graduation, preferred industries, and reasons for their choices."

        Example output (one item):
        {{
        "question_title": "Post-graduation career plans",
        "question_description": "What are your plans after graduation and which industry are you aiming to join?",
        "expected_answer": "I plan to pursue a software engineering role in fintech because I'm passionate about payments and financial inclusion; I'll apply for internships and build related projects."
        }}

        Example 2 — Campus wellbeing:
        Input Title: "Campus Wellbeing"
        Input Description: "Measure students' mental wellbeing, stressors, and use of campus counselling services."

        Example output (one item):
        {{
        "question_title": "Current mental wellbeing and stressors",
        "question_description": "How would you describe your current mental wellbeing and what are the main stressors affecting you?",
        "expected_answer": "Overall I'm managing, but I feel stressed due to upcoming exams and a part-time job. I sometimes struggle with sleep but plan to try the counselling service next semester."
        }}

        Example 3 — Employee engagement:
        Input Title: "Employee Engagement & Satisfaction"
        Input Description: "Gather employee sentiment about recognition, workload, and opportunities for growth."

        Example output (one item):
        {{
        "question_title": "Feelings of recognition and support",
        "question_description": "To what extent do you feel recognized and supported by your manager and team?",
        "expected_answer": "I generally feel recognized for my work through monthly shout-outs, but I would appreciate more constructive feedback to support my career growth."
        }}

        Example 4 — Consumer product feedback:
        Input Title: "Smartphone User Feedback"
        Input Description: "Collect user impressions on battery life, camera quality, and overall value for money."

        Example output (one item):
        {{
        "question_title": "Battery life experience",
        "question_description": "How satisfied are you with the smartphone's battery life during typical daily use?",
        "expected_answer": "The battery lasts a full day for me with moderate use; on heavy days I need to charge in the evening but overall it's acceptable."
        }}

        NOW produce the required JSON object for this survey only (no commentary, no markdown, no explanation). Use the variables below:

        Survey title: {req.title}

        Survey description: {req.description}
    """

    try:
        resp = groq_client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a concise survey question generator."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_schema", "json_schema": json_schema},
        )

        raw = resp.choices[0].message.content or ""
        parsed = extract_json_from_text(raw)

        if not parsed or not isinstance(parsed, list):
            raise ValueError("Invalid JSON returned by LLM")

        questions = []
        for item in parsed[:req.num_questions]:
            qt = item.get("question_title", "").strip()
            qd = item.get("question_description", "").strip()
            ea = item.get("expected_answer", "").strip()
            questions.append({
                "question_title": qt or "Untitled question",
                "question_description": qd or "",
                "expected_answer": ea or ""
            })

        return {"questions": questions}

    except Exception as e:
        fallback_questions = []
        for i in range(1, req.num_questions + 1):
            fallback_questions.append({
                "question_title": f"Suggested question {i} about {req.title}",
                "question_description": f"{req.description.strip()[:200]}",
                "expected_answer": "Short free-text answer (1-2 sentences)."
            })
        return {"questions": fallback_questions}


@app.get("/surveys")
async def get_all_surveys():
    surveys = list(db.surveys.find({}, {"_id": 1, "title": 1, "description": 1, "questions": 1, "createdAt": 1}))
    for s in surveys:
        s["_id"] = str(s["_id"])
        s["total_questions"] = len(s.get("questions", []))
    return surveys

@app.get("/health")
async def health_check():
    try:
        db.command("ping")
        return {"status": "healthy", "database": "connected", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(503, f"Service unhealthy: {str(e)}")
    
# ========== Admin Prev Survey CRUD Routes ==========

@app.post("/surveys")
async def create_survey(survey_data: dict):
    try:
        survey_data["_id"] = str(uuid.uuid4())
        # print( survey_data["_id"] )
        survey_data["createdAt"] = datetime.utcnow()
        survey_data["updatedAt"] = datetime.utcnow()
        
        result = db.surveys.insert_one(survey_data)
    
        reqObj = {
            'survey_id': survey_data["_id"],
            'status': 'not_started',
            'create_state': True    
        }
        print('Trying to append data to user')
        add_survey_to_user("user123", reqObj)
        
        return {"id": str(result.inserted_id), "message": "Survey created successfully"}
    except Exception as e:
        raise HTTPException(500, f"Error creating survey: {str(e)}")
    



@app.put("/surveys/{survey_id}")
async def update_survey(survey_id: str, survey_data: dict):
    try:
        # Remove _id from update data to prevent immutable field error
        if '_id' in survey_data:
            del survey_data['_id']
        
        survey_data["updatedAt"] = datetime.utcnow()
        
        result = db.surveys.update_one(
            {"_id": survey_id},
            {"$set": survey_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(404, "Survey not found")
            
        return {"message": "Survey updated successfully"}
    
       
    except Exception as e:
        raise HTTPException(500, f"Error updating survey: {str(e)}")

@app.delete("/surveys/{survey_id}")
async def delete_survey(survey_id: str):
    try:
        result = db.surveys.delete_one({"_id": survey_id})
        
        if result.deleted_count == 0:
            raise HTTPException(404, "Survey not found")
            
        return {"message": "Survey deleted successfully"}
    except Exception as e:
        raise HTTPException(500, f"Error deleting survey: {str(e)}")

# table analytics page - endpoint
@app.get("/analytics/survey-responses")
async def get_survey_responses():
    # Get all survey states with user and survey data using manual join
    survey_states = list(db.user_survey_state.find())
    
    # Format for frontend table
    analytics_data = []
    for state in survey_states:
        # Get user data
        user = db.users.find_one({"_id": state["user_id"]})
        
        # Get survey data - convert survey_id string to ObjectId
        try:
            survey = db.surveys.find_one({"_id": state["survey_id"]})
        except:
            survey = None
        
        analytics_data.append({
            "user_id": state["user_id"],
            "username": user.get("name", "Unknown") if user else "Unknown",
            "survey_title": survey.get("title", "Unknown") if survey else "Unknown",
            "status": state.get("status", "unknown"),
            "questions": survey.get("questions", []) if survey else [],
            "answers": state.get("answers", {}),
            "similarities": state.get("answer_similarities", {}),
            "confidence_score": state.get("answer_similarities", {})  # This contains the confidence scores
        })
    
    return analytics_data


class AssignSurveyRequest(BaseModel):
    survey_id: str
    status: Optional[str] = "not_started"
    create_state: Optional[bool] =True

def add_survey_to_user(user_id: str, req):
    print(f"Adding survey {req['survey_id']} to user {user_id} with status {req['status']} and create_state={req['create_state']}")
    # validate survey_id
    try:
        survey_obj_id = req['survey_id']
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid survey_id format")
    
    print(1)

    # ensure user exists
    user = db.users.find_one({"_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # ensure survey exists
    survey = db.surveys.find_one({"_id": survey_obj_id})
    if not survey:
        raise HTTPException(status_code=404, detail="Survey not found")

    # prevent duplicate assignment
    already_assigned = False
    for a in user.get("assignedSurveys", []):
        # a["surveyId"] might be an ObjectId or a dict depending on how inserted; handle both
        existing_id = a.get("surveyId")
        if isinstance(existing_id, dict) and "$oid" in existing_id:
            existing_oid = BsonObjectId(existing_id["$oid"])
        else:
            try:
                existing_oid = existing_id if isinstance(existing_id, BsonObjectId) else BsonObjectId(existing_id)
            except Exception:
                existing_oid = None

        if existing_oid == survey_obj_id:
            already_assigned = True
            break

    if already_assigned:
        raise HTTPException(status_code=409, detail="Survey already assigned to user")
    
    print('Im here 1')

    # build assignment object (match the shape you provided)
    assignment = {
        "surveyId": survey_obj_id,
        "status": req['status'],
        "startedAt": None,
        "completedAt": None,
        "currentQuestionIndex": 0,
        "answers": {},
        "answer_similarities": {},
        "conversationTurns": []
    }

    print('Im here 2')

    # push into user's assignedSurveys
    res = db.users.update_one(
        {"_id": user_id},
        {"$push": {"assignedSurveys": assignment}}
    )

    print('Im here 3')

    if res.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to append survey to user")

    # optionally create an initial user_survey_state document so other endpoints can find it
    if req['create_state']:
        state_doc = {
            "user_id": user_id,
            "survey_id": str(survey_obj_id),
            "status": req['status'],
            "current_question_index": 0,
            "answers": {},
            "answer_similarities": {},
            "user_conversations": [],
            "startedAt": None,
            "completedAt": None,
            "expecting_revision": False,
            "pending_edit_q": None,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        # upsert: only create if doesn't exist
        db.user_survey_state.update_one(
            {"user_id": user_id, "survey_id": str(survey_obj_id)},
            {"$setOnInsert": state_doc},
            upsert=True
        )

    # return the newly appended assignment (stringify survey_id for client readability)
    assignment_for_response = assignment.copy()
    assignment_for_response["surveyId"] = str(survey_obj_id)

    return {"message": "Survey assigned to user", "assignment": assignment_for_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
