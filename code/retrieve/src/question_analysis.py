import json
import re
import time
import urllib.error
import urllib.request


SYSTEM_PROMPT = """You analyze knowledge graph question answering questions.
Return only one valid JSON object. Do not include markdown, comments, or prose.

The analysis must be useful for graph retrieval. Prefer short canonical phrases
such as "spouse", "place of birth", "film director", "located in", "parent
organization", "date of birth", "award won", "contains", "part of".

Do not answer the question. Do not invent concrete answer entities.
Use unknown when the question does not reveal a field.
"""


COMPACT_SYSTEM_PROMPT = """You extract a retrieval plan for knowledge graph question answering.
Return exactly one valid JSON object and no prose.

For normal questions, relation_intents and reasoning_steps must not be empty.
Break multi-hop questions into ordered steps. Use short canonical relation phrases:
"mascot of", "team won championship", "date of event", "located in",
"contains", "borders", "leader of", "language spoken", "currency used",
"educated at", "performer", "country of origin", "religion".

Do not answer the question. Do not invent final answer entities.
"""


SCHEMA_HINT = {
    "topic_entities": ["entity explicitly mentioned by the question"],
    "answer_type": {
        "coarse": "person | place | organization | time | number | event | work | entity | boolean | unknown",
        "fine": "more specific expected answer type, e.g. city, spouse, film, date",
        "surface": "question phrase that implies the answer type",
        "confidence": 0.0,
    },
    "relation_intents": [
        {
            "intent": "canonical relation phrase",
            "aliases": ["alternative wording"],
            "source_hint": "entity/variable this relation starts from",
            "target_type": "expected target type",
            "confidence": 0.0,
        }
    ],
    "reasoning_steps": [
        {
            "step": 1,
            "source": "topic entity or variable name",
            "relation_intent": "canonical relation phrase",
            "target": "new variable name or unknown",
            "target_type": "expected target type",
            "depends_on": [],
            "optional": False,
            "evidence_text": "question words supporting this step",
            "confidence": 0.0,
        }
    ],
    "constraints": [
        {
            "type": "entity_constraint | answer_type | temporal | numeric | ordinal | comparison | qualifier | aggregation | negation",
            "value": "constraint value",
            "operator": "equals | before | after | during | first | last | largest | smallest | count | contains | unknown",
            "scope": "step number, variable, or global",
            "evidence_text": "question words supporting this constraint",
        }
    ],
    "intermediate_variables": [
        {
            "name": "variable name, e.g. spouse",
            "type": "entity type",
            "introduced_by_step": 1,
            "depends_on": [],
        }
    ],
    "dependencies": [
        {
            "from_step": 1,
            "to_step": 2,
            "type": "sequential | filter | aggregation | comparison",
        }
    ],
    "hop_count_estimate": 1,
    "requires_multi_hop": False,
    "notes": "short retrieval-oriented note",
}


REQUIRED_TOP_LEVEL_KEYS = [
    "topic_entities",
    "answer_type",
    "relation_intents",
    "reasoning_steps",
    "constraints",
    "intermediate_variables",
    "dependencies",
    "hop_count_estimate",
    "requires_multi_hop",
    "notes",
]


def _join_nonempty(parts, sep="; "):
    return sep.join(str(part).strip() for part in parts if str(part).strip())


def render_question_analysis_text(analysis):
    """Serialize structured question analysis into retrieval-intent text."""
    answer_type = _as_dict(analysis.get("answer_type"))
    topic_entities = _as_list(analysis.get("topic_entities"))
    relation_intents = _as_list(analysis.get("relation_intents"))
    reasoning_steps = _as_list(analysis.get("reasoning_steps"))
    constraints = _as_list(analysis.get("constraints"))

    answer_text = _join_nonempty([
        answer_type.get("coarse", ""),
        answer_type.get("fine", ""),
        answer_type.get("surface", ""),
    ], sep=" / ")
    relation_text = _join_nonempty(
        [
            _join_nonempty(
                [
                    _as_dict(item).get("intent", ""),
                    _join_nonempty(_as_list(_as_dict(item).get("aliases")), sep=", "),
                    _as_dict(item).get("target_type", ""),
                ],
                sep=" -> ",
            )
            for item in relation_intents
        ]
    )
    step_text = _join_nonempty(
        [render_reasoning_step_text(item, answer_type, constraints) for item in reasoning_steps]
    )
    constraint_text = _join_nonempty(
        [
            _join_nonempty(
                [
                    _as_dict(item).get("type", ""),
                    _as_dict(item).get("operator", ""),
                    _as_dict(item).get("value", ""),
                    _as_dict(item).get("evidence_text", ""),
                ],
                sep=" ",
            )
            for item in constraints
        ]
    )

    return _join_nonempty(
        [
            f"question: {analysis.get('question', '')}",
            f"topic entities: {_join_nonempty(topic_entities, sep=', ')}",
            f"answer type: {answer_text}",
            f"relation intents: {relation_text}",
            f"reasoning chain: {step_text}",
            f"constraints: {constraint_text}",
        ],
        sep="\n",
    )


def render_reasoning_step_text(step, answer_type=None, constraints=None):
    """Serialize one reasoning step so its embedding can condition one DDE hop."""
    step = _as_dict(step)
    answer_type = _as_dict(answer_type)
    constraints = constraints or []
    step_id = str(step.get("step", "")).strip()
    scoped_constraints = []
    for item in constraints:
        item = _as_dict(item)
        scope = str(item.get("scope", "")).lower()
        if scope in {"global", step_id, f"step {step_id}"}:
            scoped_constraints.append(
                _join_nonempty(
                    [
                        item.get("type", ""),
                        item.get("operator", ""),
                        item.get("value", ""),
                        item.get("evidence_text", ""),
                    ],
                    sep=" ",
                )
            )

    return _join_nonempty(
        [
            f"step {step_id}",
            f"source {step.get('source', '')}",
            f"relation {step.get('relation_intent', '')}",
            f"target type {step.get('target_type', '')}",
            f"evidence {step.get('evidence_text', '')}",
            f"answer type {_join_nonempty([answer_type.get('coarse', ''), answer_type.get('fine', '')], sep=' / ')}",
            f"constraints {_join_nonempty(scoped_constraints)}",
        ],
        sep="; ",
    )


def build_question_analysis_messages(question, topic_entities=None):
    topic_entities = topic_entities or []
    user_payload = {
        "question": question,
        "known_topic_entities": topic_entities,
        "output_schema": SCHEMA_HINT,
        "instructions": [
            "Use known_topic_entities as hints, but include only entities that are relevant to the question.",
            "Break implicit multi-hop reasoning into ordered reasoning_steps.",
            "List each relation intent once in relation_intents, using canonical phrases that can match KG relation names.",
            "Include constraints that should affect graph retrieval, including answer type, temporal, ordinal, numeric, and comparison constraints.",
            "Use variables for intermediate entities when the question requires multi-hop reasoning.",
            "Set confidence between 0 and 1.",
        ],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def build_compact_question_analysis_messages(question, topic_entities=None):
    topic_entities = topic_entities or []
    examples = [
        {
            "question": "Where was Barack Obama's wife born?",
            "known_topic_entities": ["Barack Obama"],
            "output": {
                "topic_entities": ["Barack Obama"],
                "answer_type": {
                    "coarse": "place",
                    "fine": "city",
                    "surface": "Where",
                    "confidence": 0.95,
                },
                "relation_intents": [
                    {
                        "intent": "spouse",
                        "aliases": ["wife", "married to"],
                        "source_hint": "Barack Obama",
                        "target_type": "person",
                        "confidence": 0.95,
                    },
                    {
                        "intent": "place of birth",
                        "aliases": ["born in", "birthplace"],
                        "source_hint": "wife",
                        "target_type": "city",
                        "confidence": 0.95,
                    },
                ],
                "reasoning_steps": [
                    {
                        "step": 1,
                        "source": "Barack Obama",
                        "relation_intent": "spouse",
                        "target": "wife",
                        "target_type": "person",
                        "depends_on": [],
                        "optional": False,
                        "evidence_text": "Obama's wife",
                        "confidence": 0.95,
                    },
                    {
                        "step": 2,
                        "source": "wife",
                        "relation_intent": "place of birth",
                        "target": "answer",
                        "target_type": "city",
                        "depends_on": [1],
                        "optional": False,
                        "evidence_text": "Where was ... born",
                        "confidence": 0.95,
                    },
                ],
                "constraints": [
                    {
                        "type": "answer_type",
                        "value": "city",
                        "operator": "equals",
                        "scope": "global",
                        "evidence_text": "Where",
                    }
                ],
                "intermediate_variables": [
                    {
                        "name": "wife",
                        "type": "person",
                        "introduced_by_step": 1,
                        "depends_on": [],
                    }
                ],
                "dependencies": [
                    {"from_step": 1, "to_step": 2, "type": "sequential"}
                ],
                "hop_count_estimate": 2,
                "requires_multi_hop": True,
                "notes": "Find spouse, then spouse birthplace.",
            },
        },
        {
            "question": "What country bordering France contains an airport that serves Nijmegen?",
            "known_topic_entities": ["Nijmegen", "France"],
            "output": {
                "topic_entities": ["Nijmegen", "France"],
                "answer_type": {
                    "coarse": "place",
                    "fine": "country",
                    "surface": "What country",
                    "confidence": 1.0,
                },
                "relation_intents": [
                    {
                        "intent": "serves",
                        "aliases": ["served by"],
                        "source_hint": "Nijmegen",
                        "target_type": "airport",
                        "confidence": 0.9,
                    },
                    {
                        "intent": "located in",
                        "aliases": ["contained by", "in"],
                        "source_hint": "airport",
                        "target_type": "country",
                        "confidence": 0.9,
                    },
                    {
                        "intent": "borders",
                        "aliases": ["bordering", "shares border with"],
                        "source_hint": "country",
                        "target_type": "country",
                        "confidence": 0.9,
                    },
                ],
                "reasoning_steps": [
                    {
                        "step": 1,
                        "source": "Nijmegen",
                        "relation_intent": "serves",
                        "target": "airport",
                        "target_type": "airport",
                        "depends_on": [],
                        "optional": False,
                        "evidence_text": "airport that serves Nijmegen",
                        "confidence": 0.9,
                    },
                    {
                        "step": 2,
                        "source": "airport",
                        "relation_intent": "located in",
                        "target": "country",
                        "target_type": "country",
                        "depends_on": [1],
                        "optional": False,
                        "evidence_text": "country contains an airport",
                        "confidence": 0.9,
                    },
                    {
                        "step": 3,
                        "source": "country",
                        "relation_intent": "borders",
                        "target": "France",
                        "target_type": "country",
                        "depends_on": [2],
                        "optional": False,
                        "evidence_text": "bordering France",
                        "confidence": 0.9,
                    },
                ],
                "constraints": [
                    {
                        "type": "answer_type",
                        "value": "country",
                        "operator": "equals",
                        "scope": "global",
                        "evidence_text": "What country",
                    },
                    {
                        "type": "entity_constraint",
                        "value": "France",
                        "operator": "equals",
                        "scope": "step 3",
                        "evidence_text": "bordering France",
                    },
                ],
                "intermediate_variables": [
                    {
                        "name": "airport",
                        "type": "airport",
                        "introduced_by_step": 1,
                        "depends_on": [],
                    },
                    {
                        "name": "country",
                        "type": "country",
                        "introduced_by_step": 2,
                        "depends_on": [1],
                    },
                ],
                "dependencies": [
                    {"from_step": 1, "to_step": 2, "type": "sequential"},
                    {"from_step": 2, "to_step": 3, "type": "filter"},
                ],
                "hop_count_estimate": 3,
                "requires_multi_hop": True,
                "notes": "Find airport serving Nijmegen, country containing it, then require border with France.",
            },
        },
    ]
    user_payload = {
        "required_keys": REQUIRED_TOP_LEVEL_KEYS,
        "examples": examples,
        "task": {
            "question": question,
            "known_topic_entities": topic_entities,
            "instruction": (
                "Return the same JSON schema as examples. "
                "Do not leave relation_intents or reasoning_steps empty."
            ),
        },
    }
    return [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def build_json_repair_messages(raw_text):
    return [
        {
            "role": "system",
            "content": (
                "You repair invalid JSON. Return only one valid JSON object. "
                "Do not add markdown, comments, explanations, or new facts. "
                "Preserve the original keys and values as much as possible."
            ),
        },
        {
            "role": "user",
            "content": (
                "Repair this invalid JSON into one syntactically valid JSON object:\n"
                f"{raw_text}"
            ),
        },
    ]


def extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(text[start:end + 1])


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return min(max(value, 0.0), 1.0)


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0", ""}:
            return False
    return bool(value)


def normalize_question_analysis(raw, question, sample_id=None, known_topic_entities=None):
    raw = _as_dict(raw)
    if isinstance(raw.get("output"), dict):
        raw = raw["output"]
    normalized = {key: raw.get(key) for key in REQUIRED_TOP_LEVEL_KEYS}

    normalized["id"] = sample_id
    normalized["question"] = question
    normalized["topic_entities"] = [
        str(x).strip() for x in _as_list(normalized["topic_entities"]) if str(x).strip()
    ]
    if not normalized["topic_entities"] and known_topic_entities:
        normalized["topic_entities"] = [str(x).strip() for x in known_topic_entities if str(x).strip()]

    answer_type = _as_dict(normalized["answer_type"])
    normalized["answer_type"] = {
        "coarse": str(answer_type.get("coarse", "unknown")).strip() or "unknown",
        "fine": str(answer_type.get("fine", "unknown")).strip() or "unknown",
        "surface": str(answer_type.get("surface", "")).strip(),
        "confidence": _as_float(answer_type.get("confidence", 0.0)),
    }

    relation_intents = []
    for item in _as_list(normalized["relation_intents"]):
        item = _as_dict(item)
        intent = str(item.get("intent", "")).strip()
        if not intent:
            continue
        relation_intents.append({
            "intent": intent,
            "aliases": [str(x).strip() for x in _as_list(item.get("aliases")) if str(x).strip()],
            "source_hint": str(item.get("source_hint", "unknown")).strip() or "unknown",
            "target_type": str(item.get("target_type", "unknown")).strip() or "unknown",
            "confidence": _as_float(item.get("confidence", 0.0)),
        })
    normalized["relation_intents"] = relation_intents

    reasoning_steps = []
    for idx, item in enumerate(_as_list(normalized["reasoning_steps"]), start=1):
        item = _as_dict(item)
        relation_intent = str(item.get("relation_intent", "")).strip()
        if not relation_intent:
            continue
        reasoning_steps.append({
            "step": _as_int(item.get("step", idx), idx),
            "source": str(item.get("source", "unknown")).strip() or "unknown",
            "relation_intent": relation_intent,
            "target": str(item.get("target", "unknown")).strip() or "unknown",
            "target_type": str(item.get("target_type", "unknown")).strip() or "unknown",
            "depends_on": [_as_int(x) for x in _as_list(item.get("depends_on"))],
            "optional": _as_bool(item.get("optional", False)),
            "evidence_text": str(item.get("evidence_text", "")).strip(),
            "confidence": _as_float(item.get("confidence", 0.0)),
        })
    normalized["reasoning_steps"] = reasoning_steps

    constraints = []
    for item in _as_list(normalized["constraints"]):
        item = _as_dict(item)
        value = str(item.get("value", "")).strip()
        if not value:
            continue
        constraints.append({
            "type": str(item.get("type", "qualifier")).strip() or "qualifier",
            "value": value,
            "operator": str(item.get("operator", "unknown")).strip() or "unknown",
            "scope": str(item.get("scope", "global")).strip() or "global",
            "evidence_text": str(item.get("evidence_text", "")).strip(),
        })
    normalized["constraints"] = constraints

    variables = []
    for item in _as_list(normalized["intermediate_variables"]):
        item = _as_dict(item)
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        variables.append({
            "name": name,
            "type": str(item.get("type", "unknown")).strip() or "unknown",
            "introduced_by_step": _as_int(item.get("introduced_by_step", 0), 0),
            "depends_on": [_as_int(x) for x in _as_list(item.get("depends_on"))],
        })
    normalized["intermediate_variables"] = variables

    dependencies = []
    for item in _as_list(normalized["dependencies"]):
        item = _as_dict(item)
        dependencies.append({
            "from_step": _as_int(item.get("from_step", 0), 0),
            "to_step": _as_int(item.get("to_step", 0), 0),
            "type": str(item.get("type", "sequential")).strip() or "sequential",
        })
    normalized["dependencies"] = [
        dep for dep in dependencies if dep["from_step"] > 0 and dep["to_step"] > 0
    ]

    hop_count = _as_int(normalized["hop_count_estimate"], len(reasoning_steps))
    normalized["hop_count_estimate"] = max(hop_count, len(reasoning_steps), 0)
    normalized["requires_multi_hop"] = (
        _as_bool(normalized["requires_multi_hop"]) or normalized["hop_count_estimate"] > 1
    )
    normalized["notes"] = str(normalized["notes"] or "").strip()
    return normalized


def call_chat_with_retry(
    client,
    model,
    messages,
    temperature,
    max_tokens,
    max_retries,
    retry_sleep,
    use_response_format=True,
):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            request = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if use_response_format:
                request["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request)
            return response.choices[0].message.content
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (2 ** attempt))
    raise last_error


def call_openai_compatible_http_with_retry(
    api_key,
    base_url,
    model,
    messages,
    temperature,
    max_tokens,
    max_retries,
    retry_sleep,
    use_response_format=True,
    timeout=120,
):
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
            data = json.loads(response_body)
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {error_body}")
        except Exception as exc:
            last_error = exc

        if attempt >= max_retries:
            break
        time.sleep(retry_sleep * (2 ** attempt))

    raise last_error
