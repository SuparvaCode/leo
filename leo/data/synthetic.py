"""Code-generated decision families with labels computed by code, so they are always correct.

* field_reference: structured JSON state, questions that point at fields by backtick path or by key.
* policy_check: a written policy plus case facts; code decides whether the policy applies.
* unknowable: the deciding fact is missing, so the target is uniform (teaches honest uncertainty).
* conversation: an array of messages; questions about what the customer asked for.

Rule: no family may be modelled on evaluation items (JevBench, the held-out four). New families are
designed from task descriptions and public training sources only.
"""
from __future__ import annotations

import random
from typing import Any

from leo.data.augment import choice_question, noul_question, qids

NAMES = ["Ana Silva", "Ben Okafor", "Chen Wei", "Dana Novak", "Emil Berg", "Fatima Khan", "Goran Petrov", "Hana Sato",
         "Ivan Reyes", "Julia Meyer", "Kofi Mensah", "Lena Costa", "Marco Rossi", "Nora Lind", "Omar Haddad", "Priya Nair"]
STATUSES = ["pending", "processing", "shipped", "delivered", "cancelled", "returned"]
ITEMS = ["headphones", "desk lamp", "running shoes", "coffee grinder", "backpack", "monitor", "phone case", "blender"]
PLANS = ["free", "basic", "pro", "enterprise"]
COUNTRIES = ["Germany", "Brazil", "India", "Japan", "Canada", "Kenya", "France", "Mexico", "Australia", "Poland"]
PAYMENTS = ["credit card", "PayPal", "bank transfer", "gift card"]


def _req(source: str, state: Any, qs: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    return {"source": source, "state": state, "questions": dict(zip(qids(len(qs), rng), qs))}


def field_reference(rng: random.Random) -> dict[str, Any]:
    n = rng.randint(2, 6)
    orders = [{"id": f"A-{rng.randint(1000, 9999)}", "item": rng.choice(ITEMS), "status": rng.choice(STATUSES),
               "quantity": rng.randint(1, 4), "paid": rng.random() < 0.7} for _ in range(n)]
    customer = {"name": rng.choice(NAMES), "plan": rng.choice(PLANS), "country": rng.choice(COUNTRIES),
                "verified": rng.random() < 0.6}
    state = {"customer": customer, "orders": orders}
    if rng.random() < 0.3:
        state["notes"] = rng.choice(["Customer called twice last week.", "VIP account.", "Prefers email contact.", ""])
    i = rng.randrange(n)
    ref = f"`orders[{i}]`" if rng.random() < 0.6 else f"order {orders[i]['id']}"
    qs = [choice_question(f"What is the status of {ref}?", STATUSES, orders[i]["status"], rng, allow_letters=False, p_none=0.0)]
    if rng.random() < 0.6:
        j = rng.randrange(n)
        refj = f"`orders[{j}].paid`" if rng.random() < 0.5 else f"order {orders[j]['id']}"
        qs.append(noul_question(f"Has {refj} been paid?" if "order " in refj else f"Is {refj} true?", orders[j]["paid"]))
    if rng.random() < 0.5:
        plan = customer["plan"] if rng.random() < 0.5 else rng.choice(PLANS)
        qs.append(noul_question(f"Is the customer on the {plan} plan?", customer["plan"] == plan))
    if rng.random() < 0.4:
        qs.append(choice_question("Which item is in `orders[%d]`?" % i, ITEMS, orders[i]["item"], rng, max_options=6, p_none=0.1))
    if rng.random() < 0.3:
        qs.append(noul_question("Is `customer.verified` true?", customer["verified"]))
    return _req("syn_field_reference", state, qs, rng)


CATEGORIES = ["electronics", "clothing", "groceries", "furniture", "books", "toys"]
CONDITIONS = ["unused", "opened", "damaged"]
TIERS = ["Free", "Basic", "Gold", "Platinum"]
REGIONS = ["EU", "US", "APAC", "LATAM"]


def _join(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


def policy_check(rng: random.Random) -> dict[str, Any]:
    kind = rng.choice(["returns", "support", "shipping"])
    if kind == "returns":
        cats = rng.sample(CATEGORIES, rng.randint(1, 3))
        conds = rng.sample(CONDITIONS, rng.randint(1, 2))
        policy = f"Returns are accepted for {_join(cats)} items only when the item is {_join(conds)}."
        case = {"category": rng.choice(CATEGORIES), "condition": rng.choice(CONDITIONS)}
        ok = case["category"] in cats and case["condition"] in conds
        action = "accept a return for"
    elif kind == "support":
        tiers = rng.sample(TIERS, rng.randint(1, 2))
        regions = rng.sample(REGIONS, rng.randint(1, 2))
        mode = rng.choice(["and", "or"])
        if mode == "and":
            policy = f"Phone support is available to {_join(tiers)} members located in {_join(regions)}."
        else:
            policy = f"Phone support is available to {_join(tiers)} members, and to every customer located in {_join(regions)}."
        case = {"tier": rng.choice(TIERS), "region": rng.choice(REGIONS)}
        ok = (case["tier"] in tiers and case["region"] in regions) if mode == "and" else (case["tier"] in tiers or case["region"] in regions)
        action = "offer phone support to"
    else:
        regions = rng.sample(REGIONS, rng.randint(1, 3))
        excluded = rng.sample(CATEGORIES, rng.randint(1, 2))
        policy = f"Free shipping applies to orders shipped to {_join(regions)}, except for {_join(excluded)}."
        case = {"destination": rng.choice(REGIONS), "category": rng.choice(CATEGORIES)}
        ok = case["destination"] in regions and case["category"] not in excluded
        action = "give free shipping to"
    if rng.random() < 0.5:
        state: Any = {"policy": policy, "case": case}
        instr = f"Does `policy` allow us to {action} `case`?"
    else:
        facts = "; ".join(f"{k}: {v}" for k, v in case.items())
        state = f"Policy: {policy}\nCase: {facts}"
        instr = f"Does the policy allow us to {action} this case?"
    if rng.random() < 0.7:
        q = noul_question(instr, ok)
    else:
        q = {"type": "choice", "instructions": instr,
             "criteria": {"approve": "the policy allows it", "deny": "the policy does not allow it"},
             "label": "approve" if ok else "deny"}
    return _req("syn_policy", state, [q], rng)


def unknowable(rng: random.Random) -> dict[str, Any]:
    customer = {"name": rng.choice(NAMES), "plan": rng.choice(PLANS)}
    state = {"customer": customer, "last_order": {"item": rng.choice(ITEMS), "status": rng.choice(STATUSES)}}
    r = rng.random()
    if r < 0.4:
        q = noul_question(rng.choice(["Does the customer's email address end in .org?", "Is the customer older than 40?",
                                      "Did the customer pay with a credit card?"]), 0.5)
    elif r < 0.8:
        opts = rng.sample(COUNTRIES, 4)
        q = {"type": "choice", "instructions": "Which country does the customer live in?",
             "criteria": {c: None for c in opts}, "label": {c: 0.25 for c in opts}}
    else:
        q = {"type": "choice", "instructions": "Which payment method was used for `last_order`?",
             "criteria": {p: None for p in PAYMENTS}, "label": {p: 0.25 for p in PAYMENTS}}
    qs = [q]
    if rng.random() < 0.5:  # pair with an answerable question so the state is not a giveaway
        qs.append(choice_question("What is the status of `last_order`?", STATUSES, state["last_order"]["status"], rng,
                                  allow_letters=False, p_none=0.0))
    return _req("syn_unknowable", state, qs, rng)


INTENT_LINES = {
    "refund": ["I want my money back for this.", "Please refund the charge.", "Can I get a refund?"],
    "cancel": ["Please cancel my subscription.", "I'd like to close my account.", "Cancel my plan, thanks."],
    "track": ["Where is my package?", "My order hasn't arrived yet, where is it?", "Can you check the delivery status?"],
    "change_address": ["I moved, please update my shipping address.", "Can you send it to my new address instead?"],
    "question": ["Do you ship to Canada?", "What sizes does this come in?", "Is this compatible with my laptop?"],
}
FILLER = ["Hi there.", "Thanks for the quick reply.", "Order number is A-%d." , "I have been a customer for years.", "Hello!"]


def conversation(rng: random.Random) -> dict[str, Any]:
    intent = rng.choice(list(INTENT_LINES))
    msgs = []
    for _ in range(rng.randint(0, 2)):
        f = rng.choice(FILLER)
        msgs.append({"from": "customer", "text": f % rng.randint(1000, 9999) if "%d" in f else f})
        msgs.append({"from": "agent", "text": rng.choice(["How can I help?", "Could you share more details?", "Sure, one moment."])})
    msgs.append({"from": "customer", "text": rng.choice(INTENT_LINES[intent])})
    agent_offered_refund = rng.random() < 0.25
    if agent_offered_refund:
        msgs.insert(max(0, len(msgs) - 1), {"from": "agent", "text": "I can offer you a full refund if you like."})
    state = {"messages": msgs}
    last = len(msgs) - 1
    qs = [choice_question(f"What does the customer want in `messages[{last}].text`?", list(INTENT_LINES), intent, rng,
                          descs={"refund": "money back", "cancel": "end the subscription or account",
                                 "track": "find out where an order is", "change_address": "update delivery address",
                                 "question": "a product or policy question"}, p_none=0.0)]
    if rng.random() < 0.5:
        qs.append(noul_question("Does any customer message ask for a refund?", intent == "refund"))
    if rng.random() < 0.4:
        qs.append(noul_question("Has the agent offered a refund?", agent_offered_refund))
    return _req("syn_conversation", state, qs, rng)


FAMILIES = {"syn_field_reference": (field_reference, 1800), "syn_policy": (policy_check, 1800),
            "syn_unknowable": (unknowable, 500), "syn_conversation": (conversation, 900)}
