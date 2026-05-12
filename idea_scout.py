#!/usr/bin/env python3
"""
idea_scout.py — Research a business idea and add it to the dashboard.

Usage:
    .venv/bin/python3 idea_scout.py "AI bookkeeping tool for freelancers"
    .venv/bin/python3 idea_scout.py           # interactive prompt
"""

import os
import sys
import json
import re
import openai
from ddgs import DDGS

DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "index.html")

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Opportunity Scout research agent. A user has given you a specific business idea to evaluate.

## Your job
1. Run 12-18 targeted web searches to gather real evidence
2. Score the idea on three demand metrics with cited evidence
3. Score it on all 12 feasibility dimensions
4. Call `submit_idea` exactly once at the end with the complete structured evaluation

## Founder profile (anchor your scoring to this)
- Technical: Python, web stacks, AI/ML integration
- Time: 15-25 hrs/week part-time
- Capital: under $5K to start (up to $20K if revenue justifies)
- Domain: general software/AI — NOT a regulated-industry SME
- Revenue target: $500K/year within 12-18 months
- Risk: cannot afford liability exposure, hardware logistics, or 12+ month sales cycles
- Geography: US-focused, English-speaking buyers

## Research sequence (run ALL before submitting)

### Demand Signal research (4-6 searches)
- `site:reddit.com [problem keyword] frustrated OR nightmare`
- `site:reddit.com [target customer] [problem]`
- `site:reddit.com r/[relevant subreddit] [problem]`
- `[problem keyword] community forum complaints`
- `indiehackers.com [topic]`

### Pain Severity research (3-4 searches)
- `"[problem] costs" OR "[problem] expense" annual`
- `"hours spent on [problem]" OR "[problem] takes hours`
- `[industry] [problem] report statistics 2024 OR 2025`
- glassdoor salary for people doing this manually

### Competitive Reality research (4-6 searches)
- `best [solution category] software 2025`
- `[top competitor] pricing`
- `[top competitor] pricing reddit`
- `[top competitor] alternatives`
- `[competitor] crunchbase funding`
- `site:g2.com [competitor] reviews`

## Scoring rubrics

### Demand Signal (1-10)
- 9-10: 100+ Reddit threads (50+ upvotes), 2+ communities 100K+ members, rising trends
- 7-8: 30-100 threads with engagement, 1-2 active 10K+ member communities
- 5-6: 10-30 relevant threads, scattered but real
- 3-4: Few mentions, mostly old threads
- 1-2: No genuine user complaints found

### Pain Severity (1-10)
- 9-10: $10K+ per incident or 10+ hrs/week, regulatory deadlines
- 7-8: $1-10K per incident or 3-10 hrs/week
- 5-6: $100-1K or 1-3 hrs/week
- 3-4: Mild inconvenience, unclear cost
- 1-2: Nice-to-have

### Competitive Reality (1-10)
- 9-10: 0-2 competitors, fragmented, clear gap
- 7-8: 3-5 competitors with gaps
- 5-6: 5-10 competitors, niche gap exists
- 3-4: 1-2 dominant ($10M+ ARR)
- 1-2: $50M+ funded players, network effects

### Composite formula
(Demand × 0.4) + (Pain × 0.4) + (Competitive Reality × 0.2)
- ≥7.5 → gate = "Strong", verdict = "STRONG FIT"
- 6-7.4 → gate = "Caveat", verdict = "CONSIDER WITH CAVEATS"
- <6 → gate = "Reject", verdict = "REJECT"

### Feasibility dimensions (1-10 each)
Build:
1. technical_buildability — MVP in 6-8 weeks part-time?
2. data_feasibility — Data available without a customer?
3. domain_knowledge_gap — Industry expertise needed? (higher = less needed)

Sell:
4. sales_cycle — Days to first payment (higher = faster)
5. buyer_accessibility — Reachable without trade shows?
6. decision_maker_clarity — One clear buyer?

Economics:
7. pricing_power — Can charge $100-500/mo?
8. gross_margin — Net margin after APIs/costs (higher = 60%+)
9. operating_capital — Year-one cost (higher = under $10K)

Risk:
10. liability — Worst-case lawsuit risk (higher = low risk)
11. competitive_intensity — Use Competitive Reality score directly
12. regulatory_drag — Compliance burden (higher = none)

## Auto-reject triggers
- Requires SOC 2/HIPAA/PCI/CMMC for MVP
- Hardware logistics required
- Enterprise procurement committee is the buyer
- Trade shows required for credibility
- Licensed professional required full-time
- Network effects required to be useful
- $5M+ funded incumbents dominating with 3+ year head start

## Rules
- Every score must be backed by at least one real search result
- Quote real users whenever possible; if unverifiable write "Unable to verify"
- Revenue projections must be in ranges, not points
- `is_top_pick` must always be false
- Call `submit_idea` ONCE at the very end after all research is complete
"""

# ─── Tool definitions (OpenAI function-calling format) ────────────────────────

FEASIBILITY_DIM = {
    "type": "object",
    "required": ["score", "note"],
    "properties": {
        "score": {"type": "integer"},
        "note": {"type": "string"},
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web. Use for Reddit threads, competitor pricing, "
                "Crunchbase funding, G2 reviews, industry reports, Glassdoor salaries. "
                "Run 12-18 searches total before submitting."
            ),
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_idea",
            "description": (
                "Submit the fully researched and scored idea. "
                "Call ONCE at the very end after all web searches are complete."
            ),
            "parameters": {
                "type": "object",
                "required": [
                    "id", "name", "pattern", "one_liner", "target_customer",
                    "demand_signal", "pain_severity", "competitive_reality",
                    "composite", "gate", "feasibility",
                    "reddit_quotes", "pain_quantification", "competitors",
                    "underserved_gap", "brutal_truths", "validation_plan",
                    "first_5_conversations", "kill_criteria", "revenue_path",
                    "verdict",
                ],
                "properties": {
                    "id": {"type": "string", "description": "kebab-case id e.g. 'cloud-cost-triage-copilot'"},
                    "name": {"type": "string", "description": "Product name + tagline"},
                    "pattern": {"type": "string", "enum": ["A", "B", "C", "D", "E", "F"],
                                "description": "A=Democratization B=Compliance/paperwork C=Vertical SaaS D=Defensive consumer E=Underserved pros F=Adjacent paperwork"},
                    "one_liner": {"type": "string"},
                    "target_customer": {"type": "string"},
                    "demand_signal": {
                        "type": "object", "required": ["score", "rationale"],
                        "properties": {"score": {"type": "integer"}, "rationale": {"type": "string"}},
                    },
                    "pain_severity": {
                        "type": "object", "required": ["score", "rationale"],
                        "properties": {"score": {"type": "integer"}, "rationale": {"type": "string"}},
                    },
                    "competitive_reality": {
                        "type": "object", "required": ["score", "rationale"],
                        "properties": {"score": {"type": "integer"}, "rationale": {"type": "string"}},
                    },
                    "composite": {"type": "number"},
                    "gate": {"type": "string", "enum": ["Strong", "Caveat", "Reject"]},
                    "feasibility": {
                        "type": "object",
                        "required": [
                            "technical_buildability", "data_feasibility", "domain_knowledge_gap",
                            "sales_cycle", "buyer_accessibility", "decision_maker_clarity",
                            "pricing_power", "gross_margin", "operating_capital",
                            "liability", "competitive_intensity", "regulatory_drag", "average",
                        ],
                        "properties": {
                            "technical_buildability": FEASIBILITY_DIM,
                            "data_feasibility": FEASIBILITY_DIM,
                            "domain_knowledge_gap": FEASIBILITY_DIM,
                            "sales_cycle": FEASIBILITY_DIM,
                            "buyer_accessibility": FEASIBILITY_DIM,
                            "decision_maker_clarity": FEASIBILITY_DIM,
                            "pricing_power": FEASIBILITY_DIM,
                            "gross_margin": FEASIBILITY_DIM,
                            "operating_capital": FEASIBILITY_DIM,
                            "liability": FEASIBILITY_DIM,
                            "competitive_intensity": FEASIBILITY_DIM,
                            "regulatory_drag": FEASIBILITY_DIM,
                            "average": {"type": "number"},
                        },
                    },
                    "reddit_quotes": {
                        "type": "array",
                        "items": {
                            "type": "object", "required": ["quote", "source", "url", "upvotes", "year"],
                            "properties": {
                                "quote": {"type": "string"},
                                "source": {"type": "string"},
                                "url": {"type": "string"},
                                "upvotes": {"type": "string"},
                                "year": {"type": "string"},
                            },
                        },
                    },
                    "pain_quantification": {
                        "type": "object",
                        "required": ["dollar_cost", "time_cost", "urgency_triggers", "visceral_quote"],
                        "properties": {
                            "dollar_cost": {
                                "type": "object", "required": ["value", "source_url"],
                                "properties": {"value": {"type": "string"}, "source_url": {"type": "string"}},
                            },
                            "time_cost": {
                                "type": "object", "required": ["value", "source_url"],
                                "properties": {"value": {"type": "string"}, "source_url": {"type": "string"}},
                            },
                            "urgency_triggers": {"type": "array", "items": {"type": "string"}},
                            "visceral_quote": {
                                "type": "object", "required": ["quote", "source_url"],
                                "properties": {"quote": {"type": "string"}, "source_url": {"type": "string"}},
                            },
                        },
                    },
                    "competitors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "funding", "est_revenue", "pricing", "top_complaint", "wedge"],
                            "properties": {
                                "name": {"type": "string"},
                                "funding": {"type": "string"},
                                "est_revenue": {"type": "string"},
                                "pricing": {"type": "string"},
                                "top_complaint": {"type": "string"},
                                "wedge": {"type": "string"},
                            },
                        },
                    },
                    "underserved_gap": {"type": "string"},
                    "brutal_truths": {
                        "type": "object",
                        "required": ["certain", "uncertain", "kills_it", "underestimated"],
                        "properties": {
                            "certain": {"type": "array", "items": {"type": "string"}},
                            "uncertain": {"type": "array", "items": {"type": "string"}},
                            "kills_it": {"type": "array", "items": {"type": "string"}},
                            "underestimated": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "validation_plan": {
                        "type": "array",
                        "items": {
                            "type": "object", "required": ["week", "action"],
                            "properties": {"week": {"type": "string"}, "action": {"type": "string"}},
                        },
                    },
                    "first_5_conversations": {
                        "type": "array",
                        "items": {
                            "type": "object", "required": ["who", "where_to_find"],
                            "properties": {"who": {"type": "string"}, "where_to_find": {"type": "string"}},
                        },
                    },
                    "kill_criteria": {"type": "array", "items": {"type": "string"}},
                    "revenue_path": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["STRONG FIT", "CONSIDER WITH CAVEATS", "REJECT"]},
                },
            },
        },
    },
]

# ─── Web search ───────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 8) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.get('title', '')}\n"
                f"URL: {r.get('href', '')}\n"
                f"{r.get('body', '')}\n"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"

# ─── Dashboard injection ──────────────────────────────────────────────────────

def load_dashboard_data(html: str) -> dict:
    m = re.search(r"const DATA = (\{.*?\});\s*\n", html, re.DOTALL)
    if not m:
        raise ValueError("Could not find 'const DATA = {...}' in dashboard.html")
    return json.loads(m.group(1))


def save_dashboard_data(html: str, data: dict) -> str:
    new_json = json.dumps(data, indent=2, ensure_ascii=False)
    result = re.sub(
        r"const DATA = \{.*?\};\s*\n",
        lambda _: f"const DATA = {new_json};\n",
        html,
        flags=re.DOTALL,
    )
    if result == html:
        raise ValueError("Regex replacement had no effect — DATA block not found.")
    return result


def inject_into_dashboard(idea: dict) -> None:
    with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    data = load_dashboard_data(html)

    # Replace if id already exists (idempotent)
    data["ideas"] = [i for i in data["ideas"] if i["id"] != idea["id"]]
    idea["is_top_pick"] = False
    data["ideas"].append(idea)

    # Sort by composite descending
    data["ideas"].sort(key=lambda x: x["composite"], reverse=True)

    # Recompute top pick
    passing = [i for i in data["ideas"] if i["gate"] in ("Strong", "Caveat")]
    if passing:
        best = max(passing, key=lambda x: (x["composite"], x.get("feasibility", {}).get("average", 0)))
        for i in data["ideas"]:
            i["is_top_pick"] = i["id"] == best["id"]

    # Update summary
    data["research_summary"]["candidates_evaluated"] = len(data["ideas"])
    data["research_summary"]["gate_passed"] = sum(1 for i in data["ideas"] if i["gate"] in ("Strong", "Caveat"))
    data["research_summary"]["gate_failed"] = sum(1 for i in data["ideas"] if i["gate"] == "Reject")

    html = save_dashboard_data(html, data)

    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    top = next((i for i in data["ideas"] if i["is_top_pick"]), None)
    print(f"\n[dashboard] Added '{idea['name']}' — {len(data['ideas'])} ideas total.")
    if top:
        print(f"[dashboard] Top pick is now: {top['name']}")

# ─── Agent loop ───────────────────────────────────────────────────────────────

def run(idea_prompt: str) -> None:
    client = openai.OpenAI()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Research and evaluate this business idea:\n\n**{idea_prompt}**\n\n"
                "Run all required web searches, score the three demand metrics and all 12 "
                "feasibility dimensions with cited evidence, then call submit_idea with "
                "the complete structured evaluation."
            ),
        },
    ]

    search_count = 0
    submitted_idea = None

    print(f"\n{'─' * 64}")
    print(f"  IDEA SCOUT  |  Researching: {idea_prompt}")
    print(f"{'─' * 64}\n")

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=16000,
        )

        msg = response.choices[0].message
        messages.append(msg)

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "stop":
            if msg.content:
                print(msg.content)
            # Model described the evaluation in text but forgot to call the tool — force it
            if not submitted_idea:
                print("\n[forcing submit_idea call...]", flush=True)
                messages.append({
                    "role": "user",
                    "content": "Now call submit_idea with the complete structured evaluation you just produced. Do not output text — call the tool directly.",
                })
                forced = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice={"type": "function", "function": {"name": "submit_idea"}},
                    max_tokens=16000,
                )
                forced_msg = forced.choices[0].message
                if forced_msg.tool_calls:
                    for tc in forced_msg.tool_calls:
                        if tc.function.name == "submit_idea":
                            submitted_idea = json.loads(tc.function.arguments)
                            print(f"[submit_idea] Received: {submitted_idea.get('name', '?')}")
            break

        elif finish_reason == "tool_calls":
            if msg.content:
                print(msg.content, end="", flush=True)

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if name == "web_search":
                    search_count += 1
                    q = args.get("query", "")
                    print(f"\n[search {search_count}] {q}", flush=True)
                    result = web_search(q)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                elif name == "submit_idea":
                    submitted_idea = args
                    print(f"\n[submit_idea] Received: {submitted_idea.get('name', '?')}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Idea received. Writing to dashboard.",
                    })

            if submitted_idea:
                break

        else:
            if msg.content:
                print(msg.content)
            break

    print(f"\n{'─' * 64}")
    print(f"  Done. {search_count} web searches executed.")
    print(f"{'─' * 64}")

    if submitted_idea:
        inject_into_dashboard(submitted_idea)
        print(f"\nRefresh:  http://localhost:8080/dashboard.html\n")
    else:
        print("\nWarning: agent did not call submit_idea. Dashboard not updated.\n")

# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if len(sys.argv) > 1:
        idea = " ".join(sys.argv[1:])
    else:
        print("Idea Scout — Enter the business idea you want researched:\n")
        idea = input("> ").strip()
        if not idea:
            print("No idea provided. Exiting.")
            sys.exit(0)

    run(idea)


if __name__ == "__main__":
    main()
