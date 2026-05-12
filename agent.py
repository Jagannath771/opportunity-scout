#!/usr/bin/env python3
"""Opportunity Scout — evidence-based founder-fit research agent."""

import os
import sys
import json
import openai
from ddgs import DDGS

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
# Opportunity Scout: Complete Framework

Every recommendation requires "receipts": cited evidence of demand from Reddit/forums,
quantified pain in dollars/hours, and verified competitor pricing/revenue.
No demand signal = no recommendation.

## Default Founder Profile
- Technical: Python, web stacks, AI/ML integration
- Time: 15-25 hrs/week part-time
- Capital: under $5K to start, up to $20K if revenue justifies
- Domain: general software/AI, NOT regulated-industry SME
- Revenue target: $500K/year within 12-18 months
- Risk: cannot afford liability, hardware logistics, or 12+ month sales cycles
- Geography: US-focused, English-speaking buyers

## Three-Metric Demand Validation

### Metric 1: Demand Signal (1-10)
Research: Reddit (site:reddit.com searches), industry forums, Google Trends, HN/IH.
- 9-10: 100+ threads (50+ upvotes), 2+ communities 100K+ members
- 7-8: 30-100 threads, 1-2 active 10K+ member communities
- 5-6: 10-30 relevant threads, scattered
- 3-4: Few old mentions
- 1-2: No genuine complaints

### Metric 2: Pain Severity (1-10)
Research: dollar costs, hours/week, emotional intensity, urgency triggers.
- 9-10: $10K+/incident or 10+ hrs/week, regulatory deadlines
- 7-8: $1-10K or 3-10 hrs/week
- 5-6: $100-1K or 1-3 hrs/week
- 3-4: Mild inconvenience
- 1-2: Nice-to-have

### Metric 3: Competitive Reality (1-10)
Research: competitor pricing pages, Crunchbase funding, G2 reviews, "alternatives" threads.
- 9-10: 0-2 competitors, fragmented, clear gap
- 7-8: 3-5 competitors with gaps
- 5-6: 5-10 competitors, niche gap
- 3-4: 1-2 dominant ($10M+ ARR)
- 1-2: $50M+ funded, network effects

### Composite = (Demand×0.4) + (Pain×0.4) + (Comp×0.2)
- ≥7.5: Strong → proceed to feasibility
- 6-7.4: Caveat → proceed with flags
- <6: Reject

## 12-Dimension Feasibility (after gate passes)
Build: technical_buildability, data_feasibility, domain_knowledge_gap
Sell: sales_cycle, buyer_accessibility, decision_maker_clarity
Economics: pricing_power, gross_margin, operating_capital
Risk: liability, competitive_intensity, regulatory_drag

## Output Format
1. Research summary (candidates, searches, gate results)
2. Per-candidate: demand quotes, pain $, competitor table, composite score
3. Feasibility 12-dim table (gate-passers only)
4. Head-to-head comparison
5. Top recommendation + 90-day plan + 5 customer conversations + kill criteria
6. Risks and uncertainties

## Honesty Rules
- Cite every claim with a source
- Revenue in ranges not points
- Name founder blind spots directly

Your job: find verified demand, manageable competition, execution feasibility within constraints.
Every recommendation answers: "Show me the receipts."
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for Reddit threads, competitor pricing, Crunchbase funding, "
                "G2/Capterra reviews, industry reports, Glassdoor salaries, Google Trends data."
            ),
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string", "description": "The search query"}},
            },
        },
    }
]

# ─── Search ───────────────────────────────────────────────────────────────────

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

# ─── Agent loop ───────────────────────────────────────────────────────────────

def run(query: str) -> None:
    client = openai.OpenAI()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    search_count = 0

    print(f"\n{'─' * 64}")
    print("  OPPORTUNITY SCOUT  |  Evidence-Based Research Agent")
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
            break

        elif finish_reason == "tool_calls":
            if msg.content:
                print(msg.content, end="", flush=True)

            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                search_count += 1
                q = args.get("query", "")
                print(f"\n[search {search_count}] {q}", flush=True)
                result = web_search(q)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            if msg.content:
                print(msg.content)
            break

    print(f"\n{'─' * 64}")
    print(f"  Done. {search_count} web searches executed.")
    print(f"{'─' * 64}\n")

# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Opportunity Scout — What should I research?")
        print("(Press Enter for a full 5-idea scan)\n")
        query = input("> ").strip()
        if not query:
            query = (
                "Find me 5 business opportunities matching my founder profile. "
                "Run the full three-metric demand validation on each. "
                "Apply the demand gate. Score survivors on 12 feasibility dimensions. "
                "Pick the best one. Show me the receipts."
            )

    run(query)


if __name__ == "__main__":
    main()
