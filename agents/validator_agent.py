import os
from openai import OpenAI
from graph.state import GraphState

def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

def validate_analysis(analysis_result):
    client = get_client()
    
    metrics = analysis_result.get("extracted_metrics", {})
    ratios = analysis_result.get("derived_ratios", {})
    missing = analysis_result.get("missing_metrics", [])
    
    # 1. Deterministic Rule Engine (RBI/SEBI Style)
    rule_flags = []
    
    # Rule 1: Debt to Equity
    de_ratio = ratios.get("debt_to_equity")
    if de_ratio is not None:
        if de_ratio > 2.33:
             # RBI typically allows up to 2:1, but 2.33 is often a tolerance limit for infrastructure
            rule_flags.append(f"🔴 High Risk: Debt-to-Equity is {de_ratio} (Exceeds RBI Standard Norm of 2.0)")
        elif de_ratio > 1.5:
             # A bit high but acceptable
            rule_flags.append(f"🟡 Medium Risk: Debt-to-Equity is {de_ratio} (Watchlist per SEBI Guidelines)")
        else:
            rule_flags.append(f"🟢 Low Risk: Debt-to-Equity is {de_ratio} (Compliant with Standard Norms)")
    else:
        rule_flags.append("⚪ Unknown: Debt-to-Equity data missing")

    # Rule 2: Interest Coverage
    ic_ratio = ratios.get("interest_coverage")
    if ic_ratio is not None:
        if ic_ratio < 1.5:
             # RBI typically wants > 2.0
            rule_flags.append(f"🔴 High Risk: Interest Coverage is {ic_ratio} (Below RBI Recommended Min of 1.5)")
        elif ic_ratio < 2.5:
            rule_flags.append(f"🟡 Medium Risk: Interest Coverage is {ic_ratio}")
        else:
            rule_flags.append(f"🟢 Low Risk: Interest Coverage is {ic_ratio} (Healthy per Guidelines)")
    else:
         rule_flags.append("⚪ Unknown: Interest Coverage data missing")

    # 2. Confidence Scoring
    total_expected = 6 # metrics
    found_count = total_expected - len(missing)
    confidence_score = round(found_count / total_expected, 2)
    
    confidence_level = "High" if confidence_score > 0.8 else "Medium" if confidence_score > 0.5 else "Low"

    # 3. LLM Explanation (Qualitative)
    prompt = f"""
You are a financial compliance assistant.

I have performed a deterministic check of the financial data.
Your job is to explain the context of these findings.

Data Found:
{metrics}

Ratios Calculated:
{ratios}

Automated Risk Flags:
{rule_flags}

Confidence Score: {confidence_score} ({confidence_level})
Missing Data: {missing}

Task:
- Summarize the automated risk flags.
- If data is missing, recommend what documents are needed.
- Provide a final qualitative assessment.
"""

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
        "rule_engine_flags": rule_flags,
        "confidence_score": confidence_score,
        "assessment": response.choices[0].message.content
    }
