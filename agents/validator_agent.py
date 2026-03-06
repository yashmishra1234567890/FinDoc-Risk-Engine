def validate_analysis(analysis_result):
    """
    Applies RBI/SEBI norms to financial data and assigns a confidence score using pure Python (No LLM call).
    """
    metrics = analysis_result.get("extracted_metrics", {})
    ratios = analysis_result.get("derived_ratios", {})
    missing = analysis_result.get("missing_metrics", [])
    
    # 1. Deterministic Rule Engine (Review against Industry Standards)
    rule_flags = []
    
    # Check Debt-to-Equity (Standard Benchmark: 2.0)
    de_ratio = ratios.get("debt_to_equity")
    if de_ratio is not None:
        if de_ratio > 2.33:
            rule_flags.append(f"🔴 High Risk: Debt-to-Equity is {de_ratio} (Exceeds RBI Standard Norm of 2.0)")
        elif de_ratio > 1.5:
            rule_flags.append(f"🟡 Medium Risk: Debt-to-Equity is {de_ratio} (Watchlist per SEBI Guidelines)")
        else:
            rule_flags.append(f"🟢 Low Risk: Debt-to-Equity is {de_ratio} (Compliant with Standard Norms)")
    else:
        rule_flags.append("⚪ Unknown: Debt-to-Equity data missing")

    # Check Interest Coverage (Standard Benchmark: >1.5)
    ic_ratio = ratios.get("interest_coverage")
    if ic_ratio is not None:
        if ic_ratio < 1.5:
            rule_flags.append(f"🔴 High Risk: Interest Coverage is {ic_ratio} (Below RBI Recommended Min of 1.5)")
        elif ic_ratio < 2.5:
            rule_flags.append(f"🟡 Medium Risk: Interest Coverage is {ic_ratio}")
        else:
            rule_flags.append(f"🟢 Low Risk: Interest Coverage is {ic_ratio} (Healthy per Guidelines)")
    else:
         rule_flags.append("⚪ Unknown: Interest Coverage data missing")

    # 2. Confidence Scoring
    total_expected = 6 
    found_count = total_expected - len(missing)
    confidence_score = round(found_count / total_expected, 2)

    # 3. Fast Python Assessment (Removed LLM call for speed)
    assessment = "Automated Python Validation: "
    if rule_flags:
        assessment += " | ".join(rule_flags)
    else:
        assessment += "No specific risk flags triggered."

    return {
        "rule_engine_flags": rule_flags,
        "confidence_score": confidence_score,
        "assessment": assessment
    }
