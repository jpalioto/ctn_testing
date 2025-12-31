"""Reusable UI components for CTN Results Browser."""

import streamlit as st

from ctn_testing.browser.data import (
    JudgingData,
    ResponseData,
    RunSummary,
)


def render_run_info(run: RunSummary, label: str = "") -> None:
    """Render basic run information."""
    prefix = f"{label}: " if label else ""
    st.markdown(f"**{prefix}{run.run_id}**")

    cols = st.columns(4)
    with cols[0]:
        st.metric("Prompts", run.prompts_count)
    with cols[1]:
        st.metric("Constraints", len(run.constraints))
    with cols[2]:
        st.metric("SDK Calls", run.total_sdk_calls)
    with cols[3]:
        st.metric("Judge Calls", run.total_judge_calls)

    if run.strategy:
        st.caption(f"Strategy: `{run.strategy}`")
    if run.run_type == "rejudge":
        st.caption(f"Rejudge of: `{run.source_run_id}`")


def render_response_detail(response: ResponseData) -> None:
    """Render a single response in detail."""
    # Show metadata row
    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.caption(f"Tokens: {response.tokens_in} in / {response.tokens_out} out")
    with meta_cols[1]:
        if response.model:
            st.caption(f"Model: {response.model}")
    with meta_cols[2]:
        if response.provider:
            st.caption(f"Provider: {response.provider}")
    with meta_cols[3]:
        if response.invariant_check:
            if response.invariant_check.kernel_match:
                st.success("Kernel Match: PASS", icon="\u2705")
            else:
                st.error("Kernel Match: FAIL", icon="\u274c")

    if response.error:
        st.error(f"Error: {response.error}")
        return

    # Show kernel info if available
    kernel_text = response.kernel or (response.dry_run.kernel if response.dry_run else "")
    if kernel_text:
        with st.expander("Kernel Used", expanded=False):
            st.code(kernel_text, language="xml")

    # Show dry-run details if available
    if response.dry_run:
        with st.expander("Dry-Run Details", expanded=False):
            if response.dry_run.system_prompt:
                st.markdown("**System Prompt:**")
                st.code(response.dry_run.system_prompt, language="xml")
            if response.dry_run.user_prompt:
                st.markdown("**User Prompt:**")
                st.code(response.dry_run.user_prompt, language=None)
            if response.dry_run.parameters:
                st.markdown("**Parameters:**")
                st.json(response.dry_run.parameters)

    st.markdown("**Prompt:**")
    prompt_text = response.prompt_text or response.input_sent
    st.text(prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text)

    st.markdown("**Input Sent:**")
    st.code(response.input_sent, language=None)

    st.markdown("**Output:**")
    with st.container(height=400):
        st.markdown(response.output)


def render_response_comparison(
    resp_a: ResponseData | None,
    resp_b: ResponseData | None,
    label_a: str = "Run A",
    label_b: str = "Run B",
) -> None:
    """Render side-by-side response comparison."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(label_a)
        if resp_a:
            st.caption(f"Tokens: {resp_a.tokens_in} in / {resp_a.tokens_out} out")
            if resp_a.error:
                st.error(resp_a.error)
            else:
                st.markdown("**Output:**")
                with st.container(height=400):
                    st.markdown(resp_a.output)
        else:
            st.warning("No response found")

    with col2:
        st.subheader(label_b)
        if resp_b:
            st.caption(f"Tokens: {resp_b.tokens_in} in / {resp_b.tokens_out} out")
            if resp_b.error:
                st.error(resp_b.error)
            else:
                st.markdown("**Output:**")
                with st.container(height=400):
                    st.markdown(resp_b.output)
        else:
            st.warning("No response found")


def render_kernel_info(
    run: RunSummary,
    constraint: str,
    responses: list[ResponseData],
) -> None:
    """Render kernel/strategy information for a constraint.

    Shows kernel data from dry-run capture and invariant check results.
    """
    # Find a sample response for this constraint
    sample = next((r for r in responses if r.constraint_name == constraint), None)

    # Find the constraint config
    constraint_config = next((c for c in run.constraint_configs if c.name == constraint), None)

    # Display strategy info
    st.markdown(f"**Strategy:** `{run.strategy or 'unknown'}`")

    # Display constraint config
    if constraint_config:
        st.markdown(f"**Constraint:** `{constraint_config.name}`")
        if constraint_config.input_prefix:
            st.markdown(f"**Input Prefix:** `{constraint_config.input_prefix}`")
        else:
            st.markdown("**Input Prefix:** _(none - baseline)_")
        if constraint_config.description:
            st.markdown(f"**Description:** {constraint_config.description}")
    else:
        st.markdown(f"**Constraint:** `{constraint}`")

    st.markdown("---")

    # Show invariant check status
    if sample and sample.invariant_check:
        if sample.invariant_check.kernel_match:
            st.success("Kernel Match: PASS - Kernel was applied correctly", icon="\u2705")
        else:
            st.error("Kernel Match: FAIL - Kernel mismatch detected", icon="\u274c")
        st.markdown("---")

    # Show kernel from response data (if available from dry-run capture)
    kernel_text = None
    if sample:
        kernel_text = sample.kernel or (sample.dry_run.kernel if sample.dry_run else None)

    if kernel_text:
        st.markdown("**Kernel (from dry-run capture):**")
        st.code(kernel_text, language="xml")
    else:
        # Fallback explanation for older data without dry-run capture
        if run.strategy == "ctn":
            st.info(
                "**CTN Strategy:** The SDK server projects a CTN kernel with trait vectors "
                "based on the constraint prefix. No dry-run capture available for this run."
            )
        elif run.strategy == "operational":
            st.info(
                "**Operational Strategy:** The SDK server applies `<behavioral_constraints>` "
                "XML blocks. No dry-run capture available for this run."
            )
        elif run.strategy == "structural":
            st.info(
                "**Structural Strategy:** The SDK server applies structured constraint rules. "
                "No dry-run capture available for this run."
            )
        else:
            st.warning("No kernel data available. Dry-run capture may not have been enabled.")

    # Show dry-run parameters if available
    if sample and sample.dry_run and sample.dry_run.parameters:
        with st.expander("Dry-Run Parameters"):
            st.json(sample.dry_run.parameters)

    # Show system prompt from dry-run if different from kernel
    if sample and sample.dry_run and sample.dry_run.system_prompt:
        if sample.dry_run.system_prompt != kernel_text:
            with st.expander("System Prompt (from dry-run)"):
                st.code(sample.dry_run.system_prompt, language="xml")

    # Show sample input_sent
    if sample:
        with st.expander("Show Input Sent to Model"):
            st.code(sample.input_sent, language=None)


def render_scores_table(judging: JudgingData) -> None:
    """Render scores comparison table."""
    if not judging.baseline_scores and not judging.test_scores:
        st.info("No scores available")
        return

    # Get all traits
    all_traits = set(judging.baseline_scores.keys()) | set(judging.test_scores.keys())

    # Build table data
    rows = []
    for trait in sorted(all_traits):
        baseline_data = judging.baseline_scores.get(trait, {})
        test_data = judging.test_scores.get(trait, {})

        baseline_score = (
            baseline_data.get("score", 0) if isinstance(baseline_data, dict) else baseline_data
        )
        test_score = test_data.get("score", 0) if isinstance(test_data, dict) else test_data

        delta = test_score - baseline_score
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        delta_color = "green" if delta > 0 else "red" if delta < 0 else "gray"

        rows.append(
            {
                "Trait": trait,
                "Baseline": baseline_score,
                "Test": test_score,
                "Delta": delta_str,
                "_delta_color": delta_color,
            }
        )

    # Display as table
    st.markdown("| Trait | Baseline | Test | Delta |")
    st.markdown("|-------|----------|------|-------|")
    for row in rows:
        delta_html = f":{row['_delta_color']}[{row['Delta']}]"
        st.markdown(f"| {row['Trait']} | {row['Baseline']} | {row['Test']} | {delta_html} |")


def render_scores_detail(judging: JudgingData) -> None:
    """Render detailed scores with reasoning."""
    if not judging.baseline_scores and not judging.test_scores:
        st.info("No scores available")
        return

    all_traits = set(judging.baseline_scores.keys()) | set(judging.test_scores.keys())

    for trait in sorted(all_traits):
        baseline_data = judging.baseline_scores.get(trait, {})
        test_data = judging.test_scores.get(trait, {})

        baseline_score = (
            baseline_data.get("score", 0) if isinstance(baseline_data, dict) else baseline_data
        )
        test_score = test_data.get("score", 0) if isinstance(test_data, dict) else test_data

        with st.expander(f"**{trait}**: Baseline {baseline_score} vs Test {test_score}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Baseline Reasoning:**")
                reasons = (
                    baseline_data.get("reasons", []) if isinstance(baseline_data, dict) else []
                )
                if reasons:
                    for r in reasons:
                        st.markdown(f"- {r}")
                else:
                    st.caption("No reasoning provided")

            with col2:
                st.markdown("**Test Reasoning:**")
                reasons = test_data.get("reasons", []) if isinstance(test_data, dict) else []
                if reasons:
                    for r in reasons:
                        st.markdown(f"- {r}")
                else:
                    st.caption("No reasoning provided")


def render_errors_list(run: RunSummary) -> None:
    """Render list of errors from run."""
    if not run.errors:
        st.success("No errors in this run")
        return

    st.error(f"{len(run.errors)} error(s) occurred during this run")

    for i, error in enumerate(run.errors, 1):
        with st.expander(f"Error {i}"):
            st.code(error, language=None)


def format_run_label(run: RunSummary) -> str:
    """Format run for display in selectbox."""
    strategy_str = f" ({run.strategy})" if run.strategy else ""
    type_str = " [rejudge]" if run.run_type == "rejudge" else ""
    return f"{run.run_id}{strategy_str}{type_str}"
