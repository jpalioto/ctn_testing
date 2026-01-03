"""CTN Testing Results Browser - Main Streamlit App."""

import sys
from pathlib import Path

import streamlit as st

from ctn_testing.browser.components import (
    format_run_label,
    render_errors_list,
    render_kernel_info,
    render_response_comparison,
    render_response_detail,
    render_run_info,
    render_scores_detail,
    render_scores_table,
)
from ctn_testing.browser.data import (
    ResponseData,
    RunSummary,
    SingleScoreData,
    get_analysis_summary,
    get_unique_constraints,
    get_unique_prompts,
    is_single_score_run,
    list_runs,
    load_judgings,
    load_responses,
    load_single_scores,
)
from ctn_testing.runners.evaluation import EvaluationResult
from ctn_testing.statistics.constraint_analysis import format_report, full_analysis


@st.cache_data
def cached_list_runs(results_dir: str) -> list[RunSummary]:
    """Cached run listing."""
    return list_runs(Path(results_dir))


@st.cache_data
def cached_load_responses(run_path: str):
    """Cached response loading."""
    return load_responses(Path(run_path))


@st.cache_data
def cached_load_judgings(run_path: str):
    """Cached judging loading."""
    return load_judgings(Path(run_path))


@st.cache_data
def cached_load_single_scores(run_path: str):
    """Cached single-score loading."""
    return load_single_scores(Path(run_path))


@st.cache_data
def cached_get_analysis_summary(run_path: str):
    """Cached analysis summary loading."""
    return get_analysis_summary(Path(run_path))


def render_summary_tab(
    run_a: RunSummary,
    run_b: RunSummary | None,
    prompt_id: str | None,
    constraint: str | None,
) -> None:
    """Render summary tab with statistical analysis."""
    # Show filter info if applied
    if prompt_id or constraint:
        filter_parts = []
        if prompt_id:
            filter_parts.append(f"Prompt: `{prompt_id}`")
        if constraint:
            filter_parts.append(f"Constraint: `{constraint}`")
        st.caption(f"Filtered by: {', '.join(filter_parts)}")

    if run_b:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Run A")
            render_run_info(run_a)
            _render_analysis(run_a)
        with col2:
            st.subheader("Run B")
            render_run_info(run_b)
            _render_analysis(run_b)
    else:
        render_run_info(run_a)
        _render_analysis(run_a)


def _render_analysis(run: RunSummary) -> None:
    """Render statistical analysis for a run."""
    try:
        # Check if this is a single-score run
        summary = cached_get_analysis_summary(str(run.path))

        if summary.get("run_type") == "single_score":
            # Render single-score summary
            _render_single_score_analysis(run, summary)
        else:
            # Render comparison analysis
            result = EvaluationResult.load(run.path)
            if result.comparisons:
                analyses = full_analysis(result)
                report = format_report(analyses)
                st.code(report, language=None)
            else:
                st.info("No judging comparisons available for analysis")
    except Exception as e:
        st.warning(f"Could not load analysis: {e}")


def _render_single_score_analysis(run: RunSummary, summary: dict) -> None:
    """Render analysis for single-score (baseline-only) runs."""
    st.subheader("Single-Response Scoring Summary")

    st.metric("Prompts Scored", summary.get("prompts_count", 0))
    st.caption(f"Constraint: {summary.get('constraint', 'baseline')}")

    single_scores = summary.get("single_scores", {})
    if not single_scores:
        st.info("No single-score data available")
        return

    # Build DataFrame for display
    import pandas as pd

    rows = []
    for trait, stats in sorted(single_scores.items()):
        rows.append(
            {
                "Trait": trait,
                "Mean": f"{stats.get('mean', 0):.1f}",
                "Std": f"{stats.get('std', 0):.1f}",
                "Min": stats.get("min", 0),
                "Max": stats.get("max", 0),
                "N": stats.get("n", 0),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_responses_tab(
    run_a: RunSummary,
    run_b: RunSummary | None,
    prompt_id: str | None,
    constraint: str | None,
    responses_a: list[ResponseData],
) -> None:
    """Render responses tab."""
    if not responses_a:
        st.warning("No responses found in Run A")
        return

    if not prompt_id or not constraint:
        st.info("Select a prompt and constraint from the sidebar to view responses.")
        return

    # Find the specific response
    resp_a = next(
        (r for r in responses_a if r.prompt_id == prompt_id and r.constraint_name == constraint),
        None,
    )

    if run_b:
        responses_b = cached_load_responses(str(run_b.path))
        resp_b = next(
            (
                r
                for r in responses_b
                if r.prompt_id == prompt_id and r.constraint_name == constraint
            ),
            None,
        )
        render_response_comparison(
            resp_a,
            resp_b,
            label_a=f"Run A ({run_a.strategy or 'unknown'})",
            label_b=f"Run B ({run_b.strategy or 'unknown'})",
        )
    else:
        if resp_a:
            render_response_detail(resp_a)
        else:
            st.warning(f"No response found for {prompt_id} × {constraint}")


def render_kernels_tab(
    run_a: RunSummary,
    run_b: RunSummary | None,
    constraint: str | None,
    responses_a: list[ResponseData],
) -> None:
    """Render kernels tab - shows strategy and constraint configuration."""
    if not constraint:
        st.info("Select a constraint from the sidebar to view kernel information.")
        return

    if run_b:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Run A ({run_a.strategy or 'unknown'})")
            render_kernel_info(run_a, constraint, responses_a)
        with col2:
            responses_b = cached_load_responses(str(run_b.path))
            st.subheader(f"Run B ({run_b.strategy or 'unknown'})")
            render_kernel_info(run_b, constraint, responses_b)
    else:
        render_kernel_info(run_a, constraint, responses_a)


def render_scores_tab(
    run_a: RunSummary,
    run_b: RunSummary | None,
    prompt_id: str | None,
    constraint: str | None,
) -> None:
    """Render scores tab with judging results."""
    # Check if this is a single-score run
    if is_single_score_run(run_a.path):
        _render_single_scores_tab(run_a, run_b, prompt_id, constraint)
        return

    # Original pairwise comparison logic
    judgings_a = cached_load_judgings(str(run_a.path))

    if not judgings_a:
        st.warning("No judging results found")
        return

    if not prompt_id or not constraint:
        st.info("Select a prompt and constraint from the sidebar to view scores.")
        return

    # Scores are for test constraints (not baseline)
    if constraint == "baseline":
        st.info("Select a test constraint (not baseline) to view comparison scores.")
        return

    # Find matching judging
    judging_a = next(
        (j for j in judgings_a if j.prompt_id == prompt_id and j.test_constraint == constraint),
        None,
    )

    if run_b:
        judgings_b = cached_load_judgings(str(run_b.path))
        judging_b = next(
            (j for j in judgings_b if j.prompt_id == prompt_id and j.test_constraint == constraint),
            None,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Run A ({run_a.strategy or 'unknown'})")
            if judging_a:
                render_scores_table(judging_a)
                render_scores_detail(judging_a)
            else:
                st.warning("No judging found for this selection")

        with col2:
            st.subheader(f"Run B ({run_b.strategy or 'unknown'})")
            if judging_b:
                render_scores_table(judging_b)
                render_scores_detail(judging_b)
            else:
                st.warning("No judging found for this selection")
    else:
        if judging_a:
            render_scores_table(judging_a)
            st.markdown("---")
            render_scores_detail(judging_a)

            # Show responses
            with st.expander("Show Responses"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Baseline Response:**")
                    st.markdown(judging_a.baseline_response)
                with col2:
                    st.markdown("**Test Response:**")
                    st.markdown(judging_a.test_response)

            # Show raw judge response
            if judging_a.raw_response:
                with st.expander("Show Raw Judge Response"):
                    st.code(judging_a.raw_response, language="json")
        else:
            st.warning(f"No judging found for {prompt_id} × {constraint}")


def _render_single_scores_tab(
    run_a: RunSummary,
    run_b: RunSummary | None,
    prompt_id: str | None,
    constraint: str | None,
) -> None:
    """Render scores tab for single-score (baseline-only) runs."""
    import pandas as pd

    single_scores_a = cached_load_single_scores(str(run_a.path))

    if not single_scores_a:
        st.warning("No single-response scores found")
        return

    # If no prompt selected, show all scores in a table
    if not prompt_id:
        st.subheader("All Response Scores")

        # Build rows for DataFrame
        rows = []
        for score in single_scores_a:
            row = {
                "Prompt": score.prompt_id[:20] + "..."
                if len(score.prompt_id) > 20
                else score.prompt_id
            }
            for trait, data in sorted(score.scores.items()):
                if isinstance(data, dict):
                    row[trait[:12]] = data.get("score", 0)
                else:
                    row[trait[:12]] = data
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        return

    # Find specific score
    score_a = next(
        (s for s in single_scores_a if s.prompt_id == prompt_id),
        None,
    )

    if run_b and is_single_score_run(run_b.path):
        single_scores_b = cached_load_single_scores(str(run_b.path))
        score_b = next(
            (s for s in single_scores_b if s.prompt_id == prompt_id),
            None,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Run A ({run_a.strategy or 'unknown'})")
            if score_a:
                _render_single_score_detail(score_a)
            else:
                st.warning("No score found for this prompt")

        with col2:
            st.subheader(f"Run B ({run_b.strategy or 'unknown'})")
            if score_b:
                _render_single_score_detail(score_b)
            else:
                st.warning("No score found for this prompt")
    else:
        if score_a:
            _render_single_score_detail(score_a)
        else:
            st.warning(f"No score found for {prompt_id}")


def _render_single_score_detail(score: SingleScoreData) -> None:
    """Render detailed view of a single-response score."""
    import pandas as pd

    st.markdown(f"**Prompt:** {score.prompt_text[:100]}...")

    # Build scores table
    rows = []
    for trait, data in sorted(score.scores.items()):
        if isinstance(data, dict):
            rows.append(
                {
                    "Trait": trait,
                    "Score": data.get("score", 0),
                    "Reasons": ", ".join(data.get("reasons", [])) if data.get("reasons") else "",
                }
            )
        else:
            rows.append({"Trait": trait, "Score": data, "Reasons": ""})

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Show response
    with st.expander("Show Response"):
        st.markdown(score.response)

    # Show raw judge response
    if score.raw_response:
        with st.expander("Show Raw Judge Response"):
            st.code(score.raw_response, language="json")


def render_errors_tab(run_a: RunSummary, run_b: RunSummary | None) -> None:
    """Render errors tab."""
    if run_b:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Run A Errors")
            render_errors_list(run_a)
        with col2:
            st.subheader("Run B Errors")
            render_errors_list(run_b)
    else:
        render_errors_list(run_a)


def main():
    """Main application entry point."""
    st.set_page_config(
        layout="wide",
        page_title="CTN Results Browser",
        page_icon=":bar_chart:",
    )

    # Get results dir from command line or default
    if len(sys.argv) > 1 and not sys.argv[-1].startswith("-"):
        default_dir = sys.argv[-1]
    else:
        default_dir = "domains/constraint_adherence/results"

    # Sidebar: folder and run selection
    with st.sidebar:
        st.title("CTN Results Browser")

        results_dir = st.text_input("Results folder", default_dir)

        runs = cached_list_runs(results_dir)

        if not runs:
            st.warning(f"No runs found in: {results_dir}")
            st.info("Make sure the folder contains run directories with manifest.json files.")
            return

        st.success(f"Found {len(runs)} run(s)")

        st.subheader("Select Run")
        run_a = st.selectbox(
            "Run A",
            runs,
            format_func=format_run_label,
        )

        compare_mode = st.checkbox("Compare with another run", value=False)

        if compare_mode and len(runs) > 1:
            # Default to second run for comparison
            default_idx = 1 if len(runs) > 1 else 0
            run_b = st.selectbox(
                "Run B",
                runs,
                format_func=format_run_label,
                index=default_idx,
            )
        else:
            run_b = None

        # Load responses for filter options
        responses_a: list[ResponseData] = []
        prompts: list[tuple[str, str]] = []
        constraints: list[str] = []

        if run_a:
            responses_a = cached_load_responses(str(run_a.path))
            if responses_a:
                prompts = get_unique_prompts(responses_a)
                constraints = get_unique_constraints(responses_a)

        # Global filters
        st.markdown("---")
        st.subheader("Filter")

        selected_prompt = None
        selected_constraint = None

        if prompts:
            selected_prompt = st.selectbox(
                "Prompt",
                [None] + prompts,
                format_func=lambda x: "All prompts"
                if x is None
                else (f"{x[0]}: {x[1][:40]}..." if len(x[1]) > 40 else f"{x[0]}: {x[1]}"),
            )

        if constraints:
            selected_constraint = st.selectbox(
                "Constraint",
                [None] + constraints,
                format_func=lambda x: "All constraints" if x is None else x,
            )

        # Extract IDs
        prompt_id = selected_prompt[0] if selected_prompt else None
        constraint = selected_constraint

        st.markdown("---")
        st.caption("Refresh data:")
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.rerun()

    # Main content area
    if run_a is None:
        st.info("Select a run from the sidebar to begin.")
        return

    # Tab navigation
    tabs = st.tabs(
        [
            "Summary",
            "Responses",
            "Kernels",
            "Scores",
            "Errors",
        ]
    )

    with tabs[0]:
        render_summary_tab(run_a, run_b, prompt_id, constraint)

    with tabs[1]:
        render_responses_tab(run_a, run_b, prompt_id, constraint, responses_a)

    with tabs[2]:
        render_kernels_tab(run_a, run_b, constraint, responses_a)

    with tabs[3]:
        render_scores_tab(run_a, run_b, prompt_id, constraint)

    with tabs[4]:
        render_errors_tab(run_a, run_b)


if __name__ == "__main__":
    main()
