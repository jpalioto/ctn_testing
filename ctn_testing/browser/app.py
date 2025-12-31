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
    get_unique_constraints,
    get_unique_prompts,
    list_runs,
    load_judgings,
    load_responses,
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
        result = EvaluationResult.load(run.path)
        if result.comparisons:
            analyses = full_analysis(result)
            report = format_report(analyses)
            st.code(report, language=None)
        else:
            st.info("No judging comparisons available for analysis")
    except Exception as e:
        st.warning(f"Could not load analysis: {e}")


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
