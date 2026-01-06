"""
AFCAT 2026 AI Dashboard
=======================
Interactive dashboard to view AI-generated questions and predictions.
Run with: streamlit run dashboard_app.py
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Configure page
st.set_page_config(
    page_title="AFCAT 2026 AI Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data functions
@st.cache_data
def load_mock_test() -> Dict[str, Any]:
    """Load AI-generated mock test."""
    try:
        with open(ROOT / "output" / "predictions_2026" / "ai_mock_test_2026.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Mock test file not found. Run the question generator first.")
        return {}

@st.cache_data
def load_predicted_questions() -> Dict[str, Any]:
    """Load AI-predicted questions."""
    try:
        with open(ROOT / "output" / "predictions_2026" / "ai_predicted_questions.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Predicted questions file not found. Run the question generator first.")
        return {}

def display_question_card(question: Dict[str, Any], index: int):
    """Display a single question in a card format."""
    with st.container():
        # Question header
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"**Q{index + 1}:** {question['question_text']}")
        with col2:
            st.caption(f"Topic: {question.get('topic', 'N/A').replace('_', ' ').title()}")
        with col3:
            difficulty = question.get('predicted_difficulty', 'medium')
            color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "🟡")
            st.caption(f"{color} {difficulty.title()}")

        # Options
        if 'options' in question and question['options']:
            options = question['options']
            cols = st.columns(len(options))
            for i, option in enumerate(options):
                with cols[i]:
                    if option == question.get('correct_answer'):
                        st.success(f"**{chr(65+i)}) {option}** ✓")
                    else:
                        st.info(f"{chr(65+i)}) {option}")

        # Metadata
        with st.expander("📊 Question Details", expanded=False):
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.write(f"**Section:** {question.get('section', 'N/A').title()}")
            with meta_col2:
                st.write(f"**Type:** {question.get('question_type', 'N/A').title()}")
            with meta_col3:
                confidence = question.get('confidence', 0)
                st.write(f"**Confidence:** {confidence:.1%}")

            if question.get('generation_reason'):
                st.write(f"**Generation:** {question['generation_reason']}")

            if question.get('similar_past_questions'):
                st.write("**Similar Past Questions:**")
                for sim_q in question['similar_past_questions'][:3]:
                    st.write(f"- {sim_q}")

        st.divider()

def main():
    # Title
    st.title("🎯 AFCAT 2026 AI Dashboard")
    st.markdown("*AI-Powered Question Prediction & Generation System*")

    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")

    # Load data
    mock_data = load_mock_test()
    pred_data = load_predicted_questions()

    # Summary stats
    if mock_data and pred_data:
        total_mock = mock_data.get('metadata', {}).get('total_questions', 0)
        total_pred = pred_data.get('metadata', {}).get('total', 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AI Mock Questions", total_mock)
        with col2:
            st.metric("Predicted Questions", total_pred)
        with col3:
            st.metric("Total AI Questions", total_mock + total_pred)

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📝 Mock Test", "🔮 Predictions", "📊 Analytics"])

    with tab1:
        st.header("AI-Generated Mock Test")
        if mock_data and 'all_questions' in mock_data:
            questions = mock_data['all_questions']

            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                sections = list(set(q.get('section', 'Unknown') for q in questions))
                selected_section = st.selectbox("Filter by Section", ["All"] + sections, key="mock_section")
            with col2:
                topics = list(set(q.get('topic', 'Unknown') for q in questions))
                selected_topic = st.selectbox("Filter by Topic", ["All"] + topics, key="mock_topic")

            # Filter questions
            filtered_questions = questions
            if selected_section != "All":
                filtered_questions = [q for q in questions if q.get('section') == selected_section]
            if selected_topic != "All":
                filtered_questions = [q for q in filtered_questions if q.get('topic') == selected_topic]

            st.write(f"Showing {len(filtered_questions)} questions")

            # Display questions
            for i, question in enumerate(filtered_questions):
                display_question_card(question, i)
        else:
            st.warning("No mock test data available.")

    with tab2:
        st.header("AI-Predicted Questions")
        if pred_data and 'questions' in pred_data:
            questions = pred_data['questions']

            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                sections = list(set(q.get('section', 'Unknown') for q in questions))
                selected_section = st.selectbox("Filter by Section", ["All"] + sections, key="pred_section")
            with col2:
                topics = list(set(q.get('topic', 'Unknown') for q in questions))
                selected_topic = st.selectbox("Filter by Topic", ["All"] + topics, key="pred_topic")

            # Filter questions
            filtered_questions = questions
            if selected_section != "All":
                filtered_questions = [q for q in questions if q.get('section') == selected_section]
            if selected_topic != "All":
                filtered_questions = [q for q in filtered_questions if q.get('topic') == selected_topic]

            st.write(f"Showing {len(filtered_questions)} questions")

            # Display questions
            for i, question in enumerate(filtered_questions):
                display_question_card(question, i)
        else:
            st.warning("No predicted questions data available.")

    with tab3:
        st.header("Question Analytics")

        if mock_data and pred_data:
            # Combine all questions for analytics
            all_questions = []
            if 'all_questions' in mock_data:
                all_questions.extend(mock_data['all_questions'])
            if 'questions' in pred_data:
                all_questions.extend(pred_data['questions'])

            if all_questions:
                # Create DataFrame for analysis
                df = pd.DataFrame(all_questions)

                # Section distribution
                st.subheader("📊 Section Distribution")
                section_counts = df['section'].value_counts()
                st.bar_chart(section_counts)

                # Topic distribution (top 20)
                st.subheader("🎯 Top Topics")
                topic_counts = df['topic'].value_counts().head(20)
                st.bar_chart(topic_counts)

                # Difficulty distribution
                st.subheader("⚡ Difficulty Levels")
                if 'predicted_difficulty' in df.columns:
                    diff_counts = df['predicted_difficulty'].value_counts()
                    st.bar_chart(diff_counts)

                # Question types
                st.subheader("📝 Question Types")
                if 'question_type' in df.columns:
                    type_counts = df['question_type'].value_counts()
                    st.bar_chart(type_counts)

                # Confidence distribution
                st.subheader("🎯 Confidence Scores")
                if 'confidence' in df.columns:
                    st.line_chart(df['confidence'].dropna())
            else:
                st.warning("No questions data for analytics.")
        else:
            st.warning("No data available for analytics.")

    # Footer
    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit & Local Ollama AI*")

if __name__ == "__main__":
    main()