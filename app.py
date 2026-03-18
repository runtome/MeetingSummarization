"""Streamlit demo app for meeting summarization."""

import streamlit as st
import yaml

from src.inference import load_model, parse_sections, summarize_meeting

st.set_page_config(
    page_title="Meeting Summarizer",
    page_icon="📝",
    layout="wide",
)


@st.cache_resource
def get_model(adapter_path: str, base_model: str):
    """Load model with caching to avoid reloading."""
    return load_model(adapter_path, base_model)


def load_config(path: str = "configs/training_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    st.title("📝 Meeting Summarizer")
    st.markdown("Upload or paste a meeting transcript to get structured meeting minutes.")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        adapter_path = st.text_input("Adapter path", value="./outputs/final_adapter")
        base_model = st.text_input("Base model", value="Qwen/Qwen2.5-7B-Instruct")

        with st.expander("Advanced"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
            max_new_tokens = st.slider("Max new tokens", 256, 2048, 1024, 128)
            chunk_size = st.slider("Chunk size (chars)", 1000, 5000, 3000, 500)

        load_btn = st.button("Load Model", type="primary", use_container_width=True)

        if load_btn:
            with st.spinner("Loading model..."):
                try:
                    model, tokenizer = get_model(adapter_path, base_model)
                    st.session_state["model"] = model
                    st.session_state["tokenizer"] = tokenizer
                    st.success("Model loaded!")
                except Exception as e:
                    st.error(f"Failed to load model: {e}")

    # Main area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Transcript")
        input_method = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)

        transcript = ""
        if input_method == "Paste text":
            transcript = st.text_area(
                "Paste meeting transcript",
                height=400,
                placeholder="Speaker 1: Let's discuss the Q3 budget...\nSpeaker 2: I think we should...",
            )
        else:
            uploaded = st.file_uploader("Upload .txt file", type=["txt"])
            if uploaded:
                transcript = uploaded.read().decode("utf-8")
                st.text_area("Preview", transcript, height=400, disabled=True)

    with col2:
        st.subheader("Meeting Minutes")

        if st.button("Summarize", type="primary", use_container_width=True):
            if not transcript.strip():
                st.warning("Please provide a transcript first.")
            elif "model" not in st.session_state:
                st.warning("Please load the model first (sidebar).")
            else:
                config = load_config()
                # Override with sidebar settings
                config["inference"]["temperature"] = temperature
                config["inference"]["max_new_tokens"] = max_new_tokens
                config["inference"]["chunk_size"] = chunk_size

                with st.spinner("Summarizing..."):
                    try:
                        result = summarize_meeting(
                            transcript,
                            st.session_state["model"],
                            st.session_state["tokenizer"],
                            config,
                        )
                        st.session_state["result"] = result
                    except Exception as e:
                        st.error(f"Summarization failed: {e}")

        # Display results
        if "result" in st.session_state:
            result = st.session_state["result"]
            sections = parse_sections(result)

            tabs = st.tabs(["Full Output"] + [s for s in sections])
            with tabs[0]:
                st.markdown(result)

            for i, (section_name, content) in enumerate(sections.items(), 1):
                with tabs[i]:
                    st.markdown(f"## {section_name}\n{content}")

            # Download button
            st.download_button(
                "Download as Markdown",
                data=result,
                file_name="meeting_minutes.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
