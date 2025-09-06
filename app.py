import streamlit as st
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import textstat
import matplotlib.pyplot as plt


# Use st.cache_resource to cache the models so they are only loaded once.
@st.cache_resource
def load_models():
    """Loads the summarization and sentiment analysis pipelines."""
    # Using t5-small for speed, but t5-base or bart-large-cnn would give better results.
    summarizer = pipeline("summarization", model="t5-small")
    # A general-purpose sentiment analysis model.
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return summarizer, sentiment_analyzer


summarizer, sentiment_analyzer = load_models()


def smart_chunk_text(text, max_tokens=500):
    """
    Chunks text by sentences to prevent splitting in the middle of a sentence.
    This ensures that each chunk is a meaningful, complete unit.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Check if adding the next sentence will exceed the token limit.
        if len(current_chunk.split()) + len(sentence.split()) > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def hierarchical_summarize(text):
    """
    Performs a two-step summarization for long documents to create a cohesive summary.
    1. Summarizes individual chunks of the document.
    2. Summarizes the combined summaries from step 1.
    """
    chunks = smart_chunk_text(text)
    
    # Step 1: Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        st.info(f"Summarizing chunk {i+1} of {len(chunks)}...")
        result = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        chunk_summaries.append(result[0]["summary_text"])

    # If there's more than one chunk summary, perform a second-level summarization.
    if len(chunk_summaries) > 1:
        combined_summaries = " ".join(chunk_summaries)
        st.info("Generating final summary from chunk summaries...")
        final_summary_result = summarizer(combined_summaries, max_length=150, min_length=40, do_sample=False)
        return final_summary_result[0]["summary_text"]
    else:
        return chunk_summaries[0]


st.title("🧠 AI Text Summarizer & Data Analyzer")
st.markdown("Paste your article or paragraph below to get a detailed summary and analysis.")

user_input = st.text_area("Paste your article or paragraph here:", height=300)

if st.button("Analyze & Summarize"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing and summarizing..."):
            # Get original text metrics
            original_word_count = len(user_input.split())
            original_char_count = len(user_input)
            original_readability_score = textstat.flesch_reading_ease(user_input)

            # Get sentiment
            sentiment = sentiment_analyzer(user_input[:512])[0]
            
            # Get summary
            summary = hierarchical_summarize(user_input)
            
            # Get summary metrics
            summary_word_count = len(summary.split())
            summary_char_count = len(summary)
            summary_readability_score = textstat.flesch_reading_ease(summary)

        st.success("Analysis Complete!")
        
        # Display the summary
        st.subheader("📌 Summary")
        st.write(summary)

        st.subheader("📊 Text Analysis")
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Original Word Count", value=f"{original_word_count:,}")
        with col2:
            st.metric(label="Summary Word Count", value=f"{summary_word_count:,}")
        with col3:
            st.metric(label="Compression Ratio", value=f"{summary_word_count/original_word_count:.2%}")
        
        st.markdown(f"**Original Readability:** {original_readability_score:.2f} (Flesch Reading Ease)")
        st.markdown(f"**Summary Readability:** {summary_readability_score:.2f} (Flesch Reading Ease)")
        st.markdown(f"**Overall Sentiment:** {sentiment['label']} (Confidence: {sentiment['score']:.2%})")

        # Create a bar chart for visualization
        st.subheader("📈 Word Count Comparison")
        data = {'Text Type': ['Original Text', 'Summarized Text'], 'Word Count': [original_word_count, summary_word_count]}
        
        fig, ax = plt.subplots()
        ax.bar(data['Text Type'], data['Word Count'], color=['#6495ED', '#FFA07A'])
        ax.set_ylabel('Word Count')
        ax.set_title('Word Count of Original vs. Summarized Text')
        st.pyplot(fig)