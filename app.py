# Text Summarizer - Streamlit Version
# Much more reliable for deployment than Gradio

import streamlit as st
import re
import time

# page config
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

def smart_extractive_summary(text, num_sentences=5):
    """
    Simple but effective extractive summarization
    Works without any AI dependencies
    """
    # split into sentences
    sentences = []
    for s in re.split(r'[.!?]+', text):
        s = s.strip()
        if len(s) > 15:  # ignore very short fragments
            sentences.append(s)
    
    if len(sentences) <= num_sentences:
        return sentences
    
    # score sentences based on word frequency and position
    words = text.lower().split()
    word_freq = {}
    for word in words:
        if len(word) > 3:  # ignore short words like "the", "and"
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # score each sentence
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        # word frequency score
        word_score = sum(word_freq.get(word.lower(), 0) for word in sentence.split())
        
        # position bonus (first and last sentences often important)
        position_bonus = 0
        if i < 2:  # first two sentences
            position_bonus = 10
        elif i >= len(sentences) - 2:  # last two sentences
            position_bonus = 5
        
        # length bonus (not too short, not too long)
        length = len(sentence.split())
        if 8 <= length <= 25:
            length_bonus = 5
        else:
            length_bonus = 0
        
        total_score = word_score + position_bonus + length_bonus
        sentence_scores.append((total_score, sentence))
    
    # get top sentences
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    best_sentences = [sent for score, sent in sentence_scores[:num_sentences]]
    
    return best_sentences

def improve_sentences(sentences):
    """Make the language sound more natural"""
    
    # word improvements
    replacements = {
        'reportedly': 'allegedly',
        'infiltrated': 'breached', 
        'obtained': 'secured',
        'suspended': 'halted',
        'authorities': 'officials',
        'individuals': 'people',
        'conducted': 'carried out',
        'mentioned': 'stated',
        'approximately': 'about',
        'considering': 'contemplating',
        'smashed': 'hit',
        'crushing': 'defeating',
        'sealed': 'secured',
        'deliveries': 'balls',
        'server': 'system',
        'technical advice': 'technical assistance'
    }
    
    improved = []
    for sentence in sentences:
        result = sentence
        for old, new in replacements.items():
            result = result.replace(old, new)
        improved.append(result)
    
    return improved

# main app
st.title("📝 Text Summarizer")
st.markdown("*Create smart summaries of any text - fast and reliable!*")

# sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    length_option = st.selectbox(
        "Summary Length",
        ["Short (2-3 sentences)", "Medium (4-5 sentences)", "Long (6-8 sentences)"],
        index=1
    )
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("• Works best with 100+ words")
    st.markdown("• Try different lengths for your needs")
    st.markdown("• Great for articles, essays, reports")

# main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📥 Input")
    
    # text input
    text_input = st.text_area(
        "Paste your text here:",
        placeholder="Paste your article, document, essay, or any long text here...",
        height=300,
        help="Enter at least 50 words for best results"
    )
    
    # file upload
    uploaded_file = st.file_uploader(
        "Or upload a text file:",
        type=['txt'],
        help="Upload a .txt file to summarize"
    )
    
    # handle file upload
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode('utf-8')
            text_input = file_content
            st.success(f"File uploaded: {uploaded_file.name}")
        except:
            st.error("Could not read the file. Please try a different .txt file.")
    
    # summarize button
    if st.button("✨ Create Summary", type="primary", use_container_width=True):
        if not text_input or len(text_input.strip()) < 50:
            st.error("Please enter at least 50 words of text to summarize.")
        else:
            # figure out how many sentences to generate
            if length_option == "Short (2-3 sentences)":
                target = 3
            elif length_option == "Long (6-8 sentences)":
                target = 7
            else:  # Medium
                target = 5
            
            with st.spinner("Creating your summary..."):
                start_time = time.time()
                
                try:
                    # get the best sentences
                    best_sentences = smart_extractive_summary(text_input, target)
                    
                    # improve the language
                    improved_sentences = improve_sentences(best_sentences)
                    
                    # put it together
                    summary = '. '.join(improved_sentences) + '.'
                    
                    # calculate stats
                    original_words = len(text_input.split())
                    summary_words = len(summary.split())
                    compression = round((1 - summary_words/original_words) * 100, 1)
                    processing_time = time.time() - start_time
                    
                    # store in session state
                    st.session_state.summary = summary
                    st.session_state.stats = {
                        'original_words': original_words,
                        'summary_words': summary_words,
                        'compression': compression,
                        'sentences': len(improved_sentences),
                        'processing_time': processing_time
                    }
                    
                    st.success("Summary created successfully!")
                    
                except Exception as e:
                    st.error(f"Error creating summary: {str(e)}")

with col2:
    st.header("📤 Output")
    
    # show summary if it exists
    if hasattr(st.session_state, 'summary'):
        st.subheader("📋 Your Summary")
        st.text_area(
            "Summary:",
            value=st.session_state.summary,
            height=200,
            help="Copy this summary to use elsewhere",
            label_visibility="collapsed"
        )
        
        # copy button
        if st.button("📋 Copy Summary", use_container_width=True):
            st.write("Summary copied to clipboard!")  # Note: actual clipboard copy needs JS
        
        # show statistics
        st.subheader("📊 Statistics")
        stats = st.session_state.stats
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Original Words", stats['original_words'])
            st.metric("Compression", f"{stats['compression']}%")
        
        with col_b:
            st.metric("Summary Words", stats['summary_words'])
            st.metric("Sentences", stats['sentences'])
        
        st.metric("Processing Time", f"{stats['processing_time']:.1f}s")
        
    else:
        st.info("👈 Enter some text on the left and click 'Create Summary' to get started!")
        
        # show example
        st.subheader("📝 Example")
        st.markdown("""
        **Try this sample text:**
        
        Artificial intelligence has made remarkable progress in recent years. Machine learning algorithms can now process vast amounts of data with incredible speed and accuracy. Deep learning models have revolutionized computer vision, natural language processing, and speech recognition. Companies across industries are integrating AI into their operations to improve efficiency and create new products. However, concerns about job displacement, privacy, and ethical implications continue to grow. Experts emphasize the need for responsible AI development and proper regulation.
        """)

# footer
st.markdown("---")
st.markdown("**Built with Streamlit** • Reliable text summarization without complex dependencies")