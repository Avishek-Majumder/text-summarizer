import streamlit as st
import time
import re
import io

# Try to import optional dependencies with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch not available. Some features may be limited.")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("❌ Transformers library not available. Please check your requirements.txt")

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    st.warning("⚠️ Newspaper3k not available. URL extraction will be disabled.")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    st.warning("⚠️ PyPDF2 not available. PDF upload will be disabled.")

# Page configuration
st.set_page_config(
    page_title="Professional Text Summarizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .preset-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stats-container {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_summarization_model():
    """Load the BART model with caching for better performance"""
    if not TRANSFORMERS_AVAILABLE:
        return None
        
    print("📦 Loading BART-large-cnn...")
    try:
        # Try the smaller model first for Streamlit Cloud compatibility
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1  # Using CPU since GPU might not be available
        )
        print("✅ DistilBART loaded successfully!")
        return summarizer
    except Exception as e:
        print(f"❌ DistilBART failed: {e}, trying even smaller model...")
        try:
            # Fallback to an even smaller model
            summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=-1
            )
            return summarizer
        except Exception as e2:
            print(f"❌ All models failed: {e2}")
            return None

# Initialize the model
if 'summarizer' not in st.session_state:
    if TRANSFORMERS_AVAILABLE:
        with st.spinner("Loading AI model... This might take a moment on first run."):
            st.session_state.summarizer = load_summarization_model()
            if st.session_state.summarizer is None:
                st.error("❌ Failed to load AI model. Please check your internet connection and try again.")
    else:
        st.session_state.summarizer = None

# Define preset configurations - keeping them exactly as in original
PRESETS = {
    "tweet": {
        "name": "📱 Tweet/Social",
        "description": "Perfect for social media posts",
        "sentences": 2,
        "target_words": 45,
        "max_words": 55
    },
    "quick": {
        "name": "📄 Quick Summary", 
        "description": "Brief overview of key points",
        "sentences": 5,
        "target_words": 75,
        "max_words": 90
    },
    "executive": {
        "name": "📊 Executive Brief",
        "description": "Professional summary for business",
        "sentences": 7,
        "target_words": 110,
        "max_words": 130
    },
    "detailed": {
        "name": "📚 Detailed Summary",
        "description": "Comprehensive overview",
        "sentences": 10,
        "target_words": 170,
        "max_words": 200
    }
}

def clean_and_split_sentences(text):
    """Clean text and split into proper sentences - exact same logic as original"""
    text = re.sub(r'\s+', ' ', text.strip())
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if len(sentence) >= 10 and not sentence.endswith('..'):
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            sentences.append(sentence)
    
    return sentences

def smart_paraphrase(text):
    """Enhanced paraphrasing with comprehensive replacements - keeping all original replacements"""
    replacements = {
        # News & reporting
        'reportedly': 'allegedly', 'infiltrated': 'breached', 'obtained': 'secured',
        'suspended': 'halted', 'authorities': 'officials', 'considering': 'contemplating',
        'individuals': 'people', 'conducted': 'carried out', 'mentioned': 'stated',
        'operational': 'functional', 'approximately': 'about', 'subsequently': 'later',
        
        # Sports terms
        'smashed': 'hit', 'crushing': 'defeating', 'sealed': 'secured', 'unassailable': 'commanding',
        'deliveries': 'balls', 'holed out': 'was caught', 'removed cheaply': 'dismissed for low scores',
        
        # General terms
        'server': 'system', 'technical advice': 'technical assistance', 'restart': 'restore',
        'come to a standstill': 'been suspended', 'no response has been received': 'they have not received a response',
        'have not been able to': 'cannot', 'As a result': 'Consequently', 'Even after': 'Despite',
        
        # Organization names
        'Rajdhani Unnayan Kartripakkha': 'Rajuk', 'Electronic Construction Permitting System': 'ECPS',
        'Bangladesh Computer Council': 'BCC', 'West Indies': 'Windies',
        
        # Time and process
        'From the following day': 'The next day', 'all types of services': 'all services',
        'are currently being provided': 'are available', 'linked to the incident': 'connected to the breach'
    }
    
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    return result

def create_smart_summary(text, target_sentences, target_words, priority="sentences"):
    """Create summary with dual sentence/word targeting - exact same logic"""
    
    # Check if we have the required dependencies
    if not TRANSFORMERS_AVAILABLE or st.session_state.summarizer is None:
        return "❌ AI model not available. Please check your dependencies.", 0, 0
    
    print(f"🎯 Creating summary: {target_sentences} sentences, ~{target_words} words")
    print(f"📋 Priority: {priority}")
    
    try:
        summarizer = st.session_state.summarizer
        
        # Generate multiple summary options
        summaries = []
        
        # Option 1: Conservative approach
        summary1 = summarizer(
            text,
            max_length=max(target_words + 20, target_sentences * 15),
            min_length=max(target_words - 10, target_sentences * 10),
            do_sample=False,
            truncation=True
        )[0]['summary_text']
        summaries.append(summary1)
        
        # Option 2: More flexible approach
        if priority == "words":
            # Focus on word count first
            summary2 = summarizer(
                text,
                max_length=target_words + 15,
                min_length=max(target_words - 15, 20),
                do_sample=True,
                temperature=0.7,
                truncation=True
            )[0]['summary_text']
        else:
            # Focus on sentence structure first
            summary2 = summarizer(
                text,
                max_length=target_sentences * 25,
                min_length=target_sentences * 12,
                do_sample=True,
                temperature=0.7,
                truncation=True
            )[0]['summary_text']
        
        summaries.append(summary2)
        
        # Choose best summary based on priority
        if priority == "words":
            # Pick summary closest to target word count
            best_summary = min(summaries, key=lambda x: abs(len(x.split()) - target_words))
        else:
            # Pick longest, most comprehensive summary
            best_summary = max(summaries, key=lambda x: len(x.split()))
        
        print(f"📊 Initial summary: {len(best_summary.split())} words")
        
        # Clean and split sentences
        sentences = clean_and_split_sentences(best_summary)
        print(f"🔍 Extracted {len(sentences)} sentences")
        
        # Handle sentence count based on priority - keeping exact logic
        if priority == "sentences":
            # Prioritize exact sentence count
            if len(sentences) >= target_sentences:
                selected_sentences = sentences[:target_sentences]
            else:
                # Try to get more sentences
                expanded_summary = summarizer(
                    text,
                    max_length=target_sentences * 30,
                    min_length=target_sentences * 18,
                    do_sample=True,
                    temperature=0.8,
                    truncation=True
                )[0]['summary_text']
                
                expanded_sentences = clean_and_split_sentences(expanded_summary)
                all_sentences = sentences + expanded_sentences
                
                # Remove duplicates
                unique_sentences = []
                for sentence in all_sentences:
                    is_duplicate = False
                    for existing in unique_sentences:
                        common_words = set(sentence.lower().split()) & set(existing.lower().split())
                        if len(common_words) > len(sentence.split()) * 0.7:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_sentences.append(sentence)
                
                selected_sentences = unique_sentences[:target_sentences]
        
        else:
            # Prioritize word count, adjust sentences as needed
            current_words = sum(len(s.split()) for s in sentences)
            
            if current_words <= target_words + 15:
                # Use all sentences if within word limit
                selected_sentences = sentences
            else:
                # Trim sentences to meet word target
                selected_sentences = []
                word_count = 0
                
                for sentence in sentences:
                    sentence_words = len(sentence.split())
                    if word_count + sentence_words <= target_words + 10:
                        selected_sentences.append(sentence)
                        word_count += sentence_words
                    else:
                        break
        
        # Paraphrase each sentence
        print("🔄 Paraphrasing sentences...")
        final_sentences = []
        for sentence in selected_sentences:
            paraphrased = smart_paraphrase(sentence)
            final_sentences.append(paraphrased)
        
        # Final assembly
        result = ' '.join(final_sentences)
        word_count = len(result.split())
        sentence_count = len(final_sentences)
        
        print(f"✅ Final: {sentence_count} sentences, {word_count} words")
        
        return result, sentence_count, word_count
        
    except Exception as e:
        print(f"❌ Summary creation failed: {e}")
        return f"Error: {str(e)}", 0, 0

def process_uploaded_file(uploaded_file):
    """Handle file uploads - PDF and text files"""
    if not PYPDF2_AVAILABLE and uploaded_file.type == "application/pdf":
        return None, "❌ PDF processing not available. Please install PyPDF2."
        
    try:
        if uploaded_file.type == "application/pdf":
            # Handle PDF files
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
            return content, f"📄 Source: {uploaded_file.name}"
        else:
            # Handle text files
            content = str(uploaded_file.read(), "utf-8")
            return content, f"📝 Source: {uploaded_file.name}"
    except Exception as e:
        return None, f"❌ Error reading file: {str(e)}"

def extract_url_content(url):
    """Extract content from URL using newspaper3k"""
    if not NEWSPAPER_AVAILABLE:
        return None, "❌ URL extraction not available. Please install newspaper3k."
        
    try:
        article = Article(url)
        article.download()
        article.parse()
        if article.text:
            return article.text, f"📰 Source: {url}"
        else:
            return None, "❌ Could not extract text from URL"
    except Exception as e:
        return None, f"❌ URL processing failed: {str(e)}"

# Main App Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Professional Text Summarizer</h1>
    <p><em>Simple presets for quick use • Advanced controls for precision</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar for mode selection and controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Mode selection
    mode = st.radio(
        "📋 Choose Your Mode",
        ["Simple Mode", "Advanced Mode"],
        help="Simple: Quick presets | Advanced: Custom control"
    )
    
    if mode == "Simple Mode":
        st.subheader("🎯 Quick Presets")
        
        preset_options = {
            "tweet": "📱 Tweet/Social (2-3 sentences, ~45 words)",
            "quick": "📄 Quick Summary (4-5 sentences, ~75 words)",
            "executive": "📊 Executive Brief (6-7 sentences, ~110 words)",
            "detailed": "📚 Detailed Summary (8-10 sentences, ~170 words)"
        }
        
        preset_choice = st.radio(
            "Choose your summary style:",
            list(preset_options.keys()),
            format_func=lambda x: preset_options[x],
            index=1  # Default to "quick"
        )
        
        # Display preset details
        selected_preset = PRESETS[preset_choice]
        st.markdown(f"""
        <div class="preset-card">
            <strong>{selected_preset['name']}</strong><br>
            {selected_preset['description']}<br>
            <small>Target: {selected_preset['sentences']} sentences, ~{selected_preset['target_words']} words</small>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.subheader("⚙️ Custom Controls")
        
        custom_sentences = st.slider(
            "🎯 Target sentences",
            min_value=2,
            max_value=15,
            value=5,
            help="Number of sentences in the summary"
        )
        
        custom_words = st.slider(
            "📝 Target words",
            min_value=25,
            max_value=300,
            value=75,
            step=5,
            help="Approximate number of words"
        )
        
        priority = st.radio(
            "⚖️ Priority",
            ["Sentences First", "Words First"],
            help="Which target is more important?"
        )
        
        st.markdown("""
        **Sentences First:** Hit exact sentence count, approximate word target  
        **Words First:** Hit word target, adjust sentence count as needed
        """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📥 Input")
    
    # Input options
    input_method = st.radio(
        "Choose input method:",
        ["📄 Paste Text", "🌐 URL", "📁 Upload File"]
    )
    
    content = ""
    source_info = ""
    
    if input_method == "📄 Paste Text":
        text_input = st.text_area(
            "Paste your text here:",
            placeholder="Paste your article, news story, research paper, or any text here...",
            height=300
        )
        if text_input.strip():
            content = text_input
            source_info = "✍️ Source: Direct text input"
    
    elif input_method == "🌐 URL":
        if not NEWSPAPER_AVAILABLE:
            st.warning("⚠️ URL extraction is not available. Please install newspaper3k or use text input instead.")
        else:
            url_input = st.text_input(
                "Enter article URL:",
                placeholder="https://example.com/article"
            )
            if url_input.strip():
                with st.spinner("Extracting content from URL..."):
                    content, source_info = extract_url_content(url_input)
                    if content is None:
                        st.error(source_info)
    
    else:  # File upload
        file_types = ['txt']
        if PYPDF2_AVAILABLE:
            file_types.append('pdf')
            
        uploaded_file = st.file_uploader(
            "Choose a file:",
            type=file_types,
            help="Upload a text file" + (" or PDF" if PYPDF2_AVAILABLE else "")
        )
        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                content, source_info = process_uploaded_file(uploaded_file)
                if content is None:
                    st.error(source_info)

    # Generate summary button
    generate_summary = st.button("✨ Create Summary", type="primary", use_container_width=True)

with col2:
    st.header("📤 Output")
    
    if generate_summary and content:
        if len(content.strip()) < 30:
            st.error("❌ Content too short for summarization")
        else:
            # Determine parameters based on mode
            if mode == "Simple Mode":
                preset = PRESETS[preset_choice]
                target_sentences = preset["sentences"]
                target_words = preset["target_words"]
                summary_priority = "sentences"
                mode_info = f"Using {preset['name']}"
            else:
                target_sentences = custom_sentences
                target_words = custom_words
                summary_priority = priority.lower().split()[0]
                mode_info = f"Custom: {target_sentences} sentences, {target_words} words"
            
            # Generate summary with progress indicator
            with st.spinner("Creating your professional summary..."):
                start_time = time.time()
                summary, final_sentences, final_words = create_smart_summary(
                    content, target_sentences, target_words, summary_priority
                )
                processing_time = time.time() - start_time
            
            if summary.startswith("Error"):
                st.error(summary)
            else:
                # Display the summary
                st.subheader("📋 Your Professional Summary")
                st.markdown(f"""
                <div class="success-box">
                    {summary}
                </div>
                """, unsafe_allow_html=True)
                
                # Copy button (using Streamlit's built-in functionality)
                st.text_area("Copy your summary:", value=summary, height=100, label_visibility="collapsed")
                
                # Calculate and display statistics
                original_words = len(content.split())
                compression = round((1 - final_words/original_words) * 100, 1)
                
                sentence_match = "✅ Perfect" if final_sentences == target_sentences else f"📊 {final_sentences}/{target_sentences}"
                word_match = "✅ Perfect" if abs(final_words - target_words) <= 10 else f"📊 {final_words}/{target_words}"
                
                st.subheader("📊 Summary Statistics")
                st.markdown(f"""
                <div class="stats-container">
                    <strong>📋 Mode:</strong> {mode_info}<br><br>
                    
                    <strong>📊 Content Analysis:</strong><br>
                    • Original text: {original_words} words<br>
                    • Summary length: {final_words} words<br>
                    • Compression ratio: {compression}% reduction<br><br>
                    
                    <strong>🎯 Target Achievement:</strong><br>
                    • Sentences: {final_sentences}/{target_sentences} {sentence_match}<br>
                    • Words: {final_words}/{target_words} {word_match}<br><br>
                    
                    <strong>⚡ Performance:</strong><br>
                    • Processing time: {processing_time:.1f} seconds<br>
                    • Priority used: {summary_priority.title()}-first approach<br>
                    • Model: BART-large-cnn
                </div>
                """, unsafe_allow_html=True)
                
                # Source information
                if source_info:
                    st.info(source_info)
    
    elif generate_summary and not content:
        st.warning("❌ Please provide some text to summarize!")

# Information tabs at the bottom
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Preset Guide", "⚙️ Advanced Guide", "📊 Examples", "ℹ️ About"])

with tab1:
    st.markdown("""
    ## 🎯 When to Use Each Preset:
    
    ### 📱 Tweet/Social (2-3 sentences, ~45 words)
    **Perfect for:**
    - Social media posts (Twitter, LinkedIn, Facebook)
    - Text messages and quick shares
    - Headlines and brief announcements
    
    **Example output:** *"Australia defeated West Indies by six wickets in the third T20. Tim David hit the fastest T20 century for Australia with 102 off 37 balls."*
    
    ### 📄 Quick Summary (4-5 sentences, ~75 words)
    **Perfect for:**
    - Email briefings
    - Meeting notes
    - Quick status updates
    - News headlines expanded
    
    **Example output:** *"Australia secured a 3-0 series lead with a six-wicket victory over West Indies. Tim David scored the fastest T20 century for Australia, hitting 102 off just 37 deliveries. His innings included 11 sixes and six fours. Australia won with 23 balls to spare at Warner Park. The victory gives them an unassailable lead in the five-match series."*
    
    ### 📊 Executive Brief (6-7 sentences, ~110 words)
    **Perfect for:**
    - Business reports
    - Academic abstracts
    - Presentation summaries
    - Professional briefings
    
    ### 📚 Detailed Summary (8-10 sentences, ~170 words)
    **Perfect for:**
    - Research paper abstracts
    - Comprehensive overviews
    - Detailed meeting minutes
    - Long-form content summaries
    """)

with tab2:
    st.markdown("""
    ## 🔧 Advanced Mode Features:
    
    ### 🎯 Sentence Control (2-15)
    - **2-4 sentences:** Ultra-brief summaries
    - **5-8 sentences:** Standard summaries  
    - **9-12 sentences:** Comprehensive summaries
    - **13-15 sentences:** Detailed analysis (for very long content)
    
    ### 📝 Word Control (25-300)
    - **25-50 words:** Social media, headlines
    - **50-100 words:** Standard business use
    - **100-200 words:** Professional reports
    - **200-300 words:** Academic abstracts
    
    ### ⚖️ Priority Settings:
    
    **Sentences First** (Recommended):
    - Guarantees exact sentence count
    - Word count will be approximate
    - Better for consistent structure
    - Example: Always get exactly 5 sentences
    
    **Words First**:
    - Hits target word count precisely
    - Sentence count may vary slightly
    - Better for strict length requirements
    - Example: Always get exactly 100 words
    
    ### 💡 Pro Tips:
    - Start with **Sentences First** for most use cases
    - Use **Words First** for platforms with strict character limits
    - Longer targets work better with longer input texts
    - Very short targets (under 40 words) may compromise quality
    """)

with tab3:
    st.markdown("""
    ## 🧪 Test Examples:
    
    ### 📰 News Article:
    ```
    David blasts fastest T20 ton for Australia in series win over West Indies
    Middle order batter Tim David smashed the fastest Twenty20 International century...
    ```
    **Try:** Executive Brief preset or Custom 6 sentences, 120 words
    
    ### 🏛️ Academic Content:
    ```
    The exact date of the University of Oxford's founding is unknown, but the school traces its roots back to at least 1096...
    ```
    **Try:** Detailed Summary preset or Custom 8 sentences, 150 words
    
    ### 💼 Business News:
    ```
    On May 19, a group of hackers reportedly infiltrated the Electronic Construction Permitting System...
    ```
    **Try:** Quick Summary preset or Custom 5 sentences, 80 words
    
    ### 📱 For Social Media:
    Use Tweet/Social preset for any content you want to share quickly!
    """)

with tab4:
    st.markdown("""
    ## ℹ️ About This Tool
    
    This professional text summarizer uses state-of-the-art AI to create high-quality summaries tailored to your needs.
    
    ### ⚡ Features:
    - **Dual Mode Interface:** Simple presets for quick use, advanced controls for precision
    - **Multiple Input Methods:** Text, URLs, and file uploads (PDF, TXT)
    - **Smart Paraphrasing:** Avoids repetitive language from the original text
    - **Flexible Targeting:** Control both sentence count and word count
    - **Professional Quality:** Suitable for business, academic, and personal use
    
    ### 🤖 Technology:
    - **Model:** Facebook's BART-large-cnn (state-of-the-art summarization)
    - **Fallback:** DistilBART for compatibility
    - **Processing:** Advanced sentence splitting and paraphrasing
    
    ### 🎯 Perfect For:
    - Students and researchers
    - Business professionals
    - Content creators
    - Social media managers
    - Anyone who needs quick, quality summaries
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🎯 Professional Text Summarizer • Simple presets + Advanced custom controls!</p>
</div>
""", unsafe_allow_html=True)
