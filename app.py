import streamlit as st
import re
import io
from datetime import datetime

# Try to import optional dependencies with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

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
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Check dependencies and show status
def show_dependency_status():
    """Show which features are available"""
    status_text = "🔧 **System Status:**\n\n"
    
    if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        status_text += "✅ AI Summarization: Available\n"
    else:
        status_text += "❌ AI Summarization: Not Available\n"
        
    if NEWSPAPER_AVAILABLE:
        status_text += "✅ URL Extraction: Available\n"
    else:
        status_text += "❌ URL Extraction: Not Available\n"
        
    if PYPDF2_AVAILABLE:
        status_text += "✅ PDF Processing: Available\n"
    else:
        status_text += "❌ PDF Processing: Not Available\n"
    
    return status_text

@st.cache_resource
def load_summarization_model():
    """Load the BART model with caching for better performance"""
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        return None
        
    try:
        # Try the smaller, more reliable model first
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1
        )
        return summarizer
    except Exception as e:
        st.error(f"Failed to load AI model: {str(e)}")
        return None

# Initialize the model only if dependencies are available
if 'summarizer_loaded' not in st.session_state:
    st.session_state.summarizer_loaded = False
    st.session_state.summarizer = None
    
    if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        with st.spinner("Loading AI model... This might take a moment."):
            st.session_state.summarizer = load_summarization_model()
            st.session_state.summarizer_loaded = True

# Define preset configurations
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
    """Clean text and split into proper sentences"""
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
    """Enhanced paraphrasing with comprehensive replacements"""
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

def create_basic_summary(text, target_sentences, target_words):
    """Create a basic summary when AI is not available"""
    sentences = clean_and_split_sentences(text)
    
    if len(sentences) <= target_sentences:
        selected_sentences = sentences
    else:
        # Select sentences from beginning, middle, and end
        if target_sentences >= 3:
            # Take first sentence, some from middle, and last sentence
            selected_sentences = [sentences[0]]
            
            if target_sentences > 2:
                middle_count = target_sentences - 2
                middle_start = len(sentences) // 3
                middle_end = (2 * len(sentences)) // 3
                middle_sentences = sentences[middle_start:middle_end]
                
                if len(middle_sentences) >= middle_count:
                    step = len(middle_sentences) // middle_count
                    selected_middle = [middle_sentences[i] for i in range(0, len(middle_sentences), step)][:middle_count]
                else:
                    selected_middle = middle_sentences
                
                selected_sentences.extend(selected_middle)
            
            if len(sentences) > 1:
                selected_sentences.append(sentences[-1])
        else:
            selected_sentences = sentences[:target_sentences]
    
    # Apply paraphrasing
    final_sentences = [smart_paraphrase(sentence) for sentence in selected_sentences]
    
    result = ' '.join(final_sentences)
    word_count = len(result.split())
    sentence_count = len(final_sentences)
    
    return result, sentence_count, word_count

def create_ai_summary(text, target_sentences, target_words, priority="sentences"):
    """Create AI-powered summary"""
    try:
        summarizer = st.session_state.summarizer
        
        # Generate summary
        summary_result = summarizer(
            text,
            max_length=max(target_words + 20, target_sentences * 15),
            min_length=max(target_words - 10, target_sentences * 10),
            do_sample=False,
            truncation=True
        )[0]['summary_text']
        
        # Clean and split sentences
        sentences = clean_and_split_sentences(summary_result)
        
        # Adjust based on target
        if len(sentences) > target_sentences:
            selected_sentences = sentences[:target_sentences]
        else:
            selected_sentences = sentences
        
        # Apply paraphrasing
        final_sentences = [smart_paraphrase(sentence) for sentence in selected_sentences]
        
        result = ' '.join(final_sentences)
        word_count = len(result.split())
        sentence_count = len(final_sentences)
        
        return result, sentence_count, word_count
        
    except Exception as e:
        return f"AI summarization failed: {str(e)}", 0, 0

def create_smart_summary(text, target_sentences, target_words, priority="sentences"):
    """Create summary using available methods"""
    if st.session_state.summarizer is not None:
        return create_ai_summary(text, target_sentences, target_words, priority)
    else:
        return create_basic_summary(text, target_sentences, target_words)

def process_uploaded_file(uploaded_file):
    """Handle file uploads"""
    if uploaded_file.type == "application/pdf" and not PYPDF2_AVAILABLE:
        return None, "❌ PDF processing not available."
        
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
            return content, f"📄 Source: {uploaded_file.name}"
        else:
            content = str(uploaded_file.read(), "utf-8")
            return content, f"📝 Source: {uploaded_file.name}"
    except Exception as e:
        return None, f"❌ Error reading file: {str(e)}"

def extract_url_content(url):
    """Extract content from URL"""
    if not NEWSPAPER_AVAILABLE:
        return None, "❌ URL extraction not available."
        
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

# Show system status
with st.expander("🔧 System Status", expanded=False):
    st.markdown(show_dependency_status())
    if not (TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE):
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Limited Mode:</strong> AI summarization is not available. 
        The app will use basic text processing for summaries.
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
            index=1
        )
        
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
            st.warning("⚠️ URL extraction is not available. Please use text input instead.")
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
                start_time = datetime.now()
                summary, final_sentences, final_words = create_smart_summary(
                    content, target_sentences, target_words, summary_priority
                )
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
            
            if summary.startswith("Error") or summary.startswith("AI summarization failed"):
                st.error(summary)
            else:
                # Display the summary
                st.subheader("📋 Your Professional Summary")
                st.markdown(f"""
                <div class="success-box">
                    {summary}
                </div>
                """, unsafe_allow_html=True)
                
                # Copy button
                st.text_area("Copy your summary:", value=summary, height=100, label_visibility="collapsed")
                
                # Calculate and display statistics
                original_words = len(content.split())
                compression = round((1 - final_words/original_words) * 100, 1) if original_words > 0 else 0
                
                sentence_match = "✅ Perfect" if final_sentences == target_sentences else f"📊 {final_sentences}/{target_sentences}"
                word_match = "✅ Perfect" if abs(final_words - target_words) <= 10 else f"📊 {final_words}/{target_words}"
                
                method_used = "AI-Powered" if st.session_state.summarizer is not None else "Basic Text Processing"
                
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
                    • Method used: {method_used}<br>
                    • Priority: {summary_priority.title()}-first approach
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
    
    ### 📄 Quick Summary (4-5 sentences, ~75 words)
    **Perfect for:**
    - Email briefings
    - Meeting notes
    - Quick status updates
    - News headlines expanded
    
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
    - **13-15 sentences:** Detailed analysis
    
    ### 📝 Word Control (25-300)
    - **25-50 words:** Social media, headlines
    - **50-100 words:** Standard business use
    - **100-200 words:** Professional reports
    - **200-300 words:** Academic abstracts
    
    ### 💡 Pro Tips:
    - Start with **Sentences First** for most use cases
    - Use **Words First** for platforms with strict character limits
    - Longer targets work better with longer input texts
    """)

with tab3:
    st.markdown("""
    ## 🧪 Test Examples:
    
    Try pasting this sample text to test the summarizer:
    
    ```
    David blasts fastest T20 ton for Australia in series win over West Indies
    Middle order batter Tim David smashed the fastest Twenty20 International century for Australia as they sealed a six-wicket victory over the West Indies in the third T20 on Friday to take an unassailable 3-0 lead in their five-match series. David hit 11 sixes and six fours to finish on unbeaten 102 off 37 deliveries, with Australia crushing the hosts with 23 balls to spare at Warner Park in Basseterre, Saint Kitts.
    ```
    
    **Try:** Executive Brief preset or Custom 6 sentences, 120 words
    """)

with tab4:
    st.markdown(f"""
    ## ℹ️ About This Tool
    
    This professional text summarizer creates high-quality summaries tailored to your needs.
    
    ### ⚡ Current Features:
    - **Dual Mode Interface:** Simple presets + advanced controls
    - **Multiple Input Methods:** Text, URLs, and file uploads
    - **Smart Paraphrasing:** Avoids repetitive language
    - **Flexible Targeting:** Control sentence count and word count
    
    ### 🤖 Technology Status:
    - **AI Summarization:** {"✅ Available" if st.session_state.summarizer is not None else "❌ Not Available"}
    - **URL Extraction:** {"✅ Available" if NEWSPAPER_AVAILABLE else "❌ Not Available"}
    - **PDF Processing:** {"✅ Available" if PYPDF2_AVAILABLE else "❌ Not Available"}
    
    ### 🎯 Perfect For:
    - Students and researchers
    - Business professionals
    - Content creators
    - Social media managers
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🎯 Professional Text Summarizer • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
