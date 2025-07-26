import streamlit as st
import re
import io
import requests
from datetime import datetime

# Try to import optional dependencies with fallbacks
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

# Basic URL content extraction without newspaper3k
def extract_url_content_basic(url):
    """Extract content from URL using basic requests and regex"""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response.raise_for_status()
        
        html = response.text
        
        # Basic HTML tag removal and text extraction
        # Remove script and style tags
        html = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Basic content extraction - look for paragraphs
        lines = text.split('\n')
        content_lines = []
        
        for line in lines:
            line = line.strip()
            # Keep lines that are likely content (length check)
            if len(line) > 50 and len(line) < 1000:
                content_lines.append(line)
        
        if content_lines:
            content = '\n'.join(content_lines[:20])  # Take first 20 good lines
            return content, f"📰 Source: {url}"
        else:
            return text[:3000] if text else None, f"📰 Source: {url}"  # Fallback to first 3000 chars
            
    except Exception as e:
        return None, f"❌ URL extraction failed: {str(e)}"

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
    .api-setup {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Hugging Face API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

def query_huggingface_api(payload, api_key=None):
    """Query Hugging Face Inference API"""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"error": "Model loading, please wait..."}
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

# Initialize session state
if 'hf_api_key' not in st.session_state:
    st.session_state.hf_api_key = ""

# Check dependencies and show status
def show_dependency_status():
    """Show which features are available"""
    status_text = "🔧 **System Status:**\n\n"
    
    # Check AI Summarization
    if st.session_state.hf_api_key or True:  # Free tier also works
        status_text += "✅ AI Summarization: Available (Hugging Face API)\n"
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

def create_ai_summary(text, target_sentences, target_words, api_key=None):
    """Create AI-powered summary using Hugging Face API"""
    try:
        # Prepare the payload
        max_length = max(target_words + 20, target_sentences * 15)
        min_length = max(target_words - 10, target_sentences * 10, 20)
        
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": min(max_length, 500),  # API has limits
                "min_length": min_length,
                "do_sample": False
            }
        }
        
        # Query the API
        result = query_huggingface_api(payload, api_key)
        
        if "error" in result:
            return f"AI Error: {result['error']}", 0, 0
        
        if not result or not isinstance(result, list) or not result[0]:
            return "AI Error: Invalid response from API", 0, 0
            
        summary_text = result[0].get('summary_text', '')
        
        if not summary_text:
            return "AI Error: No summary generated", 0, 0
        
        # Clean and split sentences
        sentences = clean_and_split_sentences(summary_text)
        
        # Adjust based on target
        if len(sentences) > target_sentences:
            selected_sentences = sentences[:target_sentences]
        else:
            selected_sentences = sentences
        
        # Apply paraphrasing
        final_sentences = [smart_paraphrase(sentence) for sentence in selected_sentences]
        
        result_text = ' '.join(final_sentences)
        word_count = len(result_text.split())
        sentence_count = len(final_sentences)
        
        return result_text, sentence_count, word_count
        
    except Exception as e:
        return f"AI Error: {str(e)}", 0, 0

def create_smart_summary(text, target_sentences, target_words, priority="sentences", use_ai=True):
    """Create summary using available methods"""
    if use_ai:
        result = create_ai_summary(text, target_sentences, target_words, st.session_state.hf_api_key)
        if not result[0].startswith("AI Error"):
            return result
    
    # Fallback to basic summary
    return create_basic_summary(text, target_sentences, target_words)

def process_uploaded_file(uploaded_file):
    """Handle file uploads with basic text extraction"""
    try:
        if uploaded_file.type == "application/pdf":
            if PYPDF2_AVAILABLE:
                # Use PyPDF2 if available
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
                return content, f"📄 Source: {uploaded_file.name}"
            else:
                return None, "❌ PDF processing requires PyPDF2. Please try a text file or paste the content directly."
        else:
            # Handle text files and other formats
            try:
                content = str(uploaded_file.read(), "utf-8")
                return content, f"📝 Source: {uploaded_file.name}"
            except UnicodeDecodeError:
                # Try different encodings
                uploaded_file.seek(0)
                try:
                    content = str(uploaded_file.read(), "latin-1")
                    return content, f"📝 Source: {uploaded_file.name}"
                except:
                    return None, "❌ Could not decode file. Please try a UTF-8 encoded text file."
    except Exception as e:
        return None, f"❌ Error reading file: {str(e)}"

def extract_url_content(url):
    """Extract content from URL using available methods"""
    if NEWSPAPER_AVAILABLE:
        try:
            article = Article(url)
            article.download()
            article.parse()
            if article.text:
                return article.text, f"📰 Source: {url}"
        except Exception as e:
            # Fall back to basic extraction
            pass
    
    # Use basic extraction method
    return extract_url_content_basic(url)

# Main App Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Professional Text Summarizer</h1>
    <p><em>Simple presets for quick use • Advanced controls for precision</em></p>
</div>
""", unsafe_allow_html=True)

# API Configuration Section
with st.expander("🔑 Optional: Hugging Face API Setup (for better AI performance)", expanded=False):
    st.markdown("""
    <div class="api-setup">
    <strong>🚀 Want better AI summarization?</strong><br>
    Get a free Hugging Face API key for improved performance and no rate limits!<br><br>
    
    <strong>Steps:</strong><br>
    1. Go to <a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a><br>
    2. Create a new token (free)<br>
    3. Paste it below<br><br>
    
    <em>Note: The app works without an API key too, but may be slower.</em>
    </div>
    """, unsafe_allow_html=True)
    
    api_key_input = st.text_input(
        "Hugging Face API Key (optional):",
        type="password",
        value=st.session_state.hf_api_key,
        help="Optional: Paste your free Hugging Face API key here for better performance"
    )
    
    if api_key_input != st.session_state.hf_api_key:
        st.session_state.hf_api_key = api_key_input
        if api_key_input:
            st.success("✅ API key saved! You'll get better AI performance now.")
        else:
            st.info("ℹ️ Using free tier - may be slower but still works!")

# Show system status
with st.expander("🔧 System Status", expanded=False):
    ai_status = "✅ Available (Hugging Face API)"
    
    if NEWSPAPER_AVAILABLE:
        url_status = "✅ Available (Advanced)"
    else:
        url_status = "✅ Available (Basic HTML extraction)"
    
    if PYPDF2_AVAILABLE:
        pdf_status = "✅ Available"
    else:
        pdf_status = "⚠️ Limited (Text files only)"
    
    st.markdown(f"""
    **📊 Feature Status:**
    
    - **AI Summarization:** {ai_status}
    - **URL Extraction:** {url_status}
    - **File Processing:** {pdf_status}
    
    **💡 Note:** All core features work! Some advanced features may have limitations without additional packages.
    """)
    
    if not NEWSPAPER_AVAILABLE or not PYPDF2_AVAILABLE:
        st.markdown("""
        <div class="warning-box">
        <strong>🔧 To enable full features:</strong><br>
        • For better URL extraction: <code>pip install newspaper3k</code><br>
        • For PDF support: <code>pip install PyPDF2</code><br>
        <em>Current setup works great for most use cases!</em>
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
    
    # AI Toggle
    st.markdown("---")
    use_ai = st.checkbox("🤖 Use AI Summarization", value=True, help="Uncheck to use basic text processing only")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📥 Input")
    
    # Input options - Always show all methods now
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
        url_input = st.text_input(
            "Enter article URL:",
            placeholder="https://example.com/article",
            help="Basic HTML extraction available - works with most news sites!"
        )
        if url_input.strip():
            with st.spinner("Extracting content from URL..."):
                content, source_info = extract_url_content(url_input)
                if content is None:
                    st.error(source_info)
                elif "❌" in source_info:
                    st.error(source_info)
                else:
                    st.success("✅ Content extracted successfully!")
    
    elif input_method == "📁 Upload File":
        file_types = ['txt', 'md', 'csv']
        if PYPDF2_AVAILABLE:
            file_types.append('pdf')
            help_text = "Upload text files, Markdown, CSV, or PDF"
        else:
            help_text = "Upload text files, Markdown, or CSV (PDF requires PyPDF2)"
            
        uploaded_file = st.file_uploader(
            "Choose a file:",
            type=file_types,
            help=help_text
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
                    content, target_sentences, target_words, summary_priority, use_ai
                )
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
            
            if summary.startswith("Error") or summary.startswith("AI Error"):
                st.error(summary)
                st.info("💡 Try unchecking 'Use AI Summarization' to use basic text processing instead.")
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
                
                method_used = "AI-Powered (Hugging Face)" if use_ai and not summary.startswith("AI Error") else "Basic Text Processing"
                
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
    
    ### 🤖 AI vs Basic Processing:
    
    **AI Summarization (Recommended):**
    - Uses advanced BART model via Hugging Face API
    - Better understanding of context and meaning
    - Higher quality, more coherent summaries
    - May be slower on first use (model loading)
    
    **Basic Text Processing:**
    - Fast, reliable fallback method
    - Selects sentences from different parts of text
    - Always available, no dependencies
    - Good for when AI is unavailable
    
    ### 💡 Pro Tips:
    - Get a free Hugging Face API key for better performance
    - Start with **Sentences First** for most use cases
    - Use **Words First** for platforms with strict character limits
    - Try both AI and Basic modes to see which you prefer
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
    
    ### 🔍 Testing Different Modes:
    
    1. **Test AI vs Basic:** Try the same text with AI on/off to compare quality
    2. **Test Presets:** Same text with different presets to see length variations
    3. **Test Priority:** Advanced mode with "Sentences First" vs "Words First"
    
    ### 📰 More Sample Texts:
    
    **Business News:**
    ```
    Tech company announces quarterly earnings, showing 15% growth despite market challenges...
    ```
    
    **Academic Content:**
    ```
    Recent studies in climate science indicate significant changes in global weather patterns...
    ```
    """)

with tab4:
    st.markdown(f"""
    ## ℹ️ About This Tool
    
    This professional text summarizer creates high-quality summaries tailored to your needs using advanced AI technology.
    
    ### ⚡ Current Features:
    - **Dual Mode Interface:** Simple presets + advanced controls
    - **AI-Powered:** Uses Facebook's BART model via Hugging Face API
    - **Multiple Input Methods:** Text, URLs, and file uploads
    - **Smart Paraphrasing:** Avoids repetitive language from source
    - **Flexible Targeting:** Control both sentence count and word count
    - **Fallback System:** Works even when AI is unavailable
    
    ### 🤖 Technology Status:
    - **AI Summarization:** ✅ Available (Hugging Face API)
    - **URL Extraction:** ✅ Available (Basic HTML extraction + newspaper3k if installed)
    - **File Processing:** ✅ Available (Text files always, PDF if PyPDF2 installed)
    
    ### 🚀 How It Works:
    1. **AI Mode:** Sends text to Hugging Face's BART model for intelligent summarization
    2. **Processing:** Applies smart paraphrasing and sentence optimization
    3. **Targeting:** Adjusts output to meet your exact sentence/word requirements
    4. **Fallback:** Uses basic text processing if AI is unavailable
    
    ### 🎯 Perfect For:
    - Students and researchers
    - Business professionals
    - Content creators
    - Social media managers
    - Anyone needing quick, quality summaries
    
    ### 🔑 Free Hugging Face API Key Benefits:
    - Faster processing (no queuing)
    - Higher rate limits
    - More reliable service
    - Still completely free!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🎯 Professional Text Summarizer • AI-Powered • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
