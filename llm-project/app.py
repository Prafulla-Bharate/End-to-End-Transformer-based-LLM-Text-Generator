import streamlit as st
import requests
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import our modules for direct model access
try:
    from generate import TextGenerator
    DIRECT_MODE = True
except ImportError:
    DIRECT_MODE = False
    st.warning("Direct model access not available. Using API mode only.")

# Page configuration
st.set_page_config(
    page_title="GPT Text Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
    .generation-box {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #e1f5fe;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'text_generator' not in st.session_state:
    st.session_state.text_generator = None

def load_model_direct():
    """Load model directly for faster generation"""
    if not st.session_state.model_loaded and DIRECT_MODE:
        with st.spinner("Loading model..."):
            try:
                st.session_state.text_generator = TextGenerator()
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
                return True
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return False
    return st.session_state.model_loaded

def generate_text_direct(prompt, max_length, temperature, top_k, top_p):
    """Generate text using direct model access"""
    if not st.session_state.model_loaded:
        st.error("Model not loaded!")
        return None
    
    try:
        with st.spinner("Generating text..."):
            generated_text = st.session_state.text_generator.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        return generated_text
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return None

def generate_text_api(prompt, max_length, temperature, top_k, top_p, api_url):
    """Generate text using API endpoint"""
    try:
        with st.spinner("Generating text via API..."):
            payload = {
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            }
            
            response = requests.post(f"{api_url}/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result["generated_text"]
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
    except requests.exceptions.RequestException as e:
        st.error(f"API connection failed: {e}")
        return None
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return None

def add_to_history(prompt, generated_text, parameters, generation_time):
    """Add generation result to history"""
    st.session_state.generation_history.append({
        "timestamp": datetime.now(),
        "prompt": prompt,
        "generated_text": generated_text,
        "parameters": parameters,
        "generation_time": generation_time
    })

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 GPT Text Generator</h1>', unsafe_allow_html=True)
    st.markdown("Generate creative text using our trained GPT model")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Generation Mode",
        ["Direct Model", "API Mode"],
        help="Direct mode is faster but requires model files. API mode uses HTTP requests."
    )
    
    # API configuration (if using API mode)
    if mode == "API Mode":
        api_url = st.sidebar.text_input(
            "API URL", 
            value="http://localhost:8000",
            help="URL of the running API server"
        )
        
        # Test API connection
        if st.sidebar.button("Test API Connection"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    st.sidebar.success("API connection successful!")
                else:
                    st.sidebar.error("API connection failed!")
            except Exception as e:
                st.sidebar.error(f"Cannot connect to API: {e}")
    
    # Model loading for direct mode
    if mode == "Direct Model" and DIRECT_MODE:
        if st.sidebar.button("Load Model"):
            load_model_direct()
        
        if st.session_state.model_loaded:
            st.sidebar.success("✅ Model Ready")
        else:
            st.sidebar.warning("⚠️ Model Not Loaded")
    
    # Generation parameters
    st.sidebar.subheader("📊 Generation Parameters")
    
    max_length = st.sidebar.slider(
        "Max Length",
        min_value=10,
        max_value=300,
        value=100,
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Controls randomness (lower = more predictable)"
    )
    
    top_k = st.sidebar.slider(
        "Top-K",
        min_value=0,
        max_value=100,
        value=50,
        help="Limit vocabulary to top K tokens (0 = no limit)"
    )
    
    top_p = st.sidebar.slider(
        "Top-P",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Nucleus sampling threshold"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Text Generation")
        
        # Text input
        prompt = st.text_area(
            "Enter your prompt:",
            height=100,
            placeholder="Type your prompt here... (e.g., 'Once upon a time in a magical forest')",
            help="Enter the text you want the model to continue from"
        )
        
        # Generation buttons
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            generate_button = st.button("🚀 Generate Text", type="primary")
        
        with col_b:
            if st.button("🎨 Creative Mode"):
                # Override parameters for creativity
                temperature = 1.1
                top_k = 40
                top_p = 0.85
                st.info("Using creative settings!")
                generate_button = True
        
        with col_c:
            if st.button("📝 Conservative Mode"):
                # Override parameters for coherence
                temperature = 0.5
                top_k = 60
                top_p = 0.95
                st.info("Using conservative settings!")
                generate_button = True
        
        # Generate text when button is pressed
        if generate_button and prompt.strip():
            parameters = {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            }
            
            start_time = time.time()
            
            # Choose generation method based on mode
            if mode == "Direct Model" and st.session_state.model_loaded:
                generated_text = generate_text_direct(prompt, max_length, temperature, top_k, top_p)
            elif mode == "API Mode":
                generated_text = generate_text_api(prompt, max_length, temperature, top_k, top_p, api_url)
            else:
                st.error("Please load the model or check API connection!")
                generated_text = None
            
            generation_time = time.time() - start_time
            
            # Display results
            if generated_text:
                st.success(f"Generated in {generation_time:.2f} seconds!")
                
                # Display generated text in a nice box
                st.markdown("### 📄 Generated Text:")
                st.markdown(f'<div class="generation-box">{generated_text}</div>', unsafe_allow_html=True)
                
                # Add to history
                add_to_history(prompt, generated_text, parameters, generation_time)
                
                # Download button
                st.download_button(
                    label="📥 Download Text",
                    data=generated_text,
                    file_name=f"generated_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        elif generate_button and not prompt.strip():
            st.warning("Please enter a prompt!")
        
        # Preset prompts section
        st.subheader("🎯 Quick Start Prompts")
        
        preset_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology has advanced beyond our wildest dreams",
            "The secret to happiness lies in",
            "Once upon a time in a distant galaxy",
            "The most important lesson I've learned is",
            "In the year 2050, humanity will",
            "The old library contained a mysterious book that",
            "When I opened the door, I discovered"
        ]
        
        selected_preset = st.selectbox("Choose a preset prompt:", [""] + preset_prompts)
        
        if st.button("Use Preset Prompt") and selected_preset:
            st.text_area("Enter your prompt:", value=selected_preset, key="preset_prompt")
    
    with col2:
        st.subheader("📊 Statistics")
        
        # Display current parameters
        st.markdown("### Current Settings:")
        st.markdown(f'<div class="metric-card"><strong>Max Length:</strong> {max_length}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><strong>Temperature:</strong> {temperature}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><strong>Top-K:</strong> {top_k}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><strong>Top-P:</strong> {top_p}</div>', unsafe_allow_html=True)
        
        # Generation history stats
        if st.session_state.generation_history:
            st.markdown("### 📈 Generation Stats:")
            
            history_df = pd.DataFrame(st.session_state.generation_history)
            
            # Average generation time
            avg_time = history_df['generation_time'].mean()
            st.metric("Average Generation Time", f"{avg_time:.2f}s")
            
            # Total generations
            total_generations = len(history_df)
            st.metric("Total Generations", total_generations)
            
            # Generation time chart
            if len(history_df) > 1:
                fig = px.line(
                    history_df.reset_index(), 
                    x='index', 
                    y='generation_time',
                    title="Generation Time Over Sessions",
                    labels={'index': 'Generation #', 'generation_time': 'Time (seconds)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # History section
    if st.session_state.generation_history:
        st.subheader("📚 Generation History")
        
        # Controls for history
        col_hist1, col_hist2, col_hist3 = st.columns(3)
        
        with col_hist1:
            if st.button("🗑️ Clear History"):
                st.session_state.generation_history = []
                st.experimental_rerun()
        
        with col_hist2:
            show_history = st.checkbox("Show Full History", value=False)
        
        with col_hist3:
            export_history = st.button("📤 Export History")
        
        if export_history:
            history_text = ""
            for i, entry in enumerate(st.session_state.generation_history):
                history_text += f"Generation {i+1} ({entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}):\n"
                history_text += f"Prompt: {entry['prompt']}\n"
                history_text += f"Generated: {entry['generated_text']}\n"
                history_text += f"Parameters: {entry['parameters']}\n"
                history_text += f"Time: {entry['generation_time']:.2f}s\n"
                history_text += "-" * 50 + "\n\n"
            
            st.download_button(
                label="📥 Download History",
                data=history_text,
                file_name=f"generation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        # Display history
        if show_history:
            for i, entry in enumerate(reversed(st.session_state.generation_history[-10:])):  # Show last 10
                with st.expander(f"Generation {len(st.session_state.generation_history) - i}: {entry['prompt'][:50]}..."):
                    st.write(f"**Timestamp:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Prompt:** {entry['prompt']}")
                    st.write(f"**Generated Text:** {entry['generated_text']}")
                    st.write(f"**Parameters:** {entry['parameters']}")
                    st.write(f"**Generation Time:** {entry['generation_time']:.2f} seconds")
    
    # Batch generation section
    st.subheader("🔄 Batch Generation")
    
    with st.expander("Generate Multiple Texts"):
        batch_prompts = st.text_area(
            "Enter prompts (one per line):",
            height=100,
            placeholder="Prompt 1\nPrompt 2\nPrompt 3"
        )
        
        if st.button("🚀 Generate Batch"):
            if batch_prompts.strip():
                prompts_list = [p.strip() for p in batch_prompts.split('\n') if p.strip()]
                
                if len(prompts_list) > 10:
                    st.warning("Too many prompts! Maximum 10 allowed.")
                else:
                    st.info(f"Generating text for {len(prompts_list)} prompts...")
                    
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, batch_prompt in enumerate(prompts_list):
                        if mode == "Direct Model" and st.session_state.model_loaded:
                            result = generate_text_direct(batch_prompt, max_length, temperature, top_k, top_p)
                        elif mode == "API Mode":
                            result = generate_text_api(batch_prompt, max_length, temperature, top_k, top_p, api_url)
                        else:
                            st.error("Model not available!")
                            break
                        
                        if result:
                            results.append({"prompt": batch_prompt, "generated": result})
                        
                        progress_bar.progress((i + 1) / len(prompts_list))
                    
                    # Display batch results
                    st.success(f"Generated {len(results)} texts!")
                    
                    for i, result in enumerate(results):
                        st.write(f"**Prompt {i+1}:** {result['prompt']}")
                        st.write(f"**Generated:** {result['generated']}")
                        st.write("---")
            else:
                st.warning("Please enter at least one prompt!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🤖 GPT Text Generator | Built with Streamlit | 
        <a href='/docs' target='_blank'>API Documentation</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


