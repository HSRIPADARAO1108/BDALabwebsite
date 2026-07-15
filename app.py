import streamlit as st
import os
import contextlib
import io
import base64
import matplotlib.pyplot as plt

# Page Setup
st.set_page_config(page_title="VTU BDA & ML Dashboard", page_icon="🚀", layout="wide")

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

bin_str = get_base64_image("images.jpg")

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    /* Global Styles */
    .stApp {{
        background: {'linear-gradient(rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.9)), url("data:image/jpg;base64,' + bin_str + '")' if bin_str else '#0f172a'};
        background-size: cover;
        background-attachment: fixed;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: rgba(17, 24, 39, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}

    /* Typography */
    h1, h2, h3, p, label {{ color: #f8fafc !important; }}
    
    /* Input Elements */
    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {{
        background-color: rgba(30, 41, 59, 0.6) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px;
    }}

    /* Buttons */
    .stButton>button {{
        width: 100%;
        border-radius: 8px;
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }}
    .stButton>button:hover {{ background-color: #2563eb; }}

    /* Code Blocks */
    .stCodeBlock {{
        background-color: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}

    /* Mobile Adjustments */
    @media (max-width: 768px) {{
        .block-container {{ padding: 1rem; }}
        h1 {{ font-size: 1.5rem !important; }}
    }}
</style>
""", unsafe_allow_html=True)

# Main Application Logic
st.title("🚀 VTU BDA & ML Executable Lab Dashboard")
st.markdown("Select a laboratory assignment from the sidebar to execute ML pipelines.")

program_map = {
    "Program 01: Iris Dataset Analysis": "program1.py",
    "Program 02: Pima Diabetes Classifier": "program2.py",
    "Program 03: Hadoop WordCount": "program3.py",
    "Program 04: Weather Analysis": "program4.py",
    "Program 05: Movie Rating Calc": "program5.py",
    "Program 06: Bike Trip History": "program6.py",
    "Program 07: BigMart Sales Prediction": "program7.py",
    "Program 08: Twitter Sentiment Analysis": "program8.py"
}

st.sidebar.header("Lab Navigation")
selected_display = st.sidebar.selectbox("Select Assignment:", list(program_map.keys()))
target_filename = program_map[selected_display]

if os.path.exists(target_filename):
    st.sidebar.success(f"✅ Found: {target_filename}")
    st.header(selected_display)
    
    with open(target_filename, "r", encoding="utf-8") as f:
        script_code = f.read()
    
    is_java = "public class" in script_code or "import java." in script_code
    st.code(script_code, language="java" if is_java else "python")
    
    if not is_java and st.button(f"Run {target_filename}"):
        stdout_buffer = io.StringIO()
        plt.close('all')
        with st.spinner("Executing pipeline..."):
            try:
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
                    exec_scope = {"__name__": "__main__"}
                    exec(script_code, exec_scope, exec_scope)
                st.success("Pipeline executed successfully!")
                if stdout_buffer.getvalue():
                    st.text_area("Console Logs:", value=stdout_buffer.getvalue(), height=200)
                if plt.get_fignums():
                    st.subheader("📊 Visualizations")
                    st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"Runtime Error: {e}")
else:
    st.sidebar.error(f"❌ Missing: {target_filename}")
