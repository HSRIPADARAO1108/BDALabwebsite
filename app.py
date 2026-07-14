import streamlit as st
import os
import contextlib
import io
import base64
import matplotlib.pyplot as plt

# Page Setup Configurations (Must be the very first Streamlit command)
st.set_page_config(
    page_title="VTU BDA & ML Executable Dashboard", 
    page_icon="🚀", 
    layout="wide"
)

# Function to safely load a local image from the root directory and convert it to base64
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# Convert your root image.jpg into a secure web-safe string format
bin_str = get_base64_image("images.jpg")

# Inject Custom CSS using the base64 background string
if bin_str:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)), url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Make all dashboard headings and text contrast beautifully against the image */
        h1, h2, h3, p, span, label, .stMarkdown {{
            color: #f8fafc !important;
        }}
        
        /* Stylize text areas and select boxes to match the premium transparent theme */
        .stTextArea textarea, .stSelectbox div {{
            background-color: rgba(30, 41, 59, 0.7) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    # Fallback to a clean dark slate if the image is processing or temporarily unreadable
    st.markdown(
        """
        <style>
        .stApp { background-color: #0f172a; }
        </style>
        """,
        unsafe_allow_html=True
    )

st.title("🚀 VTU BDA & ML Executable Lab Dashboard")
st.markdown("Select a laboratory assignment program from the sidebar menu to read the source code and execute its machine learning pipelines live.")

# Exact Mapping definitions matching your uploaded file names
program_map = {
    "Program 01: Iris Dataset Analysis & Distributions": "program1.py",
    "Program 02: Pima Diabetes Classifier (Naive Bayes)": "program2.py",
    "Program 03: Distributed Hadoop WordCount Core": "program3.py",
    "Program 04: Weather Data Analysis Pipeline": "program4.py",
    "Program 05: Movie Rating Average Calculator": "program5.py",
    "Program 06: Bike Trip History User Classification": "program6.py",
    "Program 07: BigMart Sales Prediction Regression": "program7.py",
    "Program 08: Twitter Sentiment Analysis (NLP)": "program8.py"
}

st.sidebar.header("Lab Navigation Menu")
selected_display = st.sidebar.selectbox("Select Active Assignment Script:", list(program_map.keys()))
target_filename = program_map[selected_display]

st.sidebar.markdown("---")
st.sidebar.markdown("📁 **Workspace Files Inventory Check:**")

# Sidebar indicators checking verification of target script presence
if os.path.exists(target_filename):
    st.sidebar.success(f"✅ Target Found: `{target_filename}`")
else:
    st.sidebar.error(f"❌ Missing Target: `{target_filename}`")

# Render active layout structure
st.header(selected_display)

if os.path.exists(target_filename):
    # 1. Read and display the actual script text layout
    with open(target_filename, "r", encoding="utf-8") as f:
        script_code = f.read()
        
    st.subheader("📝 Script Source Code View")
    st.caption(f"Reading directly from local workspace file: `{target_filename}`")
    
    # Automatically check if the code file is Java code or Python
    is_java = "public class" in script_code or "import java." in script_code
    st.code(script_code, language="java" if is_java else "python")
    
    # 2. Execution Engine (For executable Python scripts)
    if not is_java: 
        st.subheader("⚡ Live Script Execution Console")
        st.caption("Click the trigger below to run the machine learning pipelines within this folder path location context.")
        
        if st.button(f"Run {target_filename} Live"):
            # Set up memory buffers to collect console stdout prints
            stdout_buffer = io.StringIO()
            
            # Clear previous plots to avoid graphic bleedover leaks
            plt.close('all')
            
            with st.spinner(f"Executing machine learning operations inside `{target_filename}`..."):
                try:
                    # Redirect core terminal output streams directly into our UI layout screen
                    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
                        # Unified shared execution scope dictionary forces variables and 
                        # module imports to stay bound inside pandas functions.
                        exec_scope = {"__name__": "__main__"}
                        exec(script_code, exec_scope, exec_scope)
                    
                    st.success("✨ Script engine pipeline completed execution successfully!")
                    
                    # Output printed logs onto console block text window
                    output_text = stdout_buffer.getvalue()
                    if output_text:
                        st.text_area("Console Terminal Metrics Logs:", value=output_text, height=300)
                    else:
                        st.info("Script completed successfully but returned no metrics prints text.")
                    
                    # Dynamically look for any active figures drawn by matplotlib inside the script execution runtime context
                    fig = plt.gcf()
                    if fig and fig.get_axes():
                        st.subheader("📊 Output Visualizations Canvas")
                        st.pyplot(fig)
                        
                except Exception as script_err:
                    st.error("❌ A runtime processing exception failure occurred inside the targeted script layout file.")
                    st.exception(script_err)
    else:
        st.info("ℹ️ Distributed MapReduce Java applications must be compiled and deployed using target Hadoop cluster infrastructure commands.")
else:
    st.error(f"🚨 File Not Found Error: Please place the file named `{target_filename}` directly inside this exact folder path location alongside the main `app.py` script engine.")
