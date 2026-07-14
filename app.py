import streamlit as st
import os
import contextlib
import io
import matplotlib.pyplot as plt

# Page Setup Configurations
st.set_page_config(
    page_title="VTU BDA & ML Executable Dashboard", 
    page_icon="🚀", 
    layout="wide"
)

st.title("🚀 VTU BDA & ML Executable Lab Dashboard")
st.markdown("Select a laboratory assignment program from the sidebar menu to read the source code and execute its machine learning pipelines live.")

# Exact Mapping definitions matching your uploaded file names
program_map = {
    "Program 01: Iris Dataset Analysis & Distributions": "program1.py",
    "Program 02: Pima Diabetes Classifier (Naive Bayes)": "program2.py",
    "Program 03: Distributed Hadoop WordCount Core": "program3.py",
    "Program 04: Weather Data Analysis (PySpark)": "program4.py",
    "Program 05: Movie Rating Average MapReduce": "program5.py",
    "Program 06: Bike Trip History User Classification": "program6.py",
    "Program 07: BigMart Sales Prediction Regression": "program7.py",
    "Program 08: Twitter Sentiment Analysis (NLP)": "program8.py"
}

st.sidebar.header("Lab Navigation Menu")
selected_display = st.sidebar.selectbox("Select Active Assignment Script:", list(program_map.keys()))
target_filename = program_map[selected_display]

st.sidebar.markdown("---")
st.sidebar.markdown("📁 **Workspace Files Inventory:**")

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
    st.caption(f"Reading directly from local workspace workspace file: `{target_filename}`")
    
    # Check if the code file is Java code (stored in program3/program5) or Python
    is_java = "public class" in script_code or "import java." in script_code
    st.code(script_code, language="java" if is_java else "python")
    
    # 2. Execution Engine (For executable Python scripts)
    if not is_java and target_filename not in ["program4.py"]: # Exclude PySpark if it requires standalone cluster terminal run setups
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
                        # Create an isolated local scope execution map dictionary context 
                        exec_globals = {}
                        exec_locals = {}
                        exec(script_code, exec_globals, exec_locals)
                    
                    st.success("✨ Script engine pipeline completed execution successfully!")
                    
                    # Output printed logs onto console block text window
                    output_text = stdout_buffer.getvalue()
                    if output_text:
                        st.text_area("Console Terminal Metrics Logs:", value=output_text, height=250)
                    else:
                        st.info("Script completed successfully but returned no metrics prints text.")
                    
                    # Dynamically look for any active figures drawn by matplotlib inside the script execution runtime context
                    fig = plt.gcf()
                    if fig and fig.get_axes():
                        st.subheader("📊 Output Distribution Plot Canvas")
                        st.pyplot(fig)
                        
                except Exception as script_err:
                    st.error("❌ A runtime processing exception failure occurred inside the targeted script layout file.")
                    st.exception(script_err)
                    
    elif target_filename == "program4.py":
        st.info("ℹ️ PySpark Big Data architecture scripts require execution initialization commands inside a cluster ecosystem shell terminal context framework.")
    else:
        st.info("ℹ️ Distributed MapReduce Java applications must be compiled and deployed using target Hadoop cluster infrastructure commands.")
else:
    st.error(f"🚨 File Not Found Error: Please place the file named `{target_filename}` directly inside this exact folder path location alongside the main `app.py` script engine.")
