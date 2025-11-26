import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
# streamlit run "C:\Users\eliot\OneDrive - University of Waterloo\CS Project\Project.py"

st.set_page_config(
    page_title="Distributioner",
    layout="wide",   # this makes the page use full width
    initial_sidebar_state="expanded"
)

st.title("Distribution Fitting App")

# -------------------
# Layout Columns
# -------------------
col1, col2,col3 = st.columns([3,5,3])

# Left column: Data input

with col1:
    st.header("Data Input")
    fileUploader = st.file_uploader("Upload CSV here")
    userInput = st.text_input("Enter a value")
    btn_col1,btn_col2,btn_col3 = st.columns([1,1,4])
    with btn_col1:
        addButton = st.button("Add value")
    with btn_col2: 
        resetButton = st.button("Reset data")

# Right column: Distribution settings
with col3:
    st.header("Distribution Settings")
    scipyChoice = st.selectbox("Select distribution",
                               ("Normal","Gamma","Weibull","Exponential",
                                "Beta","Chi-Squared","Triangular",
                                "Lognormal","Uniform","Pareto"))
    locSlider = st.slider("Shift (loc)", -10, 10, 0)
    scaleSlider = st.slider("Scale", 0.1, 10.0, 1.0)
    
# -------------------
# Display current data
# -------------------

if "dataList" not in st.session_state:
    st.session_state.dataList = []
DISTRIBUTIONS = {
    "Normal": stats.norm,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Beta": stats.beta,
    "Exponential": stats.expon,
    "Chi-Squared": stats.chi2,
    "Lognormal": stats.lognorm,
    "Uniform": stats.uniform,
    "Triangular": stats.triang,
    "Pareto": stats.pareto,
}
if addButton:
    try:
        num = float(userInput)
        st.session_state.dataList.append(num)
        st.session_state.userInput = ""  # clear the box
        with col1:
            st.success(f"Added {num}")
    except ValueError:
        with col1:
            st.error("Please enter a valid number")
if resetButton:
    st.session_state.dataList = []
if fileUploader is not None:
    df = pd.read_csv(fileUploader)
    st.session_state.dataList = df.values.flatten().tolist()
dist_object = DISTRIBUTIONS[scipyChoice]
data = np.array(st.session_state.dataList)



dist_manual = None
if len(data) != 0:
    try:
        params = dist_object.fit(data)
        shape_params = params[:-2]
        loc_fitted = params[-2]
        scale_fitted = params[-1]
        if scipyChoice == "Beta":
            a, b = shape_params
            # No shape slider for Beta, just use fitted shapes
            dist_manual = dist_object(a, b, loc=locSlider, scale=scaleSlider)

        elif len(shape_params) == 0:
            dist_manual = dist_object(loc=locSlider, scale=scaleSlider)

        elif len(shape_params) == 1:
            with col3:
                shapeSlider = st.slider("Shape", 0.01, 10.0, float(shape_params[0]))
            dist_manual = dist_object(shapeSlider, loc=locSlider, scale=scaleSlider)
        else:
            dist_manual = dist_object(loc=locSlider, scale=scaleSlider)

    except Exception as e:
        with col2:
            st.warning("not enough data for this distribution")
        params = None

if dist_manual is not None and len(data) != 0:
    with col3:
        with st.expander("Histogram and Fitted Distribution", expanded=True):
            x = np.linspace(min(data)-1, max(data)+1, 300)
            pdf = dist_manual.pdf(x)
            fig, ax = plt.subplots()
            ax.hist(data, bins=1, density=False, alpha=0.5)
            ax.plot(x, pdf, 'r-', linewidth=2)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{scipyChoice} Fit")
            with col2:
                st.pyplot(fig)
    
if dist_manual is not None:
    with col3:
        st.subheader("Current Data")
        st.write(f"Shape: {shape_params if shape_params else 'N/A'}")
        st.write(f"Loc: {locSlider}")
        st.write(f"Scale: {scaleSlider}")




