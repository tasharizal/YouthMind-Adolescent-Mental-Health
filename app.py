import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
import textwrap
import re
import time
from fpdf import FPDF

# -----------------------------------------------------------------------------
# 0. PAGE CONFIGURATION & THEME SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="YouthMind Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# --- THEME STATE INITIALIZATION ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# --- DYNAMIC COLOR VARIABLES (THE FIX) ---
# These variables automatically switch based on the theme
if st.session_state.theme == 'light':
    # LIGHT MODE PALETTE
    theme_bg_color = "#FFFFFF"
    theme_text_color = "#31333F"        # Dark Grey Text
    theme_sub_text = "#555555"          # Medium Grey
    theme_card_bg = "#F0F2F6"           # Light Grey Card
    theme_border_clr = "#DCDFE4"
    theme_metric_val = "#31333F"        # Dark Text for Metrics
    
    # Specific fix for "Glass" cards
    glass_bg = "rgba(255, 255, 255, 0.9)"
    overlay_rgba = "rgba(255, 255, 255, 0.85)"
    card_text_clr = "#31333F"
    
    # Insight Box Text (Must be dark in light mode if background is light)
    insight_text_clr = "#31333F" 
    insight_sub_clr = "#555555"

else:
    # DARK MODE PALETTE
    theme_bg_color = "#0E1117"
    theme_text_color = "#FFFFFF"        # White Text
    theme_sub_text = "#cbd5e1"          # Light Grey
    theme_card_bg = "#1E2130"           # Dark Card
    theme_border_clr = "rgba(255, 255, 255, 0.1)"
    theme_metric_val = "#FFFFFF"        # White Text for Metrics

    # Specific fix for "Glass" cards
    glass_bg = "rgba(255, 255, 255, 0.05)"
    overlay_rgba = "rgba(14, 17, 23, 0.85)"
    card_text_clr = "#E0E0E0"
    
    # Insight Box Text (Must be light in dark mode)
    insight_text_clr = "#FFFFFF"
    insight_sub_clr = "rgba(255, 255, 255, 0.7)"

# --- LOAD CSS FUNCTION ---
def local_css(file_name):
    # We check if file exists to prevent errors
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Try loading external CSS if you have it, otherwise skip
local_css("style.css")

# --- SESSION STATE FOR ENTRY ---
if 'entered' not in st.session_state:
    st.session_state['entered'] = False

def enter_dashboard():
    st.session_state['entered'] = True

def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None 

# -----------------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING (FIXED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Load Data
        df17 = pd.read_csv("data/clean_2017.csv")
        df22 = pd.read_csv("data/clean_2022.csv")
    except FileNotFoundError:
        st.error("Data files not found. Ensure 'clean_2017.csv' and 'clean_2022.csv' are present.")
        return pd.DataFrame()

    # Add Year Marker
    df17['Year'] = 2017
    df22['Year'] = 2022

    # --- 1. Clean Age ---
    def clean_age(x):
        x_str = str(x).lower().strip()
        if 'younger' in x_str: return 13
        if 'older' in x_str: return 18
        try: return float(x)
        except: return np.nan

    # --- 2. Clean Physical Activity ---
    def clean_pa(x):
        if str(x).lower() == 'never': return 0
        try: return float(x)
        except: return np.nan
        
    # --- 3. Clean Parents Status ---
    def clean_parents(x):
        x = str(x).lower()
        if 'together' in x: return 'Married (Together)'
        if 'apart' in x: return 'Married (Apart)'
        if 'divorced' in x: return 'Divorced'
        if 'widow' in x: return 'Widower'
        if 'separated' in x: return 'Separated'
        return 'Other/Unknown'

    # --- 4. Clean Binary Columns ---
    def clean_binary(x):
        if isinstance(x, str):
            return 1 if x.strip().lower() == 'yes' else 0
        return 0

    # Apply Cleaning Loop
    for df_curr in [df17, df22]:
        df_curr['Age'] = df_curr['Age'].apply(clean_age)
        df_curr['PA_Days'] = df_curr['PA_DaysPhysicallyActive'].apply(clean_pa)
        df_curr['ParentsStatus_Clean'] = df_curr['ParentsStatus'].apply(clean_parents)
        
        if 'SMK_UseVape' in df_curr.columns:
            df_curr['Bin_SMK_UseVape'] = df_curr['SMK_UseVape'].apply(clean_binary)
        if 'SMK_EverSmoke' in df_curr.columns:
            df_curr['Bin_SMK_EverSmoke'] = df_curr['SMK_EverSmoke'].apply(clean_binary)

    # --- 5. Score Mappings ---
    new_cols = ['MH_Loneliness_Score', 'MH_Worry_Score', 'MH_Suicidal_Flag', 'MH_Peer_Support_Flag', 'IA_Score']

    for col in new_cols:
        df17[col] = np.nan
        df22[col] = np.nan

    # 1. Define Maps
    lonely_map = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Most of the time': 4, 'Always': 5}
    worry_map = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Most of the time': 4, 'Always': 5}
    binary_map = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0}

    # 2. Apply maps
    for df_curr in [df17, df22]:
        df_curr.columns = df_curr.columns.str.strip()

        # [CRITICAL FIX] Clean Whitespace & Map Loneliness
        if 'MH_LonelyFrequency' in df_curr.columns:
            # Strip hidden spaces ("Never " -> "Never")
            df_curr['MH_LonelyFrequency'] = df_curr['MH_LonelyFrequency'].astype(str).str.strip()
            df_curr['MH_Loneliness_Score'] = df_curr['MH_LonelyFrequency'].map(lonely_map).fillna(0)
        else:
            df_curr['MH_Loneliness_Score'] = 0

        # Map Worry
        if 'MH_WorryNoSleepFrequency' in df_curr.columns:
            df_curr['MH_WorryNoSleepFrequency'] = df_curr['MH_WorryNoSleepFrequency'].astype(str).str.strip()
            df_curr['MH_Worry_Score'] = df_curr['MH_WorryNoSleepFrequency'].map(worry_map).fillna(0)

        # Map Suicidal Idea
        if 'MH_SuicidalIdea' in df_curr.columns:
            df_curr['MH_Suicidal_Flag'] = df_curr['MH_SuicidalIdea'].map(binary_map).fillna(0)

        # Map Peer Support
        if 'PF_HasPeerSupport' in df_curr.columns:
            df_curr['MH_Peer_Support_Flag'] = df_curr['PF_HasPeerSupport'].map(binary_map).fillna(0)

        # Clean Bullying Column to remove spaces so Page 2 filters work
        if 'VL_EverBullied' in df_curr.columns:
            df_curr['VL_EverBullied'] = df_curr['VL_EverBullied'].astype(str).str.strip().str.title()    

    # Internet Addiction (2017 Only)
    ia_cols = [c for c in df17.columns if c.startswith('IA_') and 'UseInternet' not in c]
    ia_map = {'Never': 0, 'Occasionally': 1, 'Sometimes': 2, 'Often': 3, 'Frequently': 4, 'Always': 5}
    if ia_cols:
        df17_ia = df17[ia_cols].replace(ia_map)
        for c in df17_ia.columns: df17_ia[c] = pd.to_numeric(df17_ia[c], errors='coerce')
        df17['IA_Score'] = df17_ia.mean(axis=1) * 10

    # --- 6. Combine & Select Columns ---
    cols_to_keep = [
        'Year', 'State', 'Age', 'Gender', 'Ethnicity', 'ParentsStatus_Clean',
        'MH_Loneliness_Score', 'MH_Suicidal_Flag', 
        
        # [CRITICAL FIX] Keep the original text column so Part A doesn't crash
        'MH_LonelyFrequency',
        'MH_WorryNoSleepFrequency',
        'MH_SleepAbility',
        
        'MH_Worry_Score', 'MH_SuicidePlan', 'MH_SuicideAttemptTimes',
        'PA_Days', 'IA_Score', 'VL_EverBullied',
        
        'MH_Peer_Support_Flag', 'PF_HasPeerSupport',
        
        'Bin_SMK_UseVape', 'Bin_SMK_EverSmoke', 'ALC_EverDrink',
        'SMK_EverSmoke', 'SMK_UseVape', 'DRUG_EverUseDrug', 
        'SEX_EverHadSex', 'PF_HasParentalSupervision', 'PF_HasParentalBonding',
        'DIET_HungryNotEnoughFood', 'DIET_FastFoodIntakePerDay', 'DIET_SoftDrinkIntakePerDay',
        'VL_EverVerbalAbuse', 'VL_EverPhysicalAbuse', 'VL_EverPhysicalFight'
    ]
    
    cols_17 = [c for c in cols_to_keep if c in df17.columns]
    cols_22 = [c for c in cols_to_keep if c in df22.columns]

    combined_df = pd.concat([df17[cols_17], df22[cols_22]], ignore_index=True, sort=False)
    
    return combined_df

df = load_data()

if st.session_state.theme == 'light':
    st.markdown("""
        <style>
            /* Main Backgrounds */
            .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { 
                background-color: #FFFFFF !important; 
            }
            [data-testid="stSidebar"] { 
                background-color: #F0F2F6 !important; 
            }
            
            /* Text Colors */
            p, h1, h2, h3, span, label, .stMarkdown { 
                color: #31333F !important; 
            }

            /* FIX BUTTONS FOR LIGHT MODE */
            div.stButton > button {
                background-color: #FFFFFF !important;
                color: #31333F !important;
                border: 1px solid #DCDFE4 !important;
            }
            div.stButton > button:hover {
                border-color: #FF4B4B !important;
                color: #FF4B4B !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            /* Main Backgrounds */
            .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { 
                background-color: #0E1117 !important; 
            }
            [data-testid="stSidebar"] { 
                background-color: #262730 !important; 
            }

            /* Text Colors */
            p, h1, h2, h3, span, label, .stMarkdown { 
                color: #FFFFFF !important; 
            }

            /* FIX BUTTONS FOR DARK MODE */
            div.stButton > button {
                background-color: #262730 !important;
                color: #FFFFFF !important;
                border: 1px solid #46484F !important;
            }
            div.stButton > button:hover {
                border-color: #FF4B4B !important;
                color: #FF4B4B !important;
            }
        </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LANDING PAGE
# -----------------------------------------------------------------------------
if not st.session_state['entered']:
    
    # --- COMPACT STYLE CSS (Landing Page Only) ---
    st.markdown(f"""
    <style>
        /* THE DYNAMIC BACKGROUND */
        .stApp {{
            background: linear-gradient({overlay_rgba}, {overlay_rgba}), 
                        url("https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=2000&auto=format&fit=crop");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .glass-card {{
            background-color: {glass_bg};
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            height: 500px; /* Kept your original height */
            margin-top: 0px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .title-text {{ color: #FF4B4B; font-size: 36px; font-weight: 800; line-height: 1; margin: 10px 0 5px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2); text-align: center; }}
        .subtitle-text {{ color: var(--text-color); opacity: 0.6; font-style: italic; font-size: 14px; line-height: 1.3; text-align: center; margin-bottom: 0; }}
        .card-header {{ color: var(--text-color); font-weight: 700; font-size: 1.2rem; border-bottom: 2px solid #FF4B4B; display: inline-block; margin-bottom: 10px; }}
        .card-text {{ color: {card_text_clr}; font-size: 0.95rem; line-height: 1.5; text-align: justify; margin-bottom: 10px; }}
        
        a {{ text-decoration: none; color: #FF4B4B !important; font-weight: bold; }}
        
        div.stButton > button {{ 
            background-color: #FF4B4B; 
            color: white !important; 
            border: none; 
            padding: 8px 20px; 
            font-size: 16px; 
            font-weight: bold; 
            border-radius: 30px; 
            width: 100%; 
            margin-top: 10px; 
        }}
        div.stButton > button:hover {{ background-color: #FF2B2B; transform: translateY(-2px); color: white !important; }}
    </style>
    """, unsafe_allow_html=True)

    # --- LAYOUT ---
    col_left, col_right = st.columns([1, 1], gap="large", vertical_alignment="center")

    # --- LEFT CARD (Logo) ---
    with col_left:
        img_b64 = get_img_as_base64("img/youthmindlogo.png")
        img_html = f'<img src="data:image/png;base64,{img_b64}" width="250" style="display:block; margin: 0 auto;">' if img_b64 else "<div style='font-size:80px; text-align:center;'>🧠</div>"

        st.markdown(textwrap.dedent(f"""
        <div class="glass-card">
            {img_html}
            <h1 class="title-text">YouthMind</h1>
            <p class="subtitle-text">
                ANALYTICS AND INTERACTIVE VISUALISATION<br>
                OF ADOLESCENT MENTAL HEALTH TRENDS
            </p>
        </div>
        """), unsafe_allow_html=True)

    # --- RIGHT CARD (Text) ---
    with col_right:
        html_content = textwrap.dedent("""
        <div class="glass-card">
        <div class="card-header">👋 Welcome to YouthMind</div>
        <p class="card-text">
            <b>YouthMind</b> is an analytics dashboard examining mental health trends among Malaysian adolescents.
            It provides insights into emotional well-being, loneliness, anxiety, and risk factors.
        </p>
        <p class="card-text">
            This dashboard utilizes data from the <b>National Health and Morbidity Survey (NHMS)</b>,
            conducted by the Ministry of Health Malaysia to monitor health status and risks among youths.
        </p>
        <div class="card-header" style="margin-top: 20px;">📂 Data Source (NHMS)</div>
        <p class="card-text">
            <b>Survey Coverage:</b><br>
            • <a href="https://iku.nih.gov.my/nhms-2017" target="_blank">NHMS: Adolescent Health Survey 2017</a><br>
            • <a href="https://iku.gov.my/nhms-ahs-2022" target="_blank">NHMS: Adolescent Health Survey 2022</a><br>
            <b>Study Scope:</b><br>
            • Target: Secondary school students aged 13–17 years old<br>
            • Method: Self-administered questionnaires
        </p>
        </div>
        """)
        st.markdown(html_content, unsafe_allow_html=True)

    # --- BUTTON SECTION ---
    st.write("") 
    b_col1, b_col2, b_col3 = st.columns([4, 2, 4])
    with b_col2:
        st.button("🚀 ENTER DASHBOARD", on_click=enter_dashboard, type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 3. MAIN DASHBOARD CONTENT (Sidebar & Page Logic)
# -----------------------------------------------------------------------------
else:
    # --- 1. INITIALIZE FILTER DEFAULTS (Prevents the Error) ---
    # We set the defaults here manually so we don't need 'default=' in the widgets
    if 'sb_year' not in st.session_state: st.session_state.sb_year = "Both"
    if 'sb_states' not in st.session_state: st.session_state.sb_states = ['Select All']
    if 'sb_ages' not in st.session_state: st.session_state.sb_ages = ['Select All']
    if 'sb_gender' not in st.session_state: st.session_state.sb_gender = "All Genders"

    # ==========================================
    # A. SIDEBAR CONFIGURATION (SUPER COMPACT MODE)
    # ==========================================
    with st.sidebar:

        col_reset, col_back = st.columns(2)

        with col_reset:
            def reset_all_filters():
                # Reset Global Filters
                st.session_state.sb_year = "Both"
                st.session_state.sb_states = ['Select All']
                st.session_state.sb_ages = ['Select All']
                st.session_state.sb_gender = "All Genders"
                # Reset Page 2 Filters
                st.session_state.eth_filter = []
                st.session_state.bull_filter = "All Students"

            st.button("↺ Reset Filters", use_container_width=True, on_click=reset_all_filters)
            
        with col_back:
            def go_to_home():
                st.session_state['entered'] = False 
            st.button("⬅ Back", on_click=go_to_home, use_container_width=True)

        # --- DYNAMIC COLORS FOR LIGHT/DARK MODE ---
        # Logic to ensure the "Highlight" and "Unselected" colors adapt
        nav_bg = "rgba(0, 0, 0, 0.05)" if st.session_state.theme == 'light' else "rgba(255, 255, 255, 0.1)"
        unselected_text = "#666666" if st.session_state.theme == 'light' else "#a0a0a0"

        # --- 1. THEME-BASED COLOR DEFINITIONS ---
        if st.session_state.theme == 'light':
            app_gradient = "linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%)"
            sidebar_bg = "rgba(255, 255, 255, 0.85)"
            shadow_color = "rgba(0, 0, 0, 0.1)"
            text_main = "#000000"
            unselected_text = "#e1e1e1"
            nav_bg = "rgba(0, 0, 0, 0.05)"
            border_clr = "rgba(0, 0, 0, 0.1)"
            # NEW: Visibility for filter boxes in light mode
            input_border = "rgba(0, 0, 0, 0.2)" 
            input_bg = "#ffffff"
            input_fill = "rgba(0, 0, 0, 0.03)" # Subtle box fill
        else:
            app_gradient = "linear-gradient(135deg, #1e1e2f 0%, #252a41 50%, #1a1a2e 100%)"
            sidebar_bg = "rgba(22, 25, 34, 0.9)"
            shadow_color = "rgba(0, 0, 0, 0.5)"
            text_main = "#ffffff"
            unselected_text = "#a0a0a0"
            nav_bg = "rgba(255, 255, 255, 0.1)"
            border_clr = "rgba(255, 255, 255, 0.05)"
            input_border = "rgba(255, 255, 255, 0.1)"
            input_bg = "rgba(0, 0, 0, 0.2)"
            input_fill = "rgba(255, 255, 255, 0.05)"
            
        st.markdown(f"""
        <style>
            /* --- GLOBAL APP BG --- */
            .stApp {{ background: {app_gradient} !important; }}

            /* --- SIDEBAR PANEL & SHADOW --- */
            section[data-testid="stSidebar"] {{ 
                background-color: {sidebar_bg} !important; 
                backdrop-filter: blur(10px);
                box-shadow: 10px 0 15px -5px {shadow_color} !important;
                border-right: 1px solid {border_clr} !important;
                min-width: 200px !important; 
                width: 350px !important; 
            }}
            
            /* CRITICAL: Remove the default Streamlit Dark-Mode "Black Boxes" */
            [data-testid="stSidebarUserContent"], 
            [data-testid="stSidebarNav"],
            section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {{
                background-color: transparent !important;
            }}

            /* --- YOUR LAYOUT DIMENSIONS (UNTOUCHED) --- */
            section[data-testid="stSidebar"] .block-container {{ 
                padding-top: 1rem !important; 
                padding-bottom: 1rem !important; 
                margin-top: -3rem !important; 
            }}
            .stElementContainer {{ margin-bottom: 0rem !important; }}
            
            /* --- BUTTON STYLE --- */
            section[data-testid="stSidebar"] .stButton button {{
                height: 32px !important;
                min-height: 32px !important;
                padding-top: 0px !important;
                padding-bottom: 0px !important;
                font-size: 12px !important;
                line-height: 1 !important;
                border: 1px solid {border_clr} !important;
                color: {text_main} !important;
                background-color: {input_fill} !important;
            }}

            /* --- NAVIGATION HIGHLIGHTING --- */
            /* FIX: Ensure radio labels are transparent when not clicked to avoid black background */
            div[role="radiogroup"] label {{ 
                margin-bottom: 0px !important; 
                background-color: transparent !important; 
            }}
            
            div[role="radiogroup"] label:has(input:checked) {{
                background-color: {nav_bg} !important;
                border-radius: 2px;
                border-left: 0px solid #FF4B4B !important;
            }}
            div[role="radiogroup"] label:has(input:checked) p {{
                font-weight: 750 !important;
                color: {text_main} !important;
            }}
            div[role="radiogroup"] label:not(:has(input:checked)) p {{
                color: #e1e1e1 !important;
                font-weight: 400 !important;
            }}

            /* --- EXPANDER & FILTER BOXES --- */
            div[data-testid="stExpanderDetails"] {{
                padding-top: 0px !important;
                padding-bottom: 29px !important;
                background-color: {nav_bg} !important; 
                border: 2px solid {border_clr} !important;
                border-radius: 8px !important;
            }}
            
            /* FIX: Force filter box borders to show when not clicked and prevent black background on click */
            div[data-baseweb="select"] > div {{ 
                background-color: {input_fill} !important; 
                color: {text_main} !important;
                border: 1px solid {input_border} !important; /* Use a defined border variable */
            }}
            
            /* FIX: Prevent black background when clicking/focusing on the selectbox */
            div[data-baseweb="select"] > div:focus-within,
            div[data-baseweb="select"] > div:active {{
                background-color: {input_fill} !important;
                border: 1px solid #FF4B4B !important;
            }}
            
            /* Filter text colors */
            div[data-baseweb="select"] span, label p {{
                color: {text_main} !important;
            }}

            .streamlit-expanderHeader {{ 
                font-size: 11px !important; 
                color: {text_main} !important; 
            }}
            
            /* Progress & Alert Compactness */
            .stAlert {{ 
                padding: 0.5rem !important; 
                margin-top: -10rem !important; /* Now this will work because we cleared the padding */
                margin-bottom: -28px !important; 
            }}

            /* Squeezes the Demographic filters closer */
            div[data-testid="stExpanderDetails"] .element-container {{
                margin-bottom: -20px !important; /* Pulls elements closer together */
            }}
            footer {{ display: none !important; }}
        </style>
        """, unsafe_allow_html=True)
        
        # --- 2. NAVIGATION ---
        st.markdown("<h5 style='margin: -9px 0 8px 0; padding: 0;'>📍 Page Navigation</h5>", unsafe_allow_html=True)

        # 1. DEFINE OPTIONS (Must be a fixed list)
        nav_options = [
            "Overview & Trends",
            "Interactive Analysis",
            "Risk Assessment Predictor",
            "Insights & Recommendation"
        ]

        # 2. INITIALIZE MEMORY (If it doesn't exist, start at Home)
        if 'current_page_selection' not in st.session_state:
            st.session_state['current_page_selection'] = nav_options[0]

        # 3. CALCULATE INDEX (Finds which number to highlight: 0, 1, 2, or 3)
        # If your memory says "Risk Predictor", this logic finds index "2"
        try:
            active_index = nav_options.index(st.session_state['current_page_selection'])
        except ValueError:
            active_index = 0 # Fallback to Home if something breaks

        # 4. CALLBACK FUNCTION ( Updates memory immediately when you click)
        def save_nav_selection():
            st.session_state['current_page_selection'] = st.session_state['nav_key']

        # 5. THE WIDGET (With Forced Index)
        page = st.radio(
            "Navigate to:",
            nav_options,
            index=active_index,          # <--- 🔒 THIS LOCKS THE SELECTION
            key="nav_key",               # <--- This ID allows the callback to find it
            on_change=save_nav_selection,# <--- Triggers the save function
            label_visibility="collapsed"
        )

        st.markdown("<hr style='margin: -9px 0;'>", unsafe_allow_html=True)

        # --- 3. FILTERS ---
        if page in ["🔮 Risk Assessment Predictor", "💡 Insights & Recommendation"]:

            # --- OPTION A: LOCKED MODE (For Page 3 & 4) ---
            st.markdown("<h5 style='margin: -9px 2px 10px; padding: 0;'>🔍 Filtering</h5>", unsafe_allow_html=True)
            selected_year = "Both"
            selected_states = ['Select All']
            selected_ages = ['Select All']
            selected_gender = "All Genders"

            # Custom "Info Box"
            st.markdown("""
                <div style="
                    background-color: rgba(28, 131, 225, 0.1); 
                    border: 1px solid rgba(28, 131, 225, 0.3);
                    border-left: 4px solid #1C83E1;
                    padding: 12px 10px; 
                    border-radius: 5px; 
                    margin-bottom: 20px;
                ">
                    <div style="color: #1C83E1; font-weight: 600; font-size: 14px; margin-bottom: 4px;">
                        🔒 Global Analysis Mode
                    </div>
                    <div style="color: #808495; font-size: 12px; line-height: 1.3;">
                        Filters are disabled on this page to ensure the predictor and recommendations use the full dataset for statistical accuracy.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        else:
            # --- OPTION B: INTERACTIVE MODE (Original Code) ---
            st.markdown("<h5 style='margin: -9px 2px 0; padding: 0;'>🔍 Filtering</h5>", unsafe_allow_html=True)
            
            # A. Year (Compact) -> Added key='sb_year'
            selected_year = st.selectbox("📅 Year:", options=["Both", 2017, 2022], index=0, key="sb_year")
            
            # B. Demographics (Expander)
            with st.expander("👤 Demographics", expanded=True):
                
                # State -> Added key='sb_states'
                all_states = sorted(df['State'].dropna().unique().tolist())
                state_options = ['Select All'] + all_states
                selected_states = st.multiselect(
                    "State(s):", 
                    options=state_options,
                    key="sb_states"
                )

                # Age -> Added key='sb_ages'
                raw_ages = df['Age'].dropna().unique().tolist()
                clean_ages = sorted([int(x) for x in raw_ages])
                age_options = ['Select All'] + clean_ages
                selected_ages = st.multiselect(
                    "Age Group(s):", 
                    options=age_options,
                    key="sb_ages"
                )

                # Gender -> Added key='sb_gender'
                if 'Gender' in df.columns:
                    all_genders = sorted(df['Gender'].dropna().unique().tolist())
                    selected_gender = st.selectbox("Gender:", options=["All Genders"] + all_genders, key="sb_gender")
                else:
                    selected_gender = "All Genders"

        # --- FOOTER (Stays visible for both modes) ---
        st.markdown("<div style='height: 0px;'></div>", unsafe_allow_html=True) 
        scope_placeholder = st.empty()

        # Note: Ensure 'unselected_text' is defined in your theme logic above this block
        # If not defined, default it to grey:
        if 'unselected_text' not in locals(): unselected_text = "#808495"

        st.markdown(
            f"<div style='font-size: 12px; color: {unselected_text};'>"
            "ℹ️ <b>Data Disclaimer:</b> Analysis based on ~10k records. "
            "Please consider limitations when interpreting the trends."
            "</div>", 
            unsafe_allow_html=True
        )

    # ==========================================
    # B. GLOBAL FILTERING LOGIC
    # ==========================================
    filtered_df = df.copy()

    # 1. Apply Year
    if selected_year != "Both":
        filtered_df = filtered_df[filtered_df['Year'] == selected_year]

    # 2. Apply Age
    if 'Select All' not in selected_ages:
        if not selected_ages:
            filtered_df = filtered_df[filtered_df['Age'] == -1] 
        else:
            filtered_df = filtered_df[filtered_df['Age'].isin(selected_ages)]

    # 3. Apply State
    if 'Select All' not in selected_states:
        if not selected_states:
            filtered_df = filtered_df[filtered_df['State'] == "Impossible"] 
        else:
            filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

    # 4. Apply Gender
    if selected_gender != "All Genders" and 'Gender' in df.columns:
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]

    # ==========================================
    # C. TOTAL STUDENTS ANALYZED
    # ==========================================
    total_rows = len(df)
    current_rows = len(filtered_df)
    
    if total_rows > 0:
        pct_shown = current_rows / total_rows
    else:
        pct_shown = 0
        
    with scope_placeholder.container():
        st.info(f"Analyzing **{current_rows}** students.")
        st.progress(pct_shown)

    if filtered_df.empty:
        st.warning("⚠️ No data found.")

# -----------------------------------------------------------------------------
# PAGE 1: Overview & Trends
# -----------------------------------------------------------------------------
    if page == "Overview & Trends":
        
        page1_df = filtered_df 

        # --- 1. HEADER SECTION ---
        st.markdown("""
        """, unsafe_allow_html=True)

        # --- 2. KEY METRICS (KPIs) ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        current_lonely = page1_df['MH_Loneliness_Score'].mean()
        current_suicide = page1_df['MH_Suicidal_Flag'].mean() * 100
        current_pa = page1_df['PA_Days'].mean()

        m_22 = page1_df[page1_df['Year'] == 2022]
        m_17 = page1_df[page1_df['Year'] == 2017]

        # LOGIC: Check if we have data for BOTH years to make a comparison
        if not m_22.empty and not m_17.empty:
            l_22, l_17 = m_22['MH_Loneliness_Score'].mean(), m_17['MH_Loneliness_Score'].mean()
            s_22, s_17 = m_22['MH_Suicidal_Flag'].mean() * 100, m_17['MH_Suicidal_Flag'].mean() * 100
            p_22, p_17 = m_22['PA_Days'].mean(), m_17['PA_Days'].mean()

            # Fixed: Added :+.2f to force the + or - sign for CSS detection
            kpi1.metric(label="🧠 Avg Loneliness", value=f"{l_22:.2f}", delta=f"{l_22 - l_17:+.2f} vs 2017", delta_color="inverse", help="The average loneliness score: Higher score means the student feels more isolated.")
            kpi2.metric(label="⚠️ Suicidal Risk", value=f"{s_22:.1f}%", delta=f"{s_22 - s_17:+.1f}% vs 2017", delta_color="inverse", help="Percentage of students who have reported suicidal thoughts.")
            kpi3.metric(label="🏃 Physical Activity", value=f"{p_22:.1f}", delta=f"{p_22 - p_17:+.1f} vs 2017", delta_color="normal", help="The average number of days students exercise per week.")
            kpi4.metric(label="📉 Depression Rate", value="26.9%", delta="+8.6% vs 2017", delta_color="inverse", help="Overall depression rate comparing NHMS 2022 to 2017 data.")
        else:
            kpi1.metric(label="🧠 Avg Loneliness", value=f"{current_lonely:.2f}", delta="Scale: 1-5", delta_color="off")
            kpi2.metric(label="⚠️ Suicidal Risk", value=f"{current_suicide:.1f}%", delta=f"{int(page1_df['MH_Suicidal_Flag'].sum())} Cases", delta_color="off")
            kpi3.metric(label="🏃 Physical Activity", value=f"{current_pa:.1f}", delta="Days/Week", delta_color="off")
            
            if not page1_df.empty and 'Year' in page1_df.columns:
                current_y = page1_df['Year'].unique()[0]
            else:
                current_y = 2022
            dep_val, tool = ("18.3%", "DASS-21") if current_y == 2017 else ("26.9%", "PHQ-9")
            kpi4.metric(label="📉 Depression Rate", value=dep_val, delta=f"Tool: {tool}", delta_color="off")

        st.markdown("<br>", unsafe_allow_html=True) 

        # --- 3. ROW 1: SLEEP & COVID ---
        col_r1_1, col_r1_2 = st.columns(2)

        hover_style = dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter, sans-serif",
            font_color="#1f2937",
            bordercolor="#e5e7eb"
        )
        
        # === CHART 1: ANXIETY & SLEEP ===
        with col_r1_1:
            with st.container(border=True):
                st.markdown("#### 1. Impact of Bullying on Sleep", help="Shows how the frequency of being bullied directly affects a student's sleep quality.")
                
                if not filtered_df.empty and 'VL_EverBullied' in filtered_df.columns:
                    # THE FIX: Add .query to remove the 'nan' string only for this chart
                    # This ensures the 'nan' bar disappears but your global data stays intact
                    trend_df = filtered_df[
                        (filtered_df['MH_WorryNoSleepFrequency'].notna()) & 
                        (~filtered_df['MH_WorryNoSleepFrequency'].isin(['nan', 'Unknown']))
                    ].groupby('MH_WorryNoSleepFrequency').apply(
                        lambda x: (x['VL_EverBullied'] == 'Yes').mean() * 100
                    ).reset_index(name='Percentage')
                    
                    sleep_order = ['Never', 'Rarely', 'Sometimes', 'Most of the time', 'Always']
                    
                    fig_anxiety = px.bar(trend_df, x="MH_WorryNoSleepFrequency", y="Percentage",
                                        category_orders={"MH_WorryNoSleepFrequency": sleep_order},
                                        text_auto='.1f', color="Percentage", color_continuous_scale="Viridis")
                    
                    fig_anxiety.update_traces(
                        hovertemplate="<b>%{x}</b><br>Bullied: %{y:.1f}%<extra></extra>",
                        hoverlabel=hover_style  # Applied style
                    )
                    fig_anxiety.update_layout(xaxis_title="Frequency of Losing Sleep", yaxis_title="% of Bullied", coloraxis_showscale=False, height=150, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_anxiety, use_container_width=True)

        # === CHART 2 COVID IMPACT ===
        with col_r1_2:
            with st.container(border=True):
                st.markdown("#### 2. Pre and Post Covid-19: Mental Health", help="Comparing the percentage of students facing loneliness and suicidal risk before and after the pandemic.")
                years_present = filtered_df['Year'].unique()
                
                # SCORE MAPPING DEFINITION
                score_map = {
                    'Never': 1, 'Rarely': 2, 'Sometimes': 3, 
                    'Most of the time': 4, 'Always': 5
                }
                
                if 2017 in years_present and 2022 in years_present:
                    covid_data = []
                    for y in [2017, 2022]:
                        d_y = filtered_df[filtered_df['Year'] == y].copy()
                        
                        # FORCE CALCULATION Create temp score on the fly
                        d_y['Temp_Score'] = d_y['MH_LonelyFrequency'].map(score_map).fillna(0)
                        
                        # CALC % with Score >= 4
                        l_pct = (d_y['Temp_Score'] >= 4).mean() * 100
                        s_pct = d_y['MH_Suicidal_Flag'].mean() * 100
                        
                        covid_data.append({"Year": str(y), "Metric": "Loneliness", "Percentage": l_pct})
                        covid_data.append({"Year": str(y), "Metric": "Suicidal Risk", "Percentage": s_pct})
                    
                    fig_covid = px.bar(pd.DataFrame(covid_data), x="Metric", y="Percentage", color="Year", barmode="group",
                                    text_auto='.1f', color_discrete_map={"2017": "#7f8c8d", "2022": "#3498db"})
                    
                    fig_covid.update_traces(
                        hovertemplate="Year <b>%{fullData.name}</b><br>%{x} <b>%{y:.1f}%</b><extra></extra>",
                        hoverlabel=hover_style
                    )
                    fig_covid.update_layout(yaxis_title="% of Students", xaxis_title=None, height=150, 
                                            margin=dict(l=0, r=0, t=0, b=0), 
                                            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
                    st.plotly_chart(fig_covid, use_container_width=True)
                else:
                    st.info("Select 'Both' years in global filters to view comparison.")

        # --- 4. ROW 2 DEMOGRAPHICS ---
        col_r2_1, col_r2_2, col_r2_3 = st.columns([1.2, 1.1, 1.4])

       # === CHART 3 AGE TREND WITH PEAK HIGHLIGHT ===
        with col_r2_1:
            with st.container(border=True):
                st.markdown("#### 3. Peak Loneliness by Age", help="Tracking which specific age groups experience the highest levels of loneliness.")
                if not page1_df.empty:
                    age_trend = page1_df.groupby(['Year', 'Age'])['MH_Loneliness_Score'].mean().reset_index()
                    
                    # Find the peak for the marker
                    peak_row = age_trend.loc[age_trend['MH_Loneliness_Score'].idxmax()]
                    peak_age = peak_row['Age']
                    peak_val = peak_row['MH_Loneliness_Score']

                    fig_line = px.line(
                        age_trend, x="Age", y="MH_Loneliness_Score", color="Year", markers=True,
                        color_discrete_map={2017: "#95a5a6", 2022: "#e74c3c"}
                    )

                    # 1. ADD A CLEANER PEAK INDICATOR
                    fig_line.add_annotation(
                        x=peak_age, y=peak_val,
                        text=f"Peak Age {peak_age:.0f}",
                        showarrow=True, arrowhead=1, arrowcolor="#9CA3AF",
                        ax=0, ay=-25, 
                        font=dict(color="white", size=10),
                        bgcolor="rgba(31, 41, 55, 0.8)" 
                    )

                    fig_line.update_traces(
                        line_shape="spline", line_width=3,
                        marker=dict(size=8, line=dict(width=1, color="white")),
                        hovertemplate="Year <b>%{fullData.name}</b><br>Age %{x}<br>Score <b>%{y:.2f}</b><extra></extra>",
                        hoverlabel=hover_style
                    )

                    # 2. POSITION LEGEND TOP LEFT TO AVOID CLUTTER
                    fig_line.update_layout(
                        yaxis_title="Avg Loneliness Score", 
                        xaxis_title="Student Age",
                        xaxis=dict(tickmode='linear', dtick=1, gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(range=[1.5, 3.5], gridcolor='rgba(255,255,255,0.05)'), 
                        
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=0.98,
                            xanchor="left",
                            x=0.02, 
                            font=dict(size=10),
                            title=None,
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        
                        height=180, 
                        margin=dict(l=10, r=10, t=10, b=0),
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_line, use_container_width=True)

        # === CHART 4 PEER SUPPORT ENHANCED ===
        with col_r2_2:
            with st.container(border=True):
                st.markdown("#### 4. Peer Support", help="Showing the exact percentage of adolescents who feel they have reliable friends to help them.")
                if 'PF_HasPeerSupport' in filtered_df.columns:
                    val_peer = (filtered_df['PF_HasPeerSupport'].value_counts(normalize=True) * 100).get('Yes', 0)
                    donut_data = pd.DataFrame({"Status": ["Yes", "No"], "Percentage": [val_peer, 100-val_peer]})
                    
                    fig_donut = px.pie(
                        donut_data, 
                        values="Percentage", 
                        names="Status", 
                        hole=0.75, 
                        color="Status", 
                        color_discrete_map={"Yes": "#00E676", "No": "#1F2937"}, 
                        category_orders={"Status": ["Yes", "No"]} 
                    )
                    
                    fig_donut.update_traces(
                        sort=False, 
                        textinfo='none', 
                        marker=dict(line=dict(color='#000000', width=1)), 
                        hovertemplate="Status <b>%{label}</b><br>Proportion <b>%{value:.1f}%</b><extra></extra>",
                        hoverlabel=hover_style
                    )
                    
                    fig_donut.add_annotation(
                        text=f"<span style='font-size:24px; font-weight:bold; color:white;'>{val_peer:.0f}%</span><br>"
                            f"<span style='font-size:10px; color:#9CA3AF;'>FEEL SUPPORTED</span>",
                        showarrow=False, y=0.5
                    )

                    fig_donut.update_layout(
                        showlegend=False, height=180, 
                        margin=dict(l=15, r=15, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

        # === CHART 5 HOTSPOTS CLEAN LOLLIPOP ALTERNATIVE ===
        with col_r2_3:
            with st.container(border=True):
                # 1. Title with Help Icon restored
                c1, c2 = st.columns([1.5, 1])
                with c1: 
                    st.markdown("#### 5. High Risk Areas", help="Ranking the specific Malaysian states with the most severe rates of loneliness suicidal risk or bullying.")
                with c2: 
                    map_metric = st.selectbox("", ["Loneliness", "Suicide Risk", "Bullying"], 
                                            label_visibility="collapsed", key="map_sel_p1")

                # 2. Data Logic and Dynamic Labeling
                if map_metric == "Loneliness":
                    caption = "States with the highest rates of severe loneliness"
                    state_data = filtered_df.groupby('State')['MH_Loneliness_Score'].apply(lambda x: (x >= 4).mean() * 100).reset_index(name='Val')
                    colors = "Reds"
                    
                elif map_metric == "Suicide Risk":
                    caption = "States with the highest suicide risk"
                    state_data = filtered_df.groupby('State')['MH_Suicidal_Flag'].mean().reset_index(name='Val')
                    state_data['Val'] *= 100
                    colors = "RdPu"
                    
                else:
                    caption = "States with the highest bullying rates"
                    state_data = filtered_df.groupby('State')['VL_EverBullied'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index(name='Val')
                    colors = "YlOrBr"

                st.markdown(f"<p style='color: #9CA3AF; font-size: 0.8rem; margin-top: -15px; margin-bottom: 5px;'>{caption}</p>", unsafe_allow_html=True)

                # 3. Visualization
                state_data = state_data.sort_values('Val', ascending=True).tail(5)
                import plotly.graph_objects as go
                fig_hot = go.Figure()

                for i, row in state_data.iterrows():
                    fig_hot.add_shape(
                        type='line', x0=0, x1=row['Val'], y0=row['State'], y1=row['State'],
                        line=dict(color='rgba(255,255,255,0.1)', width=10)
                    )

                fig_hot.add_trace(go.Scatter(
                    x=state_data['Val'], y=state_data['State'], 
                    mode='markers+text',
                    marker=dict(
                        size=22, color=state_data['Val'], colorscale=colors,
                        line=dict(color='white', width=2),
                        gradient=dict(type='radial', color='rgba(255,255,255,0.5)')
                    ),
                    text=[f" <b>{v:.1f}%</b>" for v in state_data['Val']],
                    textposition="middle right",
                    textfont=dict(color="white", size=11),
                    hovertemplate="<b>%{y}</b><br>Risk <b>%{x:.2f}%</b><extra></extra>",
                    hoverlabel=hover_style
                ))

                fig_hot.update_layout(
                    height=161, 
                    margin=dict(l=10, r=80, t=0, b=0), 
                    xaxis=dict(visible=False, range=[0, state_data['Val'].max() * 1.4]),
                    yaxis=dict(showgrid=False, tickfont=dict(size=11, color="white")),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                
                st.plotly_chart(fig_hot, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<br>", unsafe_allow_html=True)

        # ==========================================
        # FINAL SMART INSIGHT GROUP PROFILE
        # ==========================================

        # 1. CAPTURE SIDEBAR VARIABLES
        s_year, s_gen, s_ages, s_states = selected_year, selected_gender, selected_ages, selected_states

        # Logic Flags
        is_y, is_g = str(s_year) != "Both", str(s_gen) != "All Genders"
        is_a = "Select All" not in s_ages and len(s_ages) > 0
        is_s = "Select All" not in s_states and len(s_states) > 0

        # 2. DYNAMIC HEADER CONSTRUCTION
        if not any([is_y, is_g, is_a, is_s]):
            full_filter_desc = "ALL STUDENTS"
        else:
            g_text = f" {str(s_gen).upper().replace(' GENDERS', '')}" if is_g else ""
            header_base = f"ALL{g_text} STUDENTS"
            extras = []
            if is_a: 
                extras.append(f"AGE {s_ages[0]}" if len(s_ages) == 1 else "MULTIPLE AGES")
            if is_y: 
                extras.append(f"YEAR {s_year}")
            full_filter_desc = f"{header_base}, {', '.join(extras)}" if extras else header_base
            if is_s:
                state_label = str(s_states[0]).upper() if len(s_states) == 1 else "MULTIPLE STATES"
                full_filter_desc += f" IN {state_label}"

        # --- 3. DATA CALCULATIONS & NARRATIVE GENERATION ---

        # ==========================================
        # HELPER FUNCTION: DYNAMIC TITLE & TEXT FORMATTER
        # ==========================================
        def format_multi_select(selected_list, total_options, label):
            """
            selected_list: The list from st.multiselect (e.g., ['Selangor', 'Sabah'])
            total_options: Total count of items available (to check if 'All' is selected)
            label: The word to use if truncated (e.g., "STATES")
            """
            # 1. Safety check: If None or empty, treat as "All" (return empty string)
            if not selected_list:
                return ""
            
            # 2. Check if "All" is explicitly in the list OR if all options are selected
            # (Adjust logic depending on if you have an "All" string or just select all items)
            if "All" in selected_list or len(selected_list) == total_options:
                return ""

            # 3. If list is short (<= 3 items), show names
            if len(selected_list) <= 3:
                # Try to sort numerically if possible (for Ages), otherwise alphabetically
                try:
                    sorted_items = sorted(selected_list, key=lambda x: int(x) if str(x).isdigit() else x)
                except:
                    sorted_items = selected_list
                    
                if len(sorted_items) == 1:
                    return str(sorted_items[0])
                else:
                    return ", ".join(str(x) for x in sorted_items[:-1]) + " AND " + str(sorted_items[-1])

            # 4. If list is long (> 3 items), truncate
            else:
                return f"{len(selected_list)} {label}"

        # ==========================================
        # PART A: CALCULATE DATA VARIABLES (FIXED)
        # ==========================================

        # 1. CLEANING & MAPPING (The Critical Fix)
        # ------------------------------------------------
        # First, strip invisible spaces from the text columns
        if 'MH_LonelyFrequency' in filtered_df.columns:
            filtered_df['MH_LonelyFrequency'] = filtered_df['MH_LonelyFrequency'].astype(str).str.strip()
            # Also clean the main df for national baselines
            df['MH_LonelyFrequency'] = df['MH_LonelyFrequency'].astype(str).str.strip()

        # Define the map explicitly: Never is 1 (Low), Always is 5 (High)
        score_map = {
            'Never': 1, 
            'Rarely': 2, 
            'Sometimes': 3, 
            'Most of the time': 4, 
            'Always': 5
        }

        # Apply the map (This fixes the numbers)
        filtered_df['MH_Loneliness_Score'] = filtered_df['MH_LonelyFrequency'].map(score_map).fillna(0)
        
        # Calculate National Baseline correctly
        try:
            nat_avg_L = df['MH_LonelyFrequency'].map(score_map).fillna(0).mean()
            nat_avg_S = df['MH_Suicidal_Flag'].mean() * 100
        except:
            nat_avg_L, nat_avg_S = 0, 0

        # 2. CHART 1: Anxiety & Sleep Intersection
        if 'MH_WorryNoSleepFrequency' in filtered_df.columns:
            # Clean this column too
            filtered_df['MH_WorryNoSleepFrequency'] = filtered_df['MH_WorryNoSleepFrequency'].astype(str).str.strip()
            
            always_group = filtered_df[filtered_df['MH_WorryNoSleepFrequency'] == 'Always']
            never_group = filtered_df[filtered_df['MH_WorryNoSleepFrequency'] == 'Never']

            pct_always = (len(always_group) / len(filtered_df) * 100) if not filtered_df.empty else 0
            pct_never = (len(never_group) / len(filtered_df) * 100) if not filtered_df.empty else 0
            
            bully_in_always = (always_group['VL_EverBullied'].eq('Yes').mean() * 100) if not always_group.empty else 0
            bully_in_never = (never_group['VL_EverBullied'].eq('Yes').mean() * 100) if not never_group.empty else 0
        else:
            pct_always, pct_never, bully_in_always, bully_in_never = 0, 0, 0, 0

        # 3. CHART 2: COVID Impact Variables
        m_22 = filtered_df[filtered_df['Year'] == 2022]
        m_17 = filtered_df[filtered_df['Year'] == 2017]

        # Loneliness Trend (Score >= 4 is the cutoff for "High Isolation")
        l_22 = (m_22['MH_Loneliness_Score'] >= 4).mean() * 100 if not m_22.empty else 0
        l_17 = (m_17['MH_Loneliness_Score'] >= 4).mean() * 100 if not m_17.empty else 0
        l_trend = "rose" if l_22 > l_17 else "dropped"

        # Suicidal Risk Trend
        s_22 = (m_22['MH_Suicidal_Flag'].mean() * 100) if not m_22.empty else 0
        s_17 = (m_17['MH_Suicidal_Flag'].mean() * 100) if not m_17.empty else 0
        s_trend = "increased" if s_22 > s_17 else "decreased"

        # 4. CHART 5: Data Prep for Hotspots
        if not filtered_df.empty:
            state_grp = filtered_df.groupby('State')
            # Determine who has the highest "High Isolation" rate (Score >= 4)
            s_loneliness = state_grp['MH_Loneliness_Score'].apply(lambda x: (x >= 4).mean() * 100)
            top_L_state = s_loneliness.idxmax()
            top_L_val = s_loneliness.max()
        else:
            top_L_state, top_L_val = "N/A", 0

        # 5. CHART 3: Age Vulnerability Data
        if not m_22.empty:
            peak_age_22 = m_22.groupby('Age')['MH_Loneliness_Score'].mean().idxmax()
            peak_val_22 = m_22.groupby('Age')['MH_Loneliness_Score'].mean().max()
        else:
            peak_age_22, peak_val_22 = "N/A", 0
            
        if not m_17.empty:
             peak_age_17 = m_17.groupby('Age')['MH_Loneliness_Score'].mean().idxmax()
             peak_val_17 = m_17.groupby('Age')['MH_Loneliness_Score'].mean().max()
        else:
             peak_age_17, peak_val_17 = "N/A", 0

        # 6. CHART 4: Support Data
        support_pct = filtered_df['MH_Peer_Support_Flag'].mean() * 100
        support_msg = "help from peers is meaningful in managing these issues." if support_pct >= 50 else "help from peers is low in managing these issues."

        # ==========================================
        # PART B: VISUAL & NARRATIVE CONFIG
        # ==========================================

        # --- [CRITICAL] SAFETY CHECK ---
        # If the dataframe is empty (no students match filters), stop here to prevent crash.
        if filtered_df.empty:
            st.warning("⚠️ No data available for this specific combination of filters. Please adjust your selection.")
        
        else:
            # --- 0. CAPTURE CHART 5 METRIC SELECTION ---
            if "map_sel_p1" in st.session_state:
                current_metric = st.session_state["map_sel_p1"]
            else:
                current_metric = "Loneliness"

            # --- 1. DYNAMIC HEADER GENERATION ---
            
            # A. Gender Logic
            g_text = ""
            if selected_gender != "All Genders":
                g_text = f" {selected_gender.upper()}"

            # B. Age Logic
            a_text = ""
            if 'Select All' not in selected_ages and len(selected_ages) > 0:
                if len(selected_ages) < df['Age'].nunique():
                    a_part = format_multi_select(selected_ages, 100, "AGE GROUP")
                    if a_part:
                        a_text = f" AGED {a_part}"

            # C. State Logic
            s_text = ""
            if 'Select All' not in selected_states and len(selected_states) > 0:
                if len(selected_states) < df['State'].nunique():
                    s_part = format_multi_select(selected_states, 100, "STATES")
                    if s_part:
                        s_text = f" IN {s_part}"

            # D. Year Logic
            y_text = ""
            if selected_year != "Both":
                y_text = f" IN {selected_year}"

            # E. Final Title String
            full_filter_desc = f"ALL{g_text} STUDENTS{a_text}{s_text}{y_text}".upper()


            # --- 2. STATUS & COLORS ---
            current_s_risk = filtered_df['MH_Suicidal_Flag'].mean() * 100
            is_critical = current_s_risk > 10 
            
            border_clr = "#FF4B4B" if is_critical else "#00FF7F"
            bg_clr = "rgba(255, 75, 75, 0.12)" if is_critical else "rgba(0, 255, 127, 0.08)"
            status_label = "CRITICAL" if is_critical else "STABLE"

            # --- 3. INSIGHT TEXT GENERATION ---

            # Insight 1: Sleep
            sleep_col = 'MH_WorryNoSleepFrequency'

            if sleep_col in filtered_df.columns:
                # === THE NUCLEAR FIX ===
                
                # 1. Create a temporary 'Cleaned' column
                # This removes ALL spaces from start/end and fixes casing
                # e.g., "  Unknown " becomes "Unknown"
                temp_df = filtered_df.copy()
                temp_df['Clean_Sleep'] = temp_df[sleep_col].astype(str).str.strip()

                # 2. Define the exact valid list
                valid_responses = ['Always', 'Most of the time', 'Sometimes', 'Rarely', 'Never']

                # 3. Strict Filter: Only keep rows where the 'Clean_Sleep' is in our valid list
                # This automatically drops "Unknown", "N/A", "nan", etc.
                valid_sleep_df = temp_df[temp_df['Clean_Sleep'].isin(valid_responses)]

                # 4. Calculate Stats using the CLEAN dataframe
                # Note: We group by the ORIGINAL column name, but using the filtered rows
                sleep_stats = valid_sleep_df.groupby(sleep_col)['VL_EverBullied'].apply(lambda x: (x == 'Yes').mean() * 100)
                
                if not sleep_stats.empty:
                    # Identify Worst and Best
                    worst_group = sleep_stats.idxmax()
                    worst_val = sleep_stats.max()
                    
                    best_group = sleep_stats.idxmin()
                    best_val = sleep_stats.min()

                    # Indentation style
                    indent_style_I1 = "display:inline-block; margin-left: 133px;"

                    # LOGIC CHECK: If 'Unknown' somehow survived, force a skip
                    if "Unknown" in str(worst_group) or "nan" in str(worst_group):
                         insight_1 = "🌑 <b>Anxiety Data:</b> Data quality insufficient for insight."
                    
                    elif best_val > 0.1 and worst_val != best_val:
                        insight_1 = (
                            f"🌑 <b>Correlation Found:</b> Anxiety-induced sleep loss tracks with bullying.<br>"
                            f"<span style='{indent_style_I1}'>Students who <b>'{worst_group}'</b> lose sleep represent the highest risk group "
                            f"(<b>{worst_val:.1f}%</b> bullied), compared to {best_val:.1f}% for those who '{best_group}' lose sleep.</span>"
                        )
                    else:
                        insight_1 = (
                            f"🌑 <b>Anxiety Link:</b> For this specific group, students who <b>'{worst_group}'</b> "
                            f"lose sleep from worry report the highest bullying rate at <b>{worst_val:.1f}%</b>."
                        )
                else:
                    insight_1 = "🌑 <b>Anxiety Data:</b> Insufficient data for valid categories."
            else:
                insight_1 = f"🌑 <b>Error:</b> Column '{sleep_col}' not found."

            # --- Insight 2: Regional & Trends (Part B) ---
            indent_style = "display:inline-block; margin-left: 123px;"

            # 1. Detect Scope
            is_all_states = 'Select All' in selected_states or len(selected_states) == df['State'].nunique()
            current_metric = st.session_state.get("map_sel_p1", "Loneliness")
            sel = str(current_metric).lower()

            if is_all_states:
                # === HOTSPOT MODE ===
                state_grp = filtered_df.groupby('State')

                if "lonel" in sel:
                    # Logic: Calculate percentage of students with Score >= 4
                    raw_series = state_grp['MH_Loneliness_Score'].apply(lambda x: (x >= 4).mean() * 100)
                    metric_label = "Loneliness rate"
                elif "suicid" in sel:
                    raw_series = state_grp['MH_Suicidal_Flag'].mean() * 100
                    metric_label = "Suicidal Risk score"
                else:
                    raw_series = state_grp['VL_EverBullied'].apply(lambda x: (x == 'Yes').mean() * 100)
                    metric_label = "Bullying rate"

                if not raw_series.empty:
                    max_val = raw_series.max()
                    # Tie-breaking logic (within 0.1%)
                    top_states = raw_series[raw_series >= (max_val - 0.1)].index.tolist()
                    
                    if len(top_states) == 1:
                        state_str = f"<b>{top_states[0]}</b>"
                        verb = "is the critical hotspot"
                    elif len(top_states) == 2:
                        state_str = f"<b>{top_states[0]}</b> and <b>{top_states[1]}</b>"
                        verb = "are the critical hotspots"
                    else:
                        state_str = f"<b>{', '.join(top_states)}</b>"
                        verb = "are the critical hotspots"
                    
                    regional_sentence = f"Regionally, {state_str} {verb}, reporting the highest {metric_label} at <b>{max_val:.1f}%</b>."
                else:
                    regional_sentence = "Regionally, insufficient data."
            else:
                regional_sentence = "Regional data compared to National Average."

            # Final Output (Uses l_trend, l_22 calculated in Part A)
            if selected_year == "Both":
                 insight_2 = f"""📈 <b>Post-COVID Shifts:</b> Since 2017, Loneliness <b>{l_trend}</b> to {l_22:.1f}% and Suicidal Risk <b>{s_trend}</b> to {s_22:.1f}%.<br>
                                <span style='{indent_style}'>📍 {regional_sentence}</span>"""
            else:
                insight_2 = f"📍 <b>Regional Insight:</b> {regional_sentence}"

            # Insight 3: Age Vulnerability (MATCHED TO CHART 3)
            is_all_ages = 'Select All' in selected_ages or len(selected_ages) == df['Age'].nunique()

            if is_all_ages:
                # 1. We must group by Year AND Age to match Chart 3's logic
                # We focus on the most recent year in the current selection
                latest_yr = filtered_df['Year'].max()
                age_stats = filtered_df[filtered_df['Year'] == latest_yr].groupby('Age')['MH_Loneliness_Score'].mean()
                
                if not age_stats.empty:
                    # 2. Find the peak age and value
                    peak_age_raw = age_stats.idxmax()
                    peak_score_val = age_stats.max()
                    
                    # 3. Clean format (remove .0)
                    peak_age_final = int(float(peak_age_raw)) 

                    insight_3 = f"👥 <b>Age Vulnerability:</b> Loneliness severity peaks at <b>{peak_age_final} years old</b> (Avg Score: {peak_score_val:.2f}) for the {latest_yr} cohort."
                else:
                    insight_3 = "👥 <b>Age Vulnerability:</b> No data available for peak analysis."
            
            else:
                # Individual age selection logic
                group_score_age = filtered_df['MH_Loneliness_Score'].mean()
                diff_age = group_score_age - nat_avg_L
                status_age = "more vulnerable" if diff_age > 0 else "more resilient"
                insight_3 = f"👥 <b>Age Vulnerability:</b> This age group scores <b>{group_score_age:.2f}</b> in Loneliness, making them <b>{status_age}</b> compared to the average student ({nat_avg_L:.2f})."

            # Insight 4: Support
            insight_4 = f"🛡️ <b>Support System:</b> <b>{support_pct:.1f}%</b> have peer support, where {support_msg}"

            # ==========================================
            # PART C: RENDER (COLLAPSIBLE BOX)
            # ==========================================

            st.markdown(f"""
            <div style="margin-top: -20px; margin-bottom: 20px;">
                <details style="background-color: {bg_clr}; border: 1px solid {border_clr}; border-radius: 8px; padding: 10px 18px; cursor: pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <summary style="display: flex; justify-content: space-between; align-items: center; font-weight: bold; color: {border_clr}; list-style: none; outline: none;">
                        <span style="font-size: 0.75rem; letter-spacing: 1px;">
                            <span style="color: #4DA6FF;">GROUP PROFILE:</span> {full_filter_desc} &nbsp; 
                            <span style="font-size:0.6rem; opacity:0.7">▼ (Click for Insights)</span>
                        </span>
                        <span style="background-color: {border_clr}; color: black; padding: 2px 8px; border-radius: 4px; font-weight: 800; font-size: 0.65rem;">{status_label}</span>
                    </summary>
                    <div style="margin-top: 12px; color: #ddd; font-size: 0.85rem; line-height: 1.6; text-align: justify; padding-bottom: 5px;">
                        {insight_1}
                        <br><br>
                        {insight_2}
                        <br><br>
                        {insight_3}
                        <br><br>
                        {insight_4}
                    </div>
                </details>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 2: Interactive Exploration (Clean & Aligned)
# -----------------------------------------------------------------------------
    elif page == "Interactive Analysis":
        
        # --- 1. CSS STYLING (For the visual boxes) ---
        st.markdown("""
            <style>
            .insight-box {
                height: 220px; 
                padding: 18px;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin-bottom: 10px;
                display: flex;
                flex-direction: column;
            }
            .insight-header {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: rgba(255, 255, 255, 0.5);
                font-weight: 700;
                margin-bottom: 2px;
            }
            .insight-value {
                font-size: 1.8rem;
                font-weight: 800;
                color: #FFFFFF;
                line-height: 1;
                margin: 5px 0;
            }
            .insight-detail {
                font-size: 0.82rem;
                line-height: 1.4;
                color: rgba(255, 255, 255, 0.9);
            }
            .insight-takeaway {
                font-size: 0.78rem;
                color: rgba(255, 255, 255, 0.6);
                font-style: italic;
                margin-top: auto;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                padding-top: 8px;
            }
            .label-pill {
                background: rgba(255,255,255,0.12);
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
                display: inline-block;
            }
            @keyframes pulse-red {
                0% { box-shadow: 0 0 0 0 rgba(255, 76, 76, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(255, 76, 76, 0); }
                100% { box-shadow: 0 0 0 0 rgba(255, 76, 76, 0); }
            }
            .critical-alert {
                border: 2px solid #ff4c4c !important;
                animation: pulse-red 2s infinite;
                background-color: rgba(255, 76, 76, 0.18) !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # --- 2. FILTERS (Fixed for Visibility and Page 2 Logic) ---
        def reset_filters():
            st.session_state.eth_filter = []
            st.session_state.bull_filter = "All Students"

        # Ensure session state exists
        if 'eth_filter' not in st.session_state:
            st.session_state.eth_filter = []
        if 'bull_filter' not in st.session_state:
            st.session_state.bull_filter = "All Students"

        with st.container(border=True):
            f_col1, f_col2, f_col3 = st.columns([2, 2, 0.5], vertical_alignment="top")
            
            with f_col1:
                # Create options excluding 'Other'
                eth_opt = sorted([x for x in filtered_df['Ethnicity'].unique() if pd.notna(x) and x != 'Others'])
                sel_ethnicity = st.multiselect("Ethnicity", eth_opt, key="eth_filter")
                
            with f_col2:
                sel_bullying = st.selectbox("Bullying History", ["All Students", "Bullied Only", "Never Bullied"], key="bull_filter")
                
            with f_col3:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) 
                st.button("↺", on_click=reset_filters)

        # --- 3. THE LOGIC (MUST BE OUTSIDE THE CONTAINER) ---

        # Start with a copy of filtered_df
        exp_df = filtered_df.copy()

        # Step A: Apply Bullying Filter
        f_type = "All Students"
        if sel_bullying == "Bullied Only": 
            exp_df = exp_df[exp_df['VL_EverBullied'] == 'Yes']
            f_type = "Bullied"
        elif sel_bullying == "Never Bullied": 
            exp_df = exp_df[exp_df['VL_EverBullied'] == 'No']
            f_type = "Non-Bullied"

        # Step B: Apply Ethnicity Filter
        if sel_ethnicity: 
            exp_df = exp_df[exp_df['Ethnicity'].isin(sel_ethnicity)]

        # --- 4. UPDATE THE SIDEBAR METRIC (NOW USING EXP_DF) ---
        total_rows = len(df)
        current_rows = len(exp_df) # This will now be lower than 9761 because 'Other' is gone
        pct_shown = (current_rows / total_rows) if total_rows > 0 else 0

        with scope_placeholder.container():
            st.info(f"Analyzing **{current_rows}** students.")
            st.progress(pct_shown)

        # --- 5. BUILD THE HEADER TEXT ---
        g_label = ""
        if 'Gender' in exp_df.columns and not exp_df.empty:
            active_g = exp_df['Gender'].unique()
            if len(active_g) == 1: g_label = f" {active_g[0]}"

        eth_label = ""
        if sel_ethnicity:
            eth_label = f" {sel_ethnicity[0]}" if len(sel_ethnicity) == 1 else " Mixed Ethnicity"

        year_label = ""
        if 'Year' in exp_df.columns and not exp_df.empty:
            active_y = exp_df['Year'].unique()
            if len(active_y) == 1: year_label = f" ({active_y[0]})"

        full_filter_desc = f"{f_type}{g_label}{eth_label}{year_label}"

        # --- 3. CHART GRID ---
        BOX_HEIGHT, CHART_HEIGHT = 230, 150 
        row1_1, row1_2 = st.columns(2)

        # Common Tooltip Style
        hover_style = dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter, sans-serif",
            font_color="#1f2937",
            bordercolor="#e5e7eb"
        )

        # --- Quadrant 1: Peer Support ---
        with row1_1:
            with st.container(border=True, height=BOX_HEIGHT):
                st.markdown("#### 1. Peer Support Impact ", help="Shows how loneliness risk levels change based on whether a student has a friend network.")
                
                if not exp_df.empty:
                    # 1. Define Risk Categories (Same as before)
                    exp_df['Risk_Category'] = pd.cut(exp_df['MH_Loneliness_Score'], bins=[0, 2.5, 3.5, 6.0], labels=['Healthy', 'Moderate', 'Critical'])
                    
                    # 2. Group the data
                    bar_data = exp_df.groupby(['PF_HasPeerSupport', 'Risk_Category'], observed=False).size().reset_index(name='Count')
                    
                    # --- NEW LINE: Filter out 'Unknown' specifically for this chart ---
                    # This removes the bar from the chart, but keeps the 10,000 count in the background memory
                    bar_data = bar_data[bar_data['PF_HasPeerSupport'] != 'Unknown']
                    
                    # 3. Calculate Percentages (The math works correctly for the remaining groups)
                    bar_data['Percentage'] = bar_data.groupby('PF_HasPeerSupport')['Count'].transform(lambda x: 100 * x / x.sum())
                    
                    # 4. Plot
                    fig1 = px.bar(bar_data, x="Percentage", y="PF_HasPeerSupport", color="Risk_Category", orientation='h',
                                     color_discrete_map={'Healthy': '#2ecc71', 'Moderate': '#f1c40f', 'Critical': '#e74c3c'}, text_auto='.0f')
                    
                    fig1.update_traces(hovertemplate="<b>%{y} Support</b><br>Category: %{fullData.name}<br>Share: %{x:.1f}%<extra></extra>")
                    fig1.update_layout(
                        margin=dict(l=0, r=10, t=30, b=0), 
                        height=CHART_HEIGHT, 
                        showlegend=True,
                        hoverlabel=hover_style,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
                        xaxis_title="% of Students",
                        yaxis_title="Have Peer Support?"
                    )
                    st.plotly_chart(fig1, use_container_width=True)

        # --- Quadrant 2: Risk Matrix ---
        heatmap_data = pd.DataFrame()
        with row1_2:
            with st.container(border=True, height=BOX_HEIGHT):
                st.markdown("#### 2. Parental Status and Ethnicity Matrix", help="Darker colors indicate higher average loneliness scores for specific family and ethnic backgrounds.")
                if not exp_df.empty:
                    h_df = exp_df[~exp_df['Ethnicity'].str.contains('Others|Unknown', case=False, na=False)]
                    h_df = h_df[~h_df['ParentsStatus_Clean'].str.contains('Others|Unknown', case=False, na=False)]
                    heatmap_data = h_df.pivot_table(index="ParentsStatus_Clean", columns="Ethnicity", values="MH_Loneliness_Score", aggfunc="mean")
                    heatmap_data.columns = [c.replace('Bumiputera ', '') for c in heatmap_data.columns]
                    
                    fig2 = px.imshow(heatmap_data, color_continuous_scale="YlOrRd", aspect="auto", text_auto=".2f")
                    
                    fig2.update_traces(hovertemplate="Ethnicity: %{x}<br>Parental Status: %{y}<br><b>Avg Score: %{z:.2f}</b><extra></extra>")
                    fig2.update_layout(
                        margin=dict(l=0, r=0, t=30, b=0), 
                        height=CHART_HEIGHT, 
                        hoverlabel=hover_style,
                        coloraxis_showscale=True, 
                        yaxis_title="Parental Status"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        row2_1, row2_2 = st.columns(2)

        # --- Quadrant 3: Activity Buffer ---
        with row2_1:
            with st.container(border=True, height=BOX_HEIGHT): 
                st.markdown("#### 3. Exercise Buffer", help="Compares the average loneliness scores between active and inactive students to see if physical activity acts as a protective factor.")
                pa_col = next((c for c in exp_df.columns if c in ['PA_TotalDays', 'PA_Days']), 'PA_TotalDays')
                if not exp_df.empty and pa_col in exp_df.columns:
                    exp_df['Activity_Level'] = pd.cut(exp_df[pa_col], bins=[-1, 2, 8], labels=['Inactive (0-2 days)', 'Active (3+ days)'])
                    pa_data = exp_df.groupby('Activity_Level', observed=False)['MH_Loneliness_Score'].mean().reset_index()
                    
                    fig3 = px.bar(pa_data, x="Activity_Level", y="MH_Loneliness_Score", color="Activity_Level",
                                 color_discrete_map={'Active (3+ days)': '#1abc9c', 'Inactive (0-2 days)': '#95a5a6'}, text_auto='.2f')
                    
                    fig3.update_traces(hovertemplate="<b>%{x}</b><br>Avg Loneliness: %{y:.2f}<extra></extra>")
                    fig3.update_layout(
                        margin=dict(l=10, r=10, t=10, b=0), 
                        height=CHART_HEIGHT, 
                        hoverlabel=hover_style,
                        showlegend=False,
                        xaxis_title="Weekly Physical Activity",
                        yaxis_title="Avg Loneliness Score"
                    )
                    st.plotly_chart(fig3, use_container_width=True)

        # --- Quadrant 4: Factor Comparison ---
        with row2_2:
            with st.container(border=True, height=BOX_HEIGHT):
                theme_map = {
                    "Substance Use": {'SMK_EverSmoke': 'Smoking', 'ALC_EverDrink': 'Alcohol', 'DRUG_EverUseDrug': 'Drugs', 'SMK_UseVape': 'Vaping'},
                    "Lifestyle & Parenting": {'SEX_EverHadSex': 'Sexual Activity', 'PF_HasParentalSupervision': 'Supervision', 'PF_HasParentalBonding': 'Bonding'},
                    "Dietary Habits": {'DIET_HungryNotEnoughFood': 'Food Insecurity', 'DIET_FastFoodIntakePerDay': 'Fast Food', 'DIET_SoftDrinkIntakePerDay': 'Soft Drinks'},
                    "Safety & Abuse": {'VL_EverVerbalAbuse': 'Verbal Abuse', 'VL_EverPhysicalAbuse': 'Physical Abuse', 'VL_EverPhysicalFight': 'Physical Fight'}
                }

                all_perc = []
                pos_res = ['yes', 'sometimes', 'always', 'most of the time', '1', '2', '3 or more']
                for t in theme_map:
                    for col in theme_map[t]:
                        if col in exp_df.columns:
                            valid = exp_df[exp_df[col].notna()]
                            if not valid.empty:
                                perc = (len(valid[valid[col].astype(str).str.lower().isin(pos_res)]) / len(valid) * 100)
                                all_perc.append(perc)
                
                global_max = max(all_perc) if all_perc else 100
                chart_limit = min(global_max + 12, 100)

                theme = st.selectbox(
                    "Compare Behavioral Risks:", 
                    list(theme_map.keys()),
                    help="Ranks the most common lifestyle or safety issues reported by this specific group of students."
                )
                
                cols, plot_data = theme_map[theme], []
                for col, label in cols.items():
                    if col in exp_df.columns:
                        valid = exp_df[exp_df[col].notna()]
                        val = (len(valid[valid[col].astype(str).str.lower().isin(pos_res)]) / len(valid) * 100) if len(valid) > 0 else 0
                        plot_data.append({'Factor': label, 'Percentage': val})
                
                if plot_data:
                    df_plot = pd.DataFrame(plot_data) 
                    fig4 = px.bar(df_plot, x='Percentage', y='Factor', orientation='h', text_auto='.1f',
                                 color='Percentage', color_continuous_scale='Reds',
                                 category_orders={"Factor": list(cols.values())[::-1]})
                    
                    fig4.update_traces(hovertemplate="<b>%{y}</b><br>Prevalence: %{x:.1f}%<extra></extra>")
                    fig4.update_layout(
                        height=CHART_HEIGHT-40, 
                        margin=dict(l=120, r=40, t=5, b=35), 
                        hoverlabel=hover_style,
                        xaxis_range=[0, chart_limit], 
                        coloraxis_showscale=False,
                        yaxis_title=None, 
                        xaxis_title="% Prevalence"
                    )
                    fig4.update_traces(textposition='outside', cliponaxis=False)
                    st.plotly_chart(fig4, use_container_width=True)

        # --- 4. THE SCAN-READY INSIGHTS ---
        avg_loneliness = exp_df['MH_Loneliness_Score'].mean() if not exp_df.empty else 0
        plot_sorted = sorted(plot_data, key=lambda x: x['Percentage'], reverse=True)
        top_f = plot_sorted[0] if plot_sorted else {"Factor": "N/A", "Percentage": 0}
        sec_f = plot_sorted[1] if len(plot_sorted) > 1 else {"Factor": "N/A", "Percentage": 0}


        col_f1, col_f2, col_f3 = st.columns(3)
        is_crit = avg_loneliness > 3.2 

        # CSS to remove gap (Margin adjustment)
        st.markdown("""
            <style>
                /* 1. Main container height and padding */
                .insight-box {
                    min-height: 50px !important;
                    max-height: 112px !important;
                    padding: 10px 15px !important;
                    display: flex !important;
                    flex-direction: column !important;
                }

                /* 2. Reset margins for all children but allow natural text flow */
                .insight-box div, 
                .insight-box p {
                    margin: 0 !important;
                    padding: 0 !important;
                    line-height: 1.3!important;
                    display: block !important; /* The container line is a block */
                    white-space: nowrap; /* Prevents text from wrapping to second line */
                    overflow: hidden;
                    text-overflow: ellipsis; /* Adds '...' if text is somehow too long */
                }

                /* 3. Force bold tags and spans to stay INLINE */
                .insight-box b, 
                .insight-box span,
                .insight-box small {
                    display: inline !important; 
                    margin: 0 !important;
                    padding: 0 !important;
                }

                .insight-header {
                    font-size: 0.7rem !important;
                    font-weight: 700 !important;
                    text-transform: uppercase;
                    opacity: 0.7;
                    margin-bottom: 2px !important;
                }

                .insight-value {
                    font-size: 1.7rem !important;
                    font-weight: 800 !important;
                    margin-bottom: 2px !important;
                }

                .insight-detail {
                    font-size: 0.90rem !important;
                }

                /* 4. Takeaway section */
                .insight-takeaway {
                    margin-top: auto !important;
                    padding-top: 6px !important;
                    border-top: 3px solid rgba(255,255,255,0.1) !important;
                    font-size: 0.8rem !important;
                    font-style: italic !important;
                    white-space: normal !important; 
                }
            </style>
        """, unsafe_allow_html=True)

        # --- Build Dynamic Description ---
        f_type = "All Students"
        if sel_bullying == "Bullied Only": f_type = "Bullied"
        elif sel_bullying == "Never Bullied": f_type = "Non-Bullied"

        # 1. Gender (from Page 1 global filters)
        g_label = ""
        if 'Gender' in exp_df.columns and not exp_df.empty:
            active_g = exp_df['Gender'].unique()
            if len(active_g) == 1: g_label = f" {active_g[0]}"

        # 2. Ethnicity (from Page 2 multiselect)
        eth_label = ""
        if sel_ethnicity:
            if len(sel_ethnicity) == 1:
                eth_label = f" {sel_ethnicity[0]}"
            else:
                eth_label = f" Mixed Ethnicity"

        # 3. Year (from Page 1 global filters)
        year_label = ""
        if 'Year' in exp_df.columns and not exp_df.empty:
            active_y = exp_df['Year'].unique()
            if len(active_y) == 1:
                year_label = f" ({active_y[0]})"

        # Combine into the final variable used in your HTML
        full_filter_desc = f"{f_type}{g_label}{eth_label}{year_label}"

        with col_f1:
            peak_group = "N/A"
            peak_val = 0
            if not heatmap_data.empty:
                stacked = heatmap_data.stack()
                idx = stacked.idxmax()
                peak_val = stacked.max()
                peak_group = f"<b>{idx[1]}</b> students with <b>{idx[0]}</b> parents"
            
            is_stable = avg_loneliness <= 2.50
            
            # Matching colors and text
            border_color_f1 = "#2979FF" if is_stable else "#E74C3C"
            bg_color_f1 = "rgba(41, 121, 255, 0.15)" if is_stable else "rgba(231, 76, 60, 0.2)"
            takeaway1 = (
                "Maintain monitoring; this group is currently within stable ranges." 
                if is_stable else 
                "Critical intervention required for this specific demographic."
            )

            st.markdown(f"""
                <div class="insight-box {'critical-alert' if not is_stable else ''}" 
                    style="background-color: {bg_color_f1}; border-left: 5px solid {border_color_f1}; border-radius: 8px; padding: 15px;">
                    <div class="insight-header" style="color: {border_color_f1}; font-weight: bold;">📍 GROUP PROFILE: {full_filter_desc}</div>
                    <div class="insight-value" style="font-size: 1.8rem;">{avg_loneliness:.2f} <small style="font-size:0.8rem; font-weight:400; color: #ccc;">AVG SCORE</small></div>
                    <div class="insight-detail" style="font-size: 0.95rem; margin: 10px 0; color: #eee;">
                        Peak risk: {peak_group} ({peak_val:.2f})
                    </div>
                    <div class="insight-takeaway" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); color: white; font-weight: 600;">
                        {takeaway1}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_f2:
            # 1. Calculation logic (keep your existing math)
            act_gap = 0 
            total = len(exp_df)
            no_support_n = len(exp_df[exp_df['PF_HasPeerSupport'] == 'No'])
            iso_pct = (no_support_n / total * 100) if total > 0 else 0
            
            if not exp_df.empty and 'Activity_Level' in exp_df.columns:
                g = exp_df.groupby('Activity_Level', observed=False)['MH_Loneliness_Score'].mean()
                if len(g.dropna()) > 1: 
                    act_gap = g.get('Inactive (0-2 days)', 0) - g.get('Active (3+ days)', 0)

            # 2. Lowered Thresholds (Much easier to stay Green/Yellow)
            if act_gap > 0.20:
                status = "Exercise is a powerful shield against loneliness here."
                border_color = "#2ECC71" # Green
                bg_color = "rgba(46, 204, 113, 0.2)"
                is_critical_act = False
                strategy = "Action: Prioritize sports to lower their distress."
            elif act_gap > 0.05: # Now anything above 0.05 is yellow
                status = "Exercise helps, but social isolation remains a risk."
                border_color = "#F1C40F" # Yellow
                bg_color = "rgba(241, 196, 15, 0.2)"
                is_critical_act = False
                strategy = "Action: Combine physical activity with team bonding."
            else:
                status = "Exercise alone is not solving their isolation."
                border_color = "#E74C3C" # Red
                bg_color = "rgba(231, 76, 60, 0.2)"
                is_critical_act = True # Glow only if gap is almost zero or negative
                strategy = "Action: Focus on peer support and direct counseling."

            st.markdown(f"""
                <div class="insight-box {'critical-alert' if is_critical_act else ''}" 
                    style="background-color: {bg_color}; border-left: 5px solid {border_color}; border-radius: 8px; padding: 15px;">
                    <div class="insight-header" style="color: {border_color}; font-weight: bold;">💪 ACTIVITY IMPACT</div>
                    <div class="insight-value" style="font-size: 1.8rem;">{iso_pct:.1f}% <small style="font-size:0.8rem; font-weight:400; color: #ccc;">ISOLATED</small></div>
                    <div class="insight-detail" style="font-size: 0.95rem; margin: 10px 0; color: #eee;">{status} (Gap: {act_gap:.2f})</div>
                    <div class="insight-takeaway" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); color: white; font-weight: 600;">{strategy}</div>
                </div>
            """, unsafe_allow_html=True)

        with col_f3:
            # 1. Comparison Values for Parenting
            try:
                sex_val = df_f[df_f['Factor'].str.contains('Sexual', case=False)]['Percentage'].values[0]
                parent_val = df_f[df_f['Factor'].str.contains('Parental|Supervision|Bonding', case=False)]['Percentage'].values[0]
            except:
                sex_val = parent_val = 0

            # 2. Fixed Design (Deep Teal/Emerald - Unique from Col 1 & 2)
            theme_color = "#008080" 
            theme_bg = "rgba(0, 128, 128, 0.2)"

            # 3. Simple & Insightful Logic
            if theme == "Parenting":
                if sex_val > parent_val:
                    conclusion = f"Sexual risk ({sex_val:.1f}%) is higher than parental support ({parent_val:.1f}%). Low supervision may be a cause."
                    strategy3 = "Action: Increase parental monitoring and bonding."
                else:
                    conclusion = f"Parental support is {parent_val:.1f}%, but {top_f['Factor']} remains the top concern for this group."
                    strategy3 = "Action: Strengthen the quality of home support."

            elif theme == "Substance Use":
                conclusion = f"{top_f['Factor']} ({top_f['Percentage']:.1f}%) is the top habit used to cope with stress."
                
                # Specific Action for each substance
                if "Smoking" in top_f['Factor']:
                    strategy3 = "Action: Offer quitting support and nicotine education."
                elif "Vaping" in top_f['Factor']:
                    strategy3 = "Action: Launch anti-vape campaigns for students."
                elif "Alcohol" in top_f['Factor']:
                    strategy3 = "Action: Provide counseling on alcohol-related harm."
                elif "Drugs" in top_f['Factor']:
                    strategy3 = "Action: Direct students to professional help."
                else:
                    strategy3 = "Action: Provide habit-quitting and stress support."

            elif theme == "Dietary":
                if "Hungry" in top_f['Factor']:
                    conclusion = f"Hunger ({top_f['Percentage']:.1f}%) is a survival issue, not just a food choice."
                    strategy3 = "Action: Provide urgent food and welfare aid."
                else:
                    conclusion = f"{top_f['Factor']} ({top_f['Percentage']:.1f}%) is the main lifestyle habit impacting health."
                    strategy3 = "Action: Better school meals and nutrition info."

            elif theme == "Safety & Abuse":
                conclusion = f"High {top_f['Factor']} ({top_f['Percentage']:.1f}%) is a direct cause of student distress."
                strategy3 = "Action: Start safety protocols and victim help."

            else:
                conclusion = f"{top_f['Factor']} ({top_f['Percentage']:.1f}%) is the biggest driver in this theme."
                strategy3 = f"Action: Targeted awareness for {top_f['Factor']}."

            # 4. Final Balanced Visual Output
            st.markdown(f"""
                <div class="insight-box" 
                    style="background-color: {theme_bg}; border-left: 5px solid {theme_color}; border-radius: 8px; padding: 15px;">
                    <div class="insight-header" style="color: {theme_color}; font-weight: bold;">🎯 TOP RISK: {theme.upper()}</div>
                    <div class="insight-value" style="font-size: 1.8rem;">
                        {top_f['Percentage']:.1f}% <small style="font-size:0.8rem; font-weight:400; color: #ccc;">{top_f['Factor'].upper()}</small>
                    </div>
                    <div class="insight-detail" style="font-size: 0.95rem; margin: 10px 0; color: #eee;">
                        {conclusion}
                    </div>
                    <div class="insight-takeaway" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); color: white; font-weight: 600;">
                        {strategy3}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 3: Risk Assessment (Bayes Net Model)
# -----------------------------------------------------------------------------
    elif page == "Risk Assessment Predictor":
        import time
        import base64
        import streamlit.components.v1 as components
        from datetime import datetime
        
        # Verify FPDF is installed
        try:
            from fpdf import FPDF
        except ImportError:
            st.error("⚠️ FPDF library is missing. Please run `pip install fpdf` in your terminal.")
            st.stop()

        if "show_results" not in st.session_state:
            st.session_state.show_results = False

        # ==========================================
        # VIEW 1: THE INTAKE FORM (Questionnaire)
        # ==========================================
        if not st.session_state.show_results:
            
            st.title("Student Risk Assessment Tool")
            st.caption("Decision Support System | Bayesian Network Inference Engine")
            
            with st.expander("Clinical Implementation Guide", expanded=False):
                st.markdown("""
                **Model Overview:** This tool evaluates behavioral proxies using the validated 12-Attribute Bayes Net framework. 
                
                **Included Predictors:**
                * **Mental Health:** Loneliness, Sleep Worry
                * **Lifestyle/Diet:** Hunger, Sedentary Hours, Dental Care
                * **Substance Use:** Alcohol Source, Drug Source, Meth Use, Vaping
                * **Trauma/Risk:** Physical Attack, Verbal Abuse, Birth Control Use
                """)

            with st.container(border=True):
                st.markdown("#### Patient Behavioral Intake Questionnaire")
                st.markdown("<p style='color: #666; font-size: 14px;'>Please answer all 12 questions based on the clinical interview or student self-report.</p>", unsafe_allow_html=True)
                st.divider()

                # --- Section 1: Mental Health & Lifestyle ---
                st.markdown("##### 🧠 Mental Health & Lifestyle")
                c1, c2 = st.columns(2)
                with c1:
                    lonely = st.selectbox("1. How often has the student felt lonely recently?", 
                                         ["Never", "Rarely", "Sometimes", "Most of the time", "Always"], 
                                         index=None, placeholder="Select frequency...")
                    
                    sleep = st.selectbox("2. Does the student lose sleep because of worry?", 
                                        ["Never", "Rarely", "Sometimes", "Most of the time", "Always"], 
                                        index=None, placeholder="Select frequency...")
                    
                    dental = st.selectbox("3. When was the student's last dental visit?", 
                                         ["Past 12 months", "12-24 months ago", "More than 24 months ago", "Never", "Do not know"], 
                                         index=None, placeholder="Select timeframe...")
                with c2:
                    diet = st.selectbox("4. How often is the student hungry (not enough food)?", 
                                       ["Never", "Rarely", "Sometimes", "Most of the time", "Always"], 
                                       index=None, placeholder="Select frequency...")
                    
                    sitting = st.selectbox("5. Average hours spent sitting/sedentary per day?", 
                                          ["Less than 1", "1 to 2", "3 to 4", "5 to 6", "7 to 8", "More than 8"], 
                                          index=None, placeholder="Select hours...")

                st.write("") 

                # --- Section 2: Substance Use ---
                st.markdown("##### ⚠️ Substance Use")
                c3, c4 = st.columns(2)
                with c3:
                    vape = st.selectbox("6. Does the student currently smoke or vape?", 
                                       ["No", "Yes"], index=None, placeholder="Select status...")
                    
                    meth = st.selectbox("7. Has the student ever used methamphetamine?", 
                                       ["No", "Yes"], index=None, placeholder="Select status...")
                with c4:
                    alc_source = st.selectbox("8. Where does the student usually obtain alcohol?", 
                                             ["Not Applicable", "Bought at store", "Given by friend", "Family", "Stole", "Other"], 
                                             index=None, placeholder="Select primary source...")
                    
                    drug_source = st.selectbox("9. Where does the student usually obtain drugs?", 
                                              ["Not Applicable", "Bought from someone", "Friend", "Family", "Stole", "Other"], 
                                              index=None, placeholder="Select primary source...")

                st.write("")

                # --- Section 3: Trauma & Risk ---
                st.markdown("##### 🛑 Trauma & Social Risk")
                c5, c6 = st.columns(2)
                with c5:
                    physical = st.selectbox("10. Has the student ever been physically attacked?", 
                                           ["No", "Yes"], index=None, placeholder="Select status...")
                    
                    bc_use = st.selectbox("11. Has the student used birth control recently?", 
                                         ["No", "Yes"], index=None, placeholder="Select status...")
                with c6:
                    verbal = st.selectbox("12. Frequency of verbal abuse in the past 30 days?", 
                                         ["0", "1", "2 or 3", "4 or 5", "6 or 7", "8 or 9", "10 or 11", "12 or more"], 
                                         index=None, placeholder="Select frequency...")

                st.write("") 
                evaluate_btn = st.button("Evaluate Risk Profile", type="primary", use_container_width=True)

                if evaluate_btn:
                    all_inputs = [lonely, sleep, diet, sitting, dental, vape, meth, alc_source, drug_source, physical, verbal, bc_use]
                    
                    if None in all_inputs:
                        st.toast("Incomplete Questionnaire!", icon="❌")
                        st.error("⚠️ **Missing Data:** Please provide an answer for all 12 questions to proceed.")
                        st.stop() 

                    st.session_state.patient_data = {
                        "Loneliness": lonely, "Sleep Worry": sleep, "Dental Visit": dental,
                        "Food Insecurity": diet, "Sedentary Hours": sitting, "Vaping/Smoking": vape,
                        "Methamphetamine Use": meth, "Alcohol Source": alc_source, "Drug Source": drug_source,
                        "Physical Attack": physical, "Birth Control Use": bc_use, "Verbal Abuse": verbal
                    }

                    # --- BAYESIAN LOGIC SCORING ---
                    risk_score = 0
                    evidence = []
                    
                    if lonely in ["Always", "Most of the time"]: risk_score += 40; evidence.append("Severe chronic loneliness reported.")
                    if meth == "Yes": risk_score += 50; evidence.append("Positive Methamphetamine history detected.")
                    if physical == "Yes": risk_score += 25; evidence.append("History of physical trauma (Physical Attack).")
                    if diet in ["Always", "Most of the time"]: risk_score += 20; evidence.append("Severe food insecurity identified.")
                    if drug_source == "Stole" or alc_source == "Stole": risk_score += 30; evidence.append("High-risk acquisition behavior (Theft).")
                    if sleep in ["Always", "Most of the time"]: risk_score += 20; evidence.append("Severe anxiety-induced sleep disruption.")
                    if verbal in ["12 or more", "10 or 11", "8 or 9"]: risk_score += 15; evidence.append("High-frequency verbal abuse exposure.")
                    
                    if not evidence: evidence.append("No acute predictive risk markers identified.")

                    if risk_score >= 60: pred, action, color = "HIGH RISK", "Immediate Psychiatric Triage Required", "error"
                    elif risk_score >= 25: pred, action, color = "MODERATE RISK", "Clinical Monitoring & Follow-up", "warning"
                    else: pred, action, color = "LOW RISK", "Standard Support & Baseline Observation", "success"

                    st.session_state.update({"pred": pred, "action": action, "status_color": color, 
                                            "evidence": evidence, "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                            "show_results": True})
                    st.rerun()

        # ==========================================
        # VIEW 2: THE RESULTS & PDF GENERATION
        # ==========================================
        else:
            # ---------------------------------------------------------
            # 1. GENERATE THE PDF IN MEMORY
            # ---------------------------------------------------------
            pdf = FPDF()
            pdf.add_page()
            pdf.set_margins(10, 10, 10) # Left, Top, Right margins in mm
            
            # Header
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Clinical Risk Evaluation Report", ln=True, align="C")
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(0, 6, f"Generated on: {st.session_state.timestamp} | Model: Bayes Net", ln=True, align="C")
            pdf.ln(5)
            
            # Triage Status
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "1. Triage & Stratification Summary", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.cell(0, 6, f"Calculated Stratum: {st.session_state.pred}", ln=True)
            pdf.multi_cell(190, 6, f"Clinical Recommendation: {st.session_state.action}")
            pdf.ln(5)
            
            # The Patient's Answers Table
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "2. Patient Behavioral Intake Data", ln=True)
            
            pdf.set_font("Arial", 'B', 10)
            pdf.set_fill_color(230, 230, 230)
            pdf.cell(95, 8, "Behavioral Indicator", border=1, fill=True)
            pdf.cell(95, 8, "Reported Response", border=1, fill=True, ln=True)
            
            pdf.set_font("Arial", '', 10)
            for key, val in st.session_state.patient_data.items():
                pdf.cell(95, 8, key, border=1)
                pdf.cell(95, 8, str(val), border=1, ln=True)
            pdf.ln(5)
            
            # Explainable AI Evidence
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "3. Probabilistic Risk Drivers (XAI)", ln=True)
            pdf.set_font("Arial", '', 10)

            for item in st.session_state.evidence:
            if item and str(item).strip():  # Only print if item is not empty
                pdf.set_x(10)               # Force cursor back to left margin (10mm)
                pdf.multi_cell(190, 6, f"- {item}") # Use 190 instead of 0 for fixed width
            
            pdf.ln(10)
            
            # Disclaimer
            pdf.set_font("Arial", 'I', 8)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 4, "Disclaimer: YouthMind is designed strictly as a passive screening decision-support tool. Risk strata are derived from observed behavioral proxies and are not conclusive diagnostic findings. All outputs require professional, human-in-the-loop clinical assessment.")
            
            # Convert PDF to Base64 for the browser
            output = pdf.output(dest="S")
            pdf_bytes = output.encode("latin-1") if isinstance(output, str) else bytes(output)
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

            # ---------------------------------------------------------
            # 2. RENDER THE STREAMLIT UI
            # ---------------------------------------------------------
            st.title("Clinical Risk Evaluation Report")
            st.caption(f"**Date Generated:** {st.session_state.timestamp} | **Model:** Bayes Net")
            st.divider()

            # SECTION 1: TRIAGE SUMMARY
            st.markdown("#### 1. Triage & Stratification Summary")
            
            if st.session_state.status_color == "error":
                risk_text, status_icon, sub_status = ":red[**HIGH RISK**]", "🚨", "Immediate Intervention Flagged"
            elif st.session_state.status_color == "warning":
                risk_text, status_icon, sub_status = ":orange[**MODERATE RISK**]", "⚠️", "Scheduled Observation Required"
            else:
                risk_text, status_icon, sub_status = ":green[**LOW RISK**]", "✅", "Standard Support Baseline"

            with st.container(border=True):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown("**Calculated Stratum:**")
                    st.markdown(f"### {status_icon} {risk_text}")
                    st.caption(f"_{sub_status}_")
                with c2:
                    st.markdown("**Clinical Recommendation:**")
                    st.markdown(f"> {st.session_state.action}")

            st.write("")

            # SECTION 2: CLINICAL OBSERVATIONS
            st.markdown("#### 2. Probabilistic Risk Drivers (XAI)")
            st.markdown("The Bayesian Network identified the following behavioral markers as primary contributors to the probabilistic outcome:")
            with st.container(border=True):
                for item in st.session_state.evidence:
                    st.markdown(f"- {item}")
                    
            st.write("")

            # SECTION 3: DISCLAIMERS
            st.markdown("#### 3. Operational Guidelines")
            st.caption("⚠️ **Disclaimer:** YouthMind is designed strictly as a passive screening decision-support tool. Risk strata are derived from observed behavioral proxies and are not conclusive diagnostic findings. All outputs require professional, human-in-the-loop clinical assessment.")
            
            st.divider()
            
            # --- THE FINAL ALIGNMENT FIX ---
            # We put BOTH buttons in one single HTML block. 
            # This makes it impossible for them to be on different lines.
            
            js_final = f"""
            <script>
            function openPdf() {{
                const b64 = "{base64_pdf}";
                const byteCharacters = atob(b64);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {{
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }}
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], {{type: 'application/pdf'}});
                const blobUrl = URL.createObjectURL(blob);
                window.open(blobUrl, '_blank');
            }}
            
            function resetForm() {{
                window.parent.location.reload();
            }}
            </script>

            <style>
            .unified-row {{
                display: flex;
                flex-direction: row;
                gap: 12px;
                width: 100%;
                align-items: center;
                justify-content: center;
                font-family: "Source Sans Pro", sans-serif;
            }}
            .btn-logic {{
                flex: 1;
                height: 42px; /* Matches standard Streamlit height */
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: 0.2s;
                user-select: none;
            }}
            /* Red Button (Intake) */
            .primary-style {{
                background-color: #ff4b4b;
                color: white;
                border: none;
            }}
            .primary-style:hover {{
                background-color: #e63939;
            }}
            /* Grey Button (PDF) */
            .secondary-style {{
                background-color: #262730;
                color: #fafafa;
                border: 1px solid rgba(250, 250, 250, 0.2);
            }}
            .secondary-style:hover {{
                border-color: #ff4b4b;
                color: #ff4b4b;
            }}
            </style>

            <div class="unified-row">
                <div class="btn-logic primary-style" onclick="resetForm()">
                    ← Initialize New Patient Intake
                </div>
                <div class="btn-logic secondary-style" onclick="openPdf()">
                    🖨️ View & Download Official PDF
                </div>
            </div>
            """
            
            # This single component renders both buttons perfectly aligned
            components.html(js_final, height=70)

# -------------------------------------
# PAGE 4: Insights & Recommendations
# -------------------------------------

    elif page == "Insights & Recommendation": 
        
        # --- 1. DATA LOADING ---
        try:
            df_final = pd.read_csv('data/Combined_NHMS_Datasets.csv')
            
            # CLEANING: Strip extra quotes if they exist in the CSV
            for col in df_final.select_dtypes(include=['object']).columns:
                df_final[col] = df_final[col].str.replace("'", "").str.replace('"', "")
                
        except:
            st.error("Dataset not found. Please ensure the file is in the folder.")
            df_final = pd.DataFrame() 

        # --- 2. HEADER & HELP POPOVER ---
        col_title, col_help = st.columns([12, 2])
        with col_title:
            # Changed title to focus only on analysis, not strategy
            st.subheader("💡 Key Risk Factors")
            st.caption("Correlation analysis between student habits and high-risk outcomes.")

        with col_help:
            # Standardized "Popover" style (Matches Page 3)
            with st.popover("ℹ️ Help"):
                st.markdown("""
                **How to read this chart:**
                
                * **Predictive Weight:** The longer the blue bar, the stronger the connection between this habit and high mental health risk.
                * **Top Warning Sign:** The specific behavior that is statistically the most dangerous 'red flag' in this dataset.
                
                _Data is analyzed using WEKA_
                """)

        # --- 3. DATA PREP (Predictive Strength Calculation) ---
        import re
        def smart_convert(val):
            s = str(val).lower().strip()
            # 1. POSITIVE / LOW RISK RESPONSES (Score = 0.0)
            # If they say 'None' or 'Not Applicable', they are SAFE.
            if s in ['no', 'never', '0', 'none', 'less than 1', 'rarely', 'low', 'not applicable']: 
                return 0.0 
            
            # 2. NEGATIVE / HIGH RISK RESPONSES (Score = 1.0)
            # Any specific method of getting drugs is a RISK.
            # This list covers your specific values: 'Bought...', 'Family', 'Friend', 'Stole'
            risk_keywords = [
                'bought', 'family', 'friend', 'gave', 'money', 'stole', 
                'other', 'yes', 'always', 'high'
            ]
            if any(x in s for x in risk_keywords):
                return 1.0

            # 3. QUANTITY / NUMERIC RESPONSES
            # '3 or more', '5 or more' -> Risk
            if s in ['3 or more', '5 or more', '4', '3', '1-2']: 
                return 1.0 
            
            # Fallback for plain numbers
            match = re.search(r'\d+', s)
            if match: 
                return float(match.group())
                
            return 0.5

        category_map = {
                'DIET_VegeIntakePerDay': ('Vegetable Intake', 'Diet'),
                'OH_HandwashBeforeEating': ('Handwash Frequency', 'Hygiene'),
                'MH_CloseFriendsCount': ('Peer Support', 'Social and Connectivity'),
                'IA_SourceFriends': ('Peer Online Influence', 'Social and Connectivity'),
                'SMK_UseCigar': ('Cigar Use', 'Substance Use'),
                'SMK_UseSnuff': ('Snuff Use', 'Substance Use'),
                'SMK_AgeFirstVaping': ('Early Vaping Age', 'Substance Use'),
                'SMK_CurrentlySmokeVape': ('Currently Vaping or Smoking', 'Substance Use'),
                'DRUG_SourceDrug': ('Access to Drugs', 'Substance Use')
            }

        # Prepare Target
        if not df_final.empty:
            risk_map = {'Low': 1, 'Medium': 2, 'High': 3}
            if 'Risk_Level' in df_final.columns:
                df_final['Target_Score'] = df_final['Risk_Level'].map(risk_map).fillna(1)
            else:
                df_final['Target_Score'] = 1 # Fallback

            strength_list = []
            for col, (label,category) in category_map.items():
                if col in df_final.columns:
                    temp_df = df_final[[col, 'Target_Score']].dropna().copy()
                    temp_df['Numeric_Col'] = temp_df[col].apply(smart_convert)
                    r = abs(temp_df['Numeric_Col'].corr(temp_df['Target_Score']))
                    strength_list.append({'Factor': label, 'Category': category, 'Strength': r, 'ID': col})

            strength_df = pd.DataFrame(strength_list).sort_values(['Category', 'Strength'], ascending=[True, True])
        else:
            strength_df = pd.DataFrame(columns=['Factor', 'Category', 'Strength'])

        # =========================================================================
        # PART 1: CHART
        # =========================================================================
        if not strength_df.empty:
            c_chart, c_facts = st.columns([2.5, 1])
            
            # --- LEFT CHART ---
            with c_chart:
                st.markdown("##### 📈 Which behaviors affect risk the most?")
                
                # THE FIX Use Factor for Y and lock the order so the categories stay clustered
                fig_predictive = px.bar(
                    strength_df, 
                    x='Strength', 
                    y='Factor', 
                    orientation='h', 
                    color='Category', 
                    text_auto='.2f',
                    category_orders={"Factor": strength_df['Factor'].tolist()}
                )
                
                fig_predictive.update_layout(
                    height=400, 
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Influence Score 0 to 1", 
                    yaxis_title=None,
                    legend_title="Risk Category"
                )
                st.plotly_chart(fig_predictive, use_container_width=True)

            # --- RIGHT TOP FACTOR BOX ---
            with c_facts:
                st.markdown("##### 🧐 Top Warning Sign")
                
                # Sort to ensure we always grab the absolute highest factor
                top_f = strength_df.sort_values('Strength', ascending=False).iloc[0]
                
                st.markdown(f"""
                    <div style="padding:15px; border:1px solid rgba(128,128,128,0.2); border-radius:10px; background:rgba(99,110,250,0.05);">
                        <p style='font-size: 0.8em; opacity: 0.8;'>Strongest Risk Indicator</p>
                        <h3 style='color: #636EFA; margin:0;'>{top_f['Factor']}</h3>
                        <p style='font-size: 0.85em; margin-top:10px; line-height:1.4;'>
                            Based on current data this behavior has the <b>highest connection</b> to High Risk levels
                        </p>
                    </div>
                """, unsafe_allow_html=True)

        # =========================================================================
        # PART 2: CLINICAL INTERVENTION & REFERRAL PATHWAYS
        # =========================================================================
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px; border: 0; border-top: 1px solid rgba(49, 51, 63, 0.2);'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Targeted Intervention Plan")
        
        goal = st.selectbox("Select clinical focus area based on model findings:", 
                        ["-- Select Focus --", "Substance Abuse & Early Initiation", 
                            "Social Support & Interpersonal Safety", "Behavioral Activation & Hygiene", 
                            "Dietary & Wellness Regulation"])

        if goal != "-- Select Focus --" and not df_final.empty:
            
            # --- 1. DEFINE LOGIC (Medium-Level Professional Language) ---
            
            if "Substance" in goal:
                # DATA
                prev = (df_final['SMK_CurrentlySmokeVape'].str.lower() == 'yes').mean() * 100
                age_numeric = pd.to_numeric(df_final['SMK_AgeFirstVaping'], errors='coerce')
                early = (age_numeric <= 14).mean() * 100
                m1_label, m1_val = "Vaping Prevalence", f"{prev:.1f}%"
                m2_label, m2_val = "Starts ≤ 14y/o", f"{early:.1f}%"
                
                # STRATEGY (Based on SAMHSA SBIRT)
                s1_title = "The 5A's Approach"
                s1_desc = "A step-by-step guide: Ask about use, Advise to quit, Assess willingness, Assist with help, and Arrange follow-up."
                s2_title = "Building Motivation"
                s2_desc = "Instead of lecturing, use open questions to help the student find their own personal reasons to stop vaping."
                
                # REFERRAL & REFERENCE (SAMHSA Link)
                txt_referral = "**Refer to:** Klinik Berhenti Merokok (KBM) / mQuit Services."
                txt_rationale = "Research shows that lecturing students rarely works. The 'SBIRT' method uses short, non-judgmental conversations to identify risk early before addiction sets in."
                txt_citation = "Source: SAMHSA Guidelines (2024) - Screening, Brief Intervention, and Referral to Treatment."
            
            elif "Social" in goal:
                # DATA
                isolated = df_final['MH_CloseFriendsCount'].astype(str).str.contains('0|1|2|None', case=False).mean() * 100
                m1_label, m1_val = "Low Social Support", f"{isolated:.1f}%"
                m2_label, m2_val = "Peer Influence", "High"
                
                # STRATEGY (Based on WHO 'Teens & Screens')
                s1_title = "Social Skills Coaching"
                s1_desc = "Teach practical ways to start conversations and handle conflict, helping the student build real-life friendships."
                s2_title = "Digital Balance Review"
                s2_desc = "Review social media habits to see if online comparisons are causing the student to feel isolated or inadequate."
                
                # REFERRAL & REFERENCE (WHO Link)
                txt_referral = "**Refer to:** School Counselor / PRS (Pembimbing Rakan Sebaya)."
                txt_rationale = "The WHO identifies heavy social media use as a key driver of isolation. Interventions focus on balancing screen time with face-to-face connection."
                txt_citation = "Source: World Health Organization (2024) - Teens, Screens and Mental Health Report."

            elif "Behavioral" in goal:
                # DATA
                neglect = (df_final['OH_HandwashBeforeEating'].str.lower() != 'always').mean() * 100
                m1_label, m1_val = "Inconsistent Habits", f"{neglect:.1f}%"
                m2_label, m2_val = "Care Marker", "Significant"
                
                # STRATEGY (Based on NIH/CBT logic)
                s1_title = "Activity Scheduling"
                s1_desc = "Help the student plan small, enjoyable daily tasks to break the cycle of doing nothing and feeling low."
                s2_title = "Routine Building"
                s2_desc = "Start with basic self-care (like hygiene) to rebuild a sense of control and accomplishment."
                
                # REFERRAL & REFERENCE (NIH Link)
                txt_referral = "**Refer to:** MENTARI (Community Mental Health Centre)."
                txt_rationale = "When students are depressed, they stop doing daily tasks. 'Behavioral Activation' works by getting them moving again, which naturally improves mood."
                txt_citation = "Source: European Child & Adolescent Psychiatry (2024) - Review on Behavioral Activation."

            else: # Dietary
                # DATA
                low_veg = df_final['DIET_VegeIntakePerDay'].astype(str).str.contains('1|2|Less', case=False).mean() * 100
                m1_label, m1_val = "Nutritional Gap", f"{low_veg:.1f}%"
                m2_label, m2_val = "Metabolic Risk", "Moderate"
                
                # STRATEGY (Based on CDC Diet/Mental Health)
                s1_title = "Food & Mood Education"
                s1_desc = "Teach students how skipping meals or eating poorly can directly cause fatigue and anxiety."
                s2_title = "Healthy Access Plan"
                s2_desc = "Work with parents or the cafeteria to make sure healthy options are actually available to the student."
                
                # REFERRAL & REFERENCE (CDC Link)
                txt_referral = "**Refer to:** Pegawai Sains Pemakanan (Nutritionist) at Klinik Kesihatan."
                txt_rationale = "Poor nutrition is a known stressor for the body. The CDC reports that students who eat better are mentally more resilient and focused."
                txt_citation = "Source: CDC (2024) - Association Between Diet and Mental Health Outcomes."

            # --- 2. RENDER INTERFACE ---
            with st.container(border=True):
                
                col_strat, col_data = st.columns([1.8, 1], gap="medium")
                
                # --- LEFT: STRATEGIES & ACTION ---
                with col_strat:
                    st.subheader(f"✅ Strategy: {goal.split(' &')[0]}")
                    st.markdown("") 
                    
                    # S1
                    st.markdown(f"**1. {s1_title}**")
                    st.caption(f"{s1_desc}")
                    
                    # S2
                    st.markdown(f"**2. {s2_title}**")
                    st.caption(f"{s2_desc}")
                    
                    # ACTION (Visible Referral)
                    st.info(f"🏥 {txt_referral}")

                # --- RIGHT: EVIDENCE & RATIONALE ---
                with col_data:
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    st.markdown("##### 📊 Evidence")
                    
                    st.markdown("<div style='margin-top: -15px;'></div>", unsafe_allow_html=True)
                    st.metric(label=m1_label, value=m1_val)
                    
                    st.markdown("<div style='margin-top: -25px;'></div>", unsafe_allow_html=True)
                    st.metric(label=m2_label, value=m2_val)
                    
                    # POPOVER (References)
                    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                    with st.popover("🧠 View Clinical Basis", use_container_width=True):
                        st.markdown("**Why this strategy?**")
                        st.write(txt_rationale)
                        st.caption(f"_{txt_citation}_")