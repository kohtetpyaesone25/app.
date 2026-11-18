import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Universal AI Trade & Data Analysis System",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Universal AI Trade & Data Analysis & Forecasting System")
st.markdown("---")

# ---------------------------------------------------------
# Helper Functions & Robust Caching
# ---------------------------------------------------------
@st.cache_data
def load_raw_and_clean_data(uploaded_file):
    """Load both Raw and Clean versions of the dataset"""
    df_raw_orig = pd.read_csv(uploaded_file)
    df_clean = df_raw_orig.copy()
    df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
    
    unwanted_cols = [
        c for c in df_clean.columns 
        if 'unnamed' in c or (c == 'index' and df_clean[c].is_unique and df_clean[c].dtype in ['int64', 'float64'])
    ]
    if unwanted_cols:
        df_clean = df_clean.drop(columns=unwanted_cols)
        
    return df_raw_orig, df_clean

@st.cache_data
def process_trade_data(df):
    """Automated Feature Engineering for Trade Balance and Total Trade"""
    processed_df = df.copy().fillna(0)
    col_names_lower = [c.lower() for c in processed_df.columns]
    if 'export' in col_names_lower and 'import' in col_names_lower:
        exp_col = processed_df.columns[col_names_lower.index('export')]
        imp_col = processed_df.columns[col_names_lower.index('import')]
        processed_df['total_trade'] = processed_df[exp_col] + processed_df[imp_col]
        processed_df['trade_balance'] = processed_df[exp_col] - processed_df[imp_col]
    return processed_df

def generate_future_labels(last_period, steps, p_type):
    """Auto-detect Monthly / Yearly period format and generate dynamic future labels"""
    last_str = str(last_period).strip()

    match_range = re.match(r'^(\d{4})[-/](\d{2,4})$', last_str)
    if match_range:
        y1, y2 = int(match_range.group(1)), int(match_range.group(2))
        is_short = len(match_range.group(2)) == 2
        diff = (2000 + y2 - y1) if is_short else (y2 - y1)
        if diff <= 0:
            diff = 1
        
        curr_y1, curr_y2 = y1, y2
        future_labels = []
        for _ in range(steps):
            curr_y1 += diff
            curr_y2 += diff
            if is_short:
                future_labels.append(f"{curr_y1}-{str(curr_y2)[-2:]}")
            else:
                future_labels.append(f"{curr_y1}-{curr_y2}")
        return future_labels

    if p_type == 'Monthly':
        try:
            parsed_date = pd.to_datetime(last_str, format='%b-%y', errors='coerce')
            if pd.isna(parsed_date):
                parsed_date = pd.to_datetime(last_str, errors='coerce')
            if not pd.isna(parsed_date):
                dates = pd.date_range(start=parsed_date + pd.DateOffset(months=1), periods=steps, freq='MS')
                return [d.strftime('%b-%y') for d in dates]
        except Exception:
            pass

    if last_str.isdigit() and len(last_str) == 4:
        start_y = int(last_str)
        return [str(start_y + i + 1) for i in range(steps)]

    prefix = "M" if p_type == 'Monthly' else "Y"
    return [f"{prefix}_Pred_{i+1}" for i in range(steps)]

def apply_advanced_layout(fig, title_text, x_title, y_title):
    fig.update_layout(
        title=title_text,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="LightGray",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(family="sans-serif", size=12, color="#1e293b"),
        hovermode="x unified"
    )
    return fig

# ---------------------------------------------------------
# Sidebar Controls & Dataset Upload
# ---------------------------------------------------------
st.sidebar.header("⚙️ Data & Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Any CSV Dataset (Trade, Economic, Social, Survey)", 
    type=['csv']
)

if uploaded_file is not None:
    try:
        df_raw_orig, df_clean = load_raw_and_clean_data(uploaded_file)
        st.sidebar.success("✅ CSV File Loaded Successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        st.stop()
else:
    st.sidebar.info("💡 Default Sample Trade Dataset ကို အသုံးပြုထားပါသည်။ မိမိပိုင် CSV ဖိုင်ကို အထက်တွင် Upload လုပ်နိုင်ပါသည်။")
    dates = pd.date_range(start="2021-01-01", periods=36, freq="MS")
    np.random.seed(42)
    df_clean = pd.DataFrame({
        'period': [d.strftime('%b-%y') for d in dates],
        'year': [d.year for d in dates],
        'product': np.random.choice(['Rice', 'Bean', 'Corn', 'Rubber'], size=36),
        'export': np.linspace(500, 900, 36) + np.random.normal(0, 30, 36),
        'import': np.linspace(450, 800, 36) + np.random.normal(0, 40, 36),
        'local_consumption': np.linspace(200, 400, 36) + np.random.normal(0, 10, 36)
    })
    df_raw_orig = df_clean.copy()

if df_clean.empty:
    st.error("❌ တင်သွင်းလာသော Dataset ထဲတွင် အချက်အလက်များ လုံးဝမပါဝင်ပါ။")
    st.stop()

# ---------------------------------------------------------
# 1. Period Column & Period Type Selection (Matches Jupyter UI)
# ---------------------------------------------------------
st.sidebar.subheader("📌 Period & Column Selection")
default_x_idx = df_clean.columns.tolist().index('period') if 'period' in df_clean.columns else 0
x_col = st.sidebar.selectbox("Period Col (X-axis):", options=df_clean.columns.tolist(), index=default_x_idx)
period_type = st.sidebar.selectbox("Period Type:", options=['Monthly', 'Yearly'], index=0, key="period_type_dropdown")

# ---------------------------------------------------------
# 2. Numeric Columns (Multiple) - Default Export & Import Only
# ---------------------------------------------------------
with st.sidebar.expander("📊 Numeric Columns (Multiple)", expanded=True):
    numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    selected_numerics = []
    has_exp_imp = any(c.lower() in ['export', 'import'] for c in numeric_cols)
    
    for i, num_col in enumerate(numeric_cols):
        default_val = (num_col.lower() in ['export', 'import']) if has_exp_imp else (i == 0)
        if st.checkbox(f"📈 {num_col}", value=default_val, key=f"num_select_chk_{num_col}"):
            selected_numerics.append(num_col)

if not selected_numerics:
    st.warning("⚠️ စစ်ဆေးရန် အနည်းဆုံး Numeric Column တစ်ခု ရွေးချယ်ပေးပါ။")
    st.stop()

# ---------------------------------------------------------
# 3. Threshold Slider (Matches Jupyter UI)
# ---------------------------------------------------------
primary_num = selected_numerics[0]
min_thresh = float(df_clean[primary_num].min())
max_thresh = float(df_clean[primary_num].max())

if min_thresh < max_thresh:
    threshold_val = st.sidebar.slider(
        f"🎚️ Threshold ({primary_num} ≥):", 
        min_value=min_thresh, 
        max_value=max_thresh, 
        value=min_thresh
    )
else:
    threshold_val = min_thresh

# ---------------------------------------------------------
# 4. Category Columns & Category Values Filter with "Select All"
# ---------------------------------------------------------
categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
for col in numeric_cols:
    if df_clean[col].nunique() <= 15 and col not in categorical_cols and col not in selected_numerics:
        categorical_cols.append(col)

with st.sidebar.expander("🗂️ Category Columns", expanded=False):
    active_cat_cols = []
    for cat_col in categorical_cols:
        if cat_col != x_col:
            if st.checkbox(f"{cat_col}", value=False, key=f"enable_cat_{cat_col}"):
                active_cat_cols.append(cat_col)

filtered_df = process_trade_data(df_clean)
filtered_df = filtered_df[filtered_df[primary_num] >= threshold_val]

if active_cat_cols:
    with st.sidebar.expander("🎯 Category Values Filter", expanded=True):
        for cat_col in active_cat_cols:
            st.markdown(f"**Category: `{cat_col}`**")
            unique_vals = sorted(df_clean[cat_col].dropna().unique().tolist(), key=str)
            
            select_all = st.checkbox("Select All", value=True, key=f"select_all_{cat_col}")
            selected_vals = []
            for val in unique_vals:
                if st.checkbox(f"{val}", value=select_all, key=f"val_chk_filter_{cat_col}_{val}"):
                    selected_vals.append(val)
            
            if selected_vals:
                filtered_df = filtered_df[filtered_df[cat_col].isin(selected_vals)]
            else:
                filtered_df = filtered_df.iloc[0:0]
            st.markdown("---")

# ---------------------------------------------------------
# 5. Statistical Calculation Modes & Grouping
# ---------------------------------------------------------
st.sidebar.subheader("📈 Statistical Analysis Mode")
analysis_mode = st.sidebar.selectbox(
    "Select Calculation Mode:", 
    options=["Raw Values", "Mean Aggregation", "Percentage Change (%)"],
    index=0
)

group_by_col = st.sidebar.selectbox("Group By Category (Optional):", options=['None'] + categorical_cols)

if not filtered_df.empty:
    if analysis_mode == "Mean Aggregation":
        if group_by_col != 'None' and group_by_col != x_col:
            filtered_df = filtered_df.groupby([x_col, group_by_col], as_index=False)[selected_numerics].mean()
        else:
            filtered_df = filtered_df.groupby(x_col, as_index=False)[selected_numerics].mean()
    elif analysis_mode == "Percentage Change (%)":
        for col in selected_numerics:
            filtered_df[col] = filtered_df[col].pct_change().fillna(0) * 100
    else:
        if group_by_col != 'None' and group_by_col != x_col:
            filtered_df = filtered_df.groupby([x_col, group_by_col], as_index=False)[selected_numerics].sum()

    # Chronological Date / Month Sorting
    if period_type == 'Monthly':
        parsed_dates = pd.to_datetime(filtered_df[x_col], format='%b-%y', errors='coerce').fillna(
            pd.to_datetime(filtered_df[x_col], errors='coerce')
        )
        if not parsed_dates.isna().all():
            filtered_df['__sort_dt'] = parsed_dates
            filtered_df = filtered_df.sort_values(by='__sort_dt').drop(columns=['__sort_dt'])
        else:
            month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
                           'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            if filtered_df[x_col].astype(str).str.lower().isin(month_order.keys()).any():
                filtered_df['__month_sort'] = filtered_df[x_col].astype(str).str.lower().map(month_order)
                filtered_df = filtered_df.sort_values(by='__month_sort').drop(columns=['__month_sort'])
    else:
        filtered_df = filtered_df.sort_values(by=x_col)

# ---------------------------------------------------------
# 6. Chart Types & Visual Options
# ---------------------------------------------------------
with st.sidebar.expander("📊 Chart Types & Visual Options", expanded=True):
    selected_charts = st.multiselect(
        "Select Chart Types:", 
        options=["Line Graph", "Bar Plot", "Histogram", "Boxplot", "Scatter Plot", "Heatmap"],
        default=["Line Graph"]
    )
    show_labels = st.checkbox("🏷️ Show Data Values/Labels", value=True)
    show_ci = st.checkbox("📊 Show 95% Confidence Interval", value=True)
    custom_title = st.text_input("Graph Title:")
    custom_xlabel = st.text_input("X-axis Label:")
    custom_ylabel = st.text_input("Y-axis Label:")

# ---------------------------------------------------------
# 7. Forecasting Models Selection
# ---------------------------------------------------------
with st.sidebar.expander("🔮 Forecasting Models Selection", expanded=True):
    model_arima = st.checkbox("📉 ARIMA Forecasting Model", value=False)
    model_sarima = st.checkbox("🔮 SARIMA Forecasting Model", value=True)
    model_linear = st.checkbox("📈 Linear Regression Forecasting Model", value=True)

    forecast_steps = st.number_input(
        "Forecast Horizon (Steps):", 
        min_value=1, 
        max_value=24, 
        value=12 if period_type == 'Monthly' else 3
    )

# ---------------------------------------------------------
# Main Layout Structure using st.tabs
# ---------------------------------------------------------
tab_dash, tab_forecast, tab_insights, tab_data = st.tabs([
    "📊 Dashboard & KPIs", 
    "🤖 AI Forecasting & Models", 
    "🧠 Intelligent Insights", 
    "📄 Data Preview & Export"
])

# --- TAB 1: DASHBOARD & KPIS ---
with tab_dash:
    st.header(f"1. 📊 Dashboard, KPIs & Trade Overview ({analysis_mode})")
    
    if filtered_df.empty:
        st.warning("⚠️ ရွေးချယ်ထားသော Filter သို့မဟုတ် Threshold အတွင်း ဒေတာမရှိပါ။")
    else:
        latest_rec = filtered_df.iloc[-1]
        prev_rec = filtered_df.iloc[-2] if len(filtered_df) > 1 else latest_rec
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        val1 = latest_rec[selected_numerics[0]]
        prev1 = prev_rec[selected_numerics[0]]
        diff1 = ((val1 - prev1) / prev1 * 100) if prev1 != 0 else 0
        val_unit = "%" if analysis_mode == "Percentage Change (%)" else ""
        
        with col_m1:
            st.metric(f"Latest {selected_numerics[0]}", f"{val1:,.2f}{val_unit}", f"{diff1:+.2f}%")
            
        if len(selected_numerics) >= 2:
            val2 = latest_rec[selected_numerics[1]]
            prev2 = prev_rec[selected_numerics[1]]
            diff2 = ((val2 - prev2) / prev2 * 100) if prev2 != 0 else 0
            with col_m2:
                st.metric(f"Latest {selected_numerics[1]}", f"{val2:,.2f}{val_unit}", f"{diff2:+.2f}%")
        
        if 'total_trade' in filtered_df.columns and analysis_mode == "Raw Values":
            tot_val = latest_rec['total_trade']
            prev_tot = prev_rec['total_trade']
            tot_diff = ((tot_val - prev_tot) / prev_tot * 100) if prev_tot != 0 else 0
            with col_m3:
                st.metric("Total Trade", f"${tot_val:,.2f}M", f"{tot_diff:+.2f}%")
                
        if 'trade_balance' in filtered_df.columns and analysis_mode == "Raw Values":
            bal_val = latest_rec['trade_balance']
            bal_status = "Surplus 🟢" if bal_val >= 0 else "Deficit 🔴"
            with col_m4:
                st.metric("Trade Balance", f"${bal_val:,.2f}M", bal_status)

        st.markdown("<br>", unsafe_allow_html=True)

        for chart_type in selected_charts:
            if chart_type == "Line Graph":
                cols_to_plot = selected_numerics.copy()
                fig_line = go.Figure()
                x_vals = filtered_df[x_col].astype(str).tolist()

                for col in cols_to_plot:
                    y_vals = filtered_df[col].astype(float).tolist()
                    fig_line.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers+text' if show_labels else 'lines+markers',
                        text=[f"{v:,.1f}" for v in y_vals] if show_labels else None,
                        textposition="top center",
                        name=col,
                        line=dict(width=3)
                    ))

                fig_line = apply_advanced_layout(
                    fig_line, 
                    custom_title if custom_title else f"Trend Analysis [{analysis_mode}] (X: {x_col})",
                    custom_xlabel if custom_xlabel else x_col,
                    custom_ylabel if custom_ylabel else "Values"
                )
                st.plotly_chart(fig_line, use_container_width=True)

            elif chart_type == "Bar Plot":
                cols_to_plot = selected_numerics.copy()
                fig_bar = go.Figure()
                x_vals = filtered_df[x_col].astype(str).tolist()

                for col in cols_to_plot:
                    y_vals = filtered_df[col].astype(float).tolist()
                    fig_bar.add_trace(go.Bar(
                        x=x_vals,
                        y=y_vals,
                        text=[f"{v:,.1f}" for v in y_vals] if show_labels else None,
                        textposition="auto",
                        name=col
                    ))

                fig_bar.update_layout(barmode='group')
                fig_bar = apply_advanced_layout(
                    fig_bar,
                    custom_title if custom_title else f"Bar Plot Comparison [{analysis_mode}]",
                    custom_xlabel if custom_xlabel else x_col,
                    custom_ylabel if custom_ylabel else "Values"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            elif chart_type == "Histogram":
                for col in selected_numerics:
                    fig_hist = px.histogram(
                        filtered_df, x=col, marginal="rug",
                        text_auto=True if show_labels else False
                    )
                    fig_hist = apply_advanced_layout(fig_hist, f"Histogram Distribution: {col}", col, "Frequency")
                    st.plotly_chart(fig_hist, use_container_width=True)

            elif chart_type == "Boxplot":
                fig_box = px.box(filtered_df, y=selected_numerics, notched=True)
                fig_box = apply_advanced_layout(fig_box, f"Boxplot Statistical Distribution [{analysis_mode}]", "Metrics", "Values")
                st.plotly_chart(fig_box, use_container_width=True)

            elif chart_type == "Scatter Plot":
                if len(selected_numerics) >= 2:
                    fig_scatter = px.scatter(
                        filtered_df, x=selected_numerics[0], y=selected_numerics[1],
                        trendline="ols",
                        text=filtered_df[x_col].astype(str) if show_labels else None
                    )
                    if show_labels:
                        fig_scatter.update_traces(textposition="top center")
                    fig_scatter = apply_advanced_layout(fig_scatter, f"Scatter Plot: {selected_numerics[0]} vs {selected_numerics[1]}", selected_numerics[0], selected_numerics[1])
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("⚠️ Scatter Plot အတွက် Numeric Columns အနည်းဆုံး ၂ ခု ရွေးပေးပါ။")

            elif chart_type == "Heatmap":
                if len(selected_numerics) >= 2:
                    corr_df = filtered_df[selected_numerics].corr()
                    fig_heat = px.imshow(corr_df, text_auto=True if show_labels else False, color_continuous_scale='RdBu_r')
                    fig_heat = apply_advanced_layout(fig_heat, "Correlation Heatmap", "Variables", "Variables")
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("⚠️ Heatmap အတွက် Numeric Columns အနည်းဆုံး ၂ ခု ရွေးပေးပါ။")

# --- TAB 2: AI FORECASTING & MODELS ---
with tab_forecast:
    st.header(f"2. 🤖 Multi-Variable {period_type} Analysis & Forecast (With CI Ranges)")

    if filtered_df.empty:
        st.warning("⚠️ ခန့်မှန်းချက်တွက်ချက်ရန် ဒေတာမရှိပါ။")
    else:
        fig = go.Figure()
        x_labels_hist = filtered_df[x_col].astype(str).tolist()
        last_period_val = x_labels_hist[-1] if len(x_labels_hist) > 0 else "0"
        future_x_labels = generate_future_labels(last_period_val, forecast_steps, period_type)

        evaluation_metrics = []
        palette = px.colors.qualitative.Plotly

        for idx, col_name in enumerate(selected_numerics):
            y_values = filtered_df[col_name].astype(float).tolist()
            base_color = palette[idx % len(palette)]
            
            # Historical Trace
            fig.add_trace(go.Scatter(
                x=x_labels_hist, y=y_values,
                mode='lines+markers+text' if show_labels else 'lines+markers',
                text=[f"{v:,.1f}" for v in y_values] if show_labels else None,
                textposition="top center",
                name=f"Historical ({col_name})",
                line=dict(width=3, color=base_color)
            ))
            
            # 1. ARIMA Model
            if model_arima and len(y_values) >= 5:
                try:
                    ts_data = pd.Series(y_values)
                    arima_res = ARIMA(ts_data, order=(1, 1, 1)).fit()
                    
                    fitted_vals = arima_res.fittedvalues
                    evaluation_metrics.append({
                        "Model": f"ARIMA (1,1,1) - {col_name}",
                        "MAE": mean_absolute_error(y_values, fitted_vals),
                        "RMSE": np.sqrt(mean_squared_error(y_values, fitted_vals))
                    })

                    forecast_obj = arima_res.get_forecast(steps=forecast_steps)
                    forecast_df = forecast_obj.summary_frame(alpha=0.05)
                    pred_y = forecast_df['mean'].tolist()

                    fig.add_trace(go.Scatter(
                        x=future_x_labels, y=pred_y,
                        mode='lines+markers+text' if show_labels else 'lines+markers',
                        text=[f"{v:,.1f}" for v in pred_y] if show_labels else None,
                        textposition="top center",
                        name=f"ARIMA Forecast ({col_name})",
                        line=dict(dash='dash', width=3, color=base_color)
                    ))

                    if show_ci:
                        fig.add_trace(go.Scatter(
                            x=future_x_labels + future_x_labels[::-1],
                            y=forecast_df['mean_ci_upper'].tolist() + forecast_df['mean_ci_lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(37, 99, 235, 0.12)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f"ARIMA 95% CI ({col_name})",
                            showlegend=True
                        ))
                except Exception as e:
                    st.error(f"ARIMA Error [{col_name}]: {e}")

            # 2. SARIMA Model + 95% Confidence Interval
            if model_sarima and len(y_values) >= 6:
                try:
                    ts_data = pd.Series(y_values)
                    s_order = (1, 1, 1, 12) if (period_type == 'Monthly' and len(y_values) >= 12) else (1, 1, 1, 2)
                    sarima_model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=s_order)
                    sarima_res = sarima_model.fit(disp=False)
                    
                    fitted_vals = sarima_res.fittedvalues
                    evaluation_metrics.append({
                        "Model": f"SARIMA - {col_name}",
                        "MAE": mean_absolute_error(y_values, fitted_vals),
                        "RMSE": np.sqrt(mean_squared_error(y_values, fitted_vals))
                    })

                    forecast_obj = sarima_res.get_forecast(steps=forecast_steps)
                    forecast_df = forecast_obj.summary_frame(alpha=0.05)
                    pred_y_sarima = forecast_df['mean'].tolist()

                    fig.add_trace(go.Scatter(
                        x=future_x_labels, y=pred_y_sarima,
                        mode='lines+markers+text' if show_labels else 'lines+markers',
                        text=[f"{v:,.1f}" for v in pred_y_sarima] if show_labels else None,
                        textposition="top center",
                        name=f"SARIMA Forecast ({col_name})",
                        line=dict(dash='dot', width=3, color=base_color)
                    ))

                    if show_ci:
                        fig.add_trace(go.Scatter(
                            x=future_x_labels + future_x_labels[::-1],
                            y=forecast_df['mean_ci_upper'].tolist() + forecast_df['mean_ci_lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(99, 102, 241, 0.15)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f"SARIMA 95% CI ({col_name})",
                            showlegend=True
                        ))
                except Exception as e:
                    st.error(f"SARIMA Error [{col_name}]: {e}")

            # 3. Linear Regression Model + 95% CI & Min/Max Range Labels
            if model_linear and len(y_values) >= 2:
                try:
                    X_train = np.arange(len(y_values)).reshape(-1, 1)
                    y_train = np.array(y_values)
                    reg = LinearRegression().fit(X_train, y_train)
                    
                    pred_train = reg.predict(X_train)
                    residuals = y_train - pred_train
                    std_err = np.std(residuals) * 1.96 if len(residuals) > 1 else np.mean(y_train) * 0.05

                    evaluation_metrics.append({
                        "Model": f"Linear Regression - {col_name}",
                        "MAE": mean_absolute_error(y_train, pred_train),
                        "RMSE": np.sqrt(mean_squared_error(y_train, pred_train))
                    })

                    X_future = np.arange(len(y_values), len(y_values) + forecast_steps).reshape(-1, 1)
                    pred_y_linear = reg.predict(X_future)
                    lin_ci_lower = pred_y_linear - std_err
                    lin_ci_upper = pred_y_linear + std_err

                    lin_texts = [
                        f"{val:,.1f}<br>[Min: {mn:,.1f} | Max: {mx:,.1f}]"
                        for val, mn, mx in zip(pred_y_linear, lin_ci_lower, lin_ci_upper)
                    ] if show_labels else None

                    fig.add_trace(go.Scatter(
                        x=future_x_labels, y=pred_y_linear,
                        mode='lines+markers+text' if show_labels else 'lines+markers',
                        text=lin_texts,
                        textposition="bottom center",
                        name=f"Linear Forecast ({col_name})",
                        line=dict(dash='dashdot', width=3, color='darkorange')
                    ))

                    if show_ci:
                        fig.add_trace(go.Scatter(
                            x=future_x_labels + future_x_labels[::-1],
                            y=lin_ci_upper.tolist() + lin_ci_lower.tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(144, 238, 144, 0.30)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f"Linear 95% CI ({col_name})",
                            showlegend=True
                        ))
                except Exception as e:
                    st.error(f"Linear Error [{col_name}]: {e}")

        fig = apply_advanced_layout(
            fig,
            custom_title if custom_title else f"Multi-Variable {period_type} Analysis & Forecast (With CI Ranges)",
            custom_xlabel if custom_xlabel else f"Period ({period_type})",
            custom_ylabel if custom_ylabel else "Values"
        )
        st.plotly_chart(fig, use_container_width=True)

        if evaluation_metrics:
            st.subheader("📌 Model Performance Evaluation Metrics (MAE & RMSE)")
            eval_df = pd.DataFrame(evaluation_metrics)
            st.dataframe(eval_df.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}"}), use_container_width=True)

# --- TAB 3: INTELLIGENT INSIGHTS ---
with tab_insights:
    st.header(f"3. 🧠 Intelligent Insights & Summary [{analysis_mode}]")
    if not filtered_df.empty and len(selected_numerics) > 0:
        primary_col = selected_numerics[0]
        st.success(f"""
        ### 📋 Statistical Summary
        * **Primary Metric Analyzed:** `{primary_col}`
        * **Calculation Mode:** `{analysis_mode}`
        * **Average (Mean):** `{filtered_df[primary_col].mean():,.2f}`
        * **Peak Value (Maximum):** `{filtered_df[primary_col].max():,.2f}`
        * **Lowest Value (Minimum):** `{filtered_df[primary_col].min():,.2f}`
        """)
        if len(selected_numerics) >= 2:
            st.subheader("📊 Correlation Analysis Matrix")
            st.dataframe(filtered_df[selected_numerics].corr().style.background_gradient(cmap='coolwarm').format("{:.2f}"), use_container_width=True)

# --- TAB 4: DATA PREVIEW & EXPORT ---
with tab_data:
    st.header("4. 📄 Data Preview (Raw vs Clean) & Styled Excel Export")
    
    # Toggle between Raw Data and Clean Processed Data (Matches Jupyter ToggleButton)
    show_raw_mode = st.toggle("🔄 Show Raw Original Dataset (Unfiltered)", value=False)
    
    if show_raw_mode:
        st.subheader("📋 Raw Dataset Preview")
        st.dataframe(df_raw_orig, use_container_width=True)
        export_df = df_raw_orig.copy()
    else:
        st.subheader("✨ Clean & Filtered Chart Summary Data")
        st.dataframe(filtered_df, use_container_width=True)
        export_df = filtered_df.copy()

    col1, col2 = st.columns(2)
    
    # Styled Excel Export using xlsxwriter (Matches Jupyter export_to_excel styling)
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, sheet_name='Chart_Summary_Data', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Chart_Summary_Data']
        
        header_format = workbook.add_format({
            'bold': True, 
            'fg_color': '#DCE6F1', 
            'border': 1
        })
        for col_num, value in enumerate(export_df.columns.values):
            worksheet.write(0, col_num, str(value), header_format)
            worksheet.set_column(col_num, col_num, 18)
            
    excel_buffer.seek(0)

    with col1:
        st.download_button(
            label="📥 Export Chart Data (Styled Excel)",
            data=excel_buffer,
            file_name="chart_summary_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col2:
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Chart Data (CSV)",
            data=csv_data,
            file_name="chart_summary_data.csv",
            mime="text/csv"
        )