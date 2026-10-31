
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# =========================================================
# AI DISASTER RISK PREDICTION SYSTEM
# Historical learning: 2000-2025
# Forecasting: 2026-2030
# SHAP: Four risk targets
# =========================================================

st.set_page_config(
    page_title="AI Disaster Risk Assessment",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# CONFIGURATION
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "worldriskindex_cleaned.csv",
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

HISTORY_START_YEAR = 2000
HISTORY_END_YEAR = 2025

FORECAST_START_YEAR = 2026
FORECAST_END_YEAR = 2030

MODEL_FILES = {
    "WRI": "wri_xgboost.pkl",
    "Exposure": "exposure_xgboost.pkl",
    "Vulnerability": "vulnerability_xgboost.pkl",
    "Susceptibility": "susceptibility_xgboost.pkl",
}

RISK_LABELS = {
    0: "Very Low",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High",
}

RISK_ICONS = {
    "WRI": "🌍",
    "Exposure": "🌊",
    "Vulnerability": "🛡️",
    "Susceptibility": "⚠️",
}

# -------------------------
# SIDEBAR AND THEME
# -------------------------

with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    dark_mode = st.toggle("Dark theme", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b1020;
            color: #eef2ff;
        }
        [data-testid="stSidebar"] {
            background-color: #101827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def normalize(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def risk_label(value):
    try:
        return RISK_LABELS.get(int(value), str(value))
    except (ValueError, TypeError):
        return str(value)


def find_column(names, columns):
    lookup = {normalize(c): c for c in columns}

    for name in names:
        if normalize(name) in lookup:
            return lookup[normalize(name)]

    return None


def match_feature(feature, columns):
    aliases = {
        "coping": ["Coping", "C", "c"],
        "adaptive": ["Adaptive", "A", "a"],
    }

    candidates = [feature]
    candidates.extend(aliases.get(normalize(feature), []))

    lookup = {normalize(c): c for c in columns}

    for candidate in candidates:
        if normalize(candidate) in lookup:
            return lookup[normalize(candidate)]

    return None


def get_model_features(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "get_booster"):
        try:
            names = model.get_booster().feature_names
            if names:
                return list(names)
        except Exception:
            pass

    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            features = get_model_features(step)
            if features:
                return features

    return None


def prepare_model_input(row, model):
    features = get_model_features(model)

    if not features:
        raise ValueError(
            "Could not determine the model's expected features."
        )

    values = {}

    for feature in features:
        if normalize(feature) == "year":
            values[feature] = float(row["Year"])
            continue

        column = match_feature(feature, row.index)

        if column is None:
            raise ValueError(
                f"Missing model feature: {feature}"
            )

        value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(value):
            raise ValueError(
                f"Missing or non-numeric value for {feature}"
            )

        values[feature] = float(value)

    return pd.DataFrame([values], columns=features)


# -------------------------
# DATA LOADING
# -------------------------

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    country_column = find_column(
        ["Country", "WRI.Country", "country"],
        data.columns,
    )
    year_column = find_column(["Year"], data.columns)

    if country_column is None or year_column is None:
        raise ValueError(
            "Dataset must contain Country and Year columns."
        )

    data = data.rename(
        columns={
            country_column: "Country",
            year_column: "Year",
        }
    )

    data["Country"] = data["Country"].astype(str).str.strip()
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")

    data = data.dropna(subset=["Country", "Year"]).copy()
    data["Year"] = data["Year"].astype(int)

    return data


@st.cache_resource
def load_models():
    loaded = {}
    errors = []

    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            errors.append(f"Missing model file: {filename}")
            continue

        try:
            loaded[name] = joblib.load(path)
        except Exception as exc:
            errors.append(f"{filename}: {exc}")

    return loaded, errors


def predict_one(model, row):
    X = prepare_model_input(row, model)
    prediction = model.predict(X)[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]

    return prediction, probabilities, X


# =========================================================
# COUNTRY-SPECIFIC FORECASTING
# =========================================================

def forecast_feature(country_history, feature, future_year):
    """
    Fit a separate trend using this country's observations
    from 2000 through 2025.
    """

    if normalize(feature) == "year":
        return float(future_year)

    if feature not in country_history.columns:
        return np.nan

    history = country_history[["Year", feature]].copy()

    history["Year"] = pd.to_numeric(
        history["Year"], errors="coerce"
    )
    history[feature] = pd.to_numeric(
        history[feature], errors="coerce"
    )

    history = history[
        history["Year"].between(
            HISTORY_START_YEAR,
            HISTORY_END_YEAR,
        )
    ].dropna(subset=["Year", feature])

    history = (
        history.groupby("Year", as_index=False)[feature]
        .mean()
        .sort_values("Year")
    )

    if history.empty:
        return np.nan

    years = history["Year"].to_numpy(dtype=float)
    values = history[feature].to_numpy(dtype=float)

    if len(values) < 3 or len(np.unique(years)) < 3:
        return float(values[-1])

    trend = LinearRegression()
    trend.fit(years.reshape(-1, 1), values)

    prediction = float(
        trend.predict([[float(future_year)]])[0]
    )

    # Avoid extreme extrapolation.
    low = float(values.min())
    high = float(values.max())
    spread = high - low

    if spread > 0:
        prediction = float(
            np.clip(
                prediction,
                low - 0.2 * spread,
                high + 0.2 * spread,
            )
        )

    return prediction


def build_future_row(country_history, future_year, model):
    history = country_history[
        country_history["Year"].between(
            HISTORY_START_YEAR,
            HISTORY_END_YEAR,
        )
    ].sort_values("Year")

    if history.empty:
        raise ValueError(
            "No historical observations between 2000 and 2025."
        )

    latest_row = history.iloc[-1].copy()
    latest_row["Year"] = future_year

    features = get_model_features(model)

    if not features:
        raise ValueError("Cannot identify model features.")

    for feature in features:
        if normalize(feature) == "year":
            continue

        column = match_feature(feature, history.columns)

        if column is None:
            raise ValueError(
                f"Feature '{feature}' not found in dataset."
            )

        value = forecast_feature(
            history,
            column,
            future_year,
        )

        if pd.isna(value):
            raise ValueError(
                f"Could not forecast feature '{feature}'."
            )

        latest_row[column] = value

    return latest_row


def run_forecast(data, country, models):
    country_history = data[
        data["Country"].astype(str) == str(country)
    ].copy()

    country_history = country_history[
        country_history["Year"].between(
            HISTORY_START_YEAR,
            HISTORY_END_YEAR,
        )
    ].sort_values("Year")

    if country_history.empty:
        raise ValueError(
            f"No {country} observations between 2000 and 2025."
        )

    results = []

    for year in range(
        FORECAST_START_YEAR,
        FORECAST_END_YEAR + 1,
    ):
        result = {"Country": country, "Year": year}
        predicted_classes = []
        errors = []

        for name, model in models.items():
            try:
                future_row = build_future_row(
                    country_history,
                    year,
                    model,
                )

                prediction, probabilities, _ = predict_one(
                    model,
                    future_row,
                )

                prediction = int(prediction)

                result[f"{name} Class"] = prediction
                result[f"{name} Risk"] = risk_label(prediction)
                predicted_classes.append(prediction)

                if probabilities is not None:
                    result[f"{name} Confidence (%)"] = round(
                        float(np.max(probabilities)) * 100,
                        2,
                    )

            except Exception as exc:
                result[f"{name} Class"] = np.nan
                result[f"{name} Risk"] = "Unavailable"
                errors.append(f"{name}: {exc}")

        if (
            len(models) == len(MODEL_FILES)
            and len(predicted_classes) == len(MODEL_FILES)
            and not errors
        ):
            score = float(np.mean(predicted_classes))
            result["Overall Score"] = round(score, 2)
            result["Overall Risk"] = risk_label(round(score))
        else:
            result["Overall Score"] = np.nan
            result["Overall Risk"] = "Unavailable"

        if errors:
            result["Prediction Notes"] = "; ".join(errors)

        results.append(result)

    return pd.DataFrame(results)


# =========================================================
# SHAP ANALYSIS
# =========================================================

def get_shap_values(model, X):
    """
    Calculate SHAP values for tree models.
    Supports common binary and multiclass output shapes.
    """

    import shap

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    raw = explanation.values
    array = np.asarray(raw)

    prediction = model.predict(X)[0]
    classes = list(getattr(model, "classes_", []))

    class_index = (
        classes.index(prediction)
        if prediction in classes
        else 0
    )

    if isinstance(raw, list):
        values = np.asarray(raw[class_index])[0]
        base_values = np.asarray(
            explanation.base_values[class_index]
        ).reshape(-1)

    elif array.ndim == 3:
        # samples x features x classes
        class_index = min(class_index, array.shape[2] - 1)
        values = array[0, :, class_index]

        base = np.asarray(explanation.base_values)
        if base.ndim >= 2:
            base_values = base[0].reshape(-1)
        else:
            base_values = base.reshape(-1)

        if len(base_values) > 1:
            base_values = np.array([base_values[class_index]])

    elif array.ndim == 2:
        values = array[0]
        base_values = np.asarray(
            explanation.base_values
        ).reshape(-1)

    elif array.ndim == 1:
        values = array
        base_values = np.asarray(
            explanation.base_values
        ).reshape(-1)

    else:
        raise ValueError(
            f"Unsupported SHAP output shape: {array.shape}"
        )

    if len(values) != X.shape[1]:
        raise ValueError(
            "SHAP values do not match the model's feature count."
        )

    base_value = float(base_values[0])

    return (
        np.asarray(values, dtype=float),
        base_value,
        prediction,
    )


# =========================================================
# MAIN DASHBOARD
# =========================================================

st.title("🌍 AI Disaster Risk Assessment")

st.write(
    "Explore historical disaster risk, assess risk categories, "
    "explain model predictions, and forecast future risk."
)

try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

models, model_errors = load_models()

for error in model_errors:
    st.warning(error)

if not models:
    st.error(
        "No trained models loaded. Check your models folder."
    )
    st.stop()

# -------------------------
# DATASET OVERVIEW
# -------------------------

st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Countries", df["Country"].nunique())
c2.metric("Records", len(df))
c3.metric("First year", int(df["Year"].min()))
c4.metric("Latest year", int(df["Year"].max()))

# -------------------------
# COUNTRY SELECTION
# -------------------------

st.subheader("🌍 Select Country and Year")

countries = sorted(df["Country"].dropna().unique().tolist())

selected_country = st.selectbox(
    "Select country",
    countries,
)

country_data = df[
    df["Country"] == selected_country
].copy()

history = country_data[
    country_data["Year"].between(
        HISTORY_START_YEAR,
        HISTORY_END_YEAR,
    )
].sort_values("Year")

if history.empty:
    st.error(
        f"No observations for {selected_country} "
        "from 2000 through 2025."
    )
    st.stop()

available_years = sorted(
    country_data["Year"].unique().tolist()
)

selected_year = st.selectbox(
    "Select historical year",
    available_years,
    index=len(available_years) - 1,
)

matches = country_data[
    country_data["Year"] == selected_year
]

if matches.empty:
    st.error("No record for the selected country and year.")
    st.stop()

selected_row = matches.iloc[-1]

st.subheader("📌 Selected Historical Record")

summary_columns = [
    col for col in [
        "Country",
        "ISO3",
        "Year",
        "WRI",
        "Exposure",
        "Vulnerability",
        "Susceptibility",
    ]
    if col in selected_row.index
]

st.dataframe(
    pd.DataFrame([
        {col: selected_row[col] for col in summary_columns}
    ]),
    use_container_width=True,
    hide_index=True,
)

# =========================================================
# RISK ASSESSMENT
# =========================================================

st.subheader("🧠 Risk Assessment")

st.info(
    "Choose a country and year, then run the saved models "
    "to estimate the four risk categories."
)

if st.button(
    "🚀 Run Risk Assessment",
    use_container_width=True,
):
    records = []
    errors = []

    for name, model in models.items():
        try:
            prediction, probabilities, _ = predict_one(
                model,
                selected_row,
            )

            record = {
                "Risk Dimension": name,
                "Predicted Class": int(prediction),
                "Predicted Risk": risk_label(prediction),
            }

            if probabilities is not None:
                record["Confidence (%)"] = round(
                    float(np.max(probabilities)) * 100,
                    2,
                )

            records.append(record)

        except Exception as exc:
            errors.append(f"{name}: {exc}")

    st.session_state["assessment_records"] = records
    st.session_state["assessment_errors"] = errors
    st.session_state["assessment_country"] = selected_country
    st.session_state["assessment_year"] = selected_year

if (
    st.session_state.get("assessment_country") == selected_country
    and st.session_state.get("assessment_year") == selected_year
):
    for error in st.session_state.get("assessment_errors", []):
        st.warning(error)

    records = st.session_state.get("assessment_records", [])

    if records:
        st.markdown("### Predicted Risk Categories")

        cols = st.columns(len(records))

        for col, record in zip(cols, records):
            with col:
                name = record["Risk Dimension"]
                st.metric(
                    f"{RISK_ICONS.get(name, '📊')} {name}",
                    record["Predicted Risk"],
                )

        prediction_df = pd.DataFrame(records)

        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True,
        )

        if len(records) == len(MODEL_FILES):
            score = np.mean([
                record["Predicted Class"]
                for record in records
            ])

            st.metric(
                "Overall Risk Score (0–4)",
                f"{score:.2f}",
                risk_label(round(score)),
            )

        st.download_button(
            "⬇️ Download Risk Assessment CSV",
            prediction_df.to_csv(index=False).encode("utf-8"),
            file_name="risk_assessment.csv",
            mime="text/csv",
        )

# =========================================================
# SHAP ANALYSIS FOR FOUR TARGETS
# =========================================================

st.subheader("🔍 Explainable AI — SHAP Analysis")

st.write(
    "Choose a target to inspect which input features contribute "
    "most strongly to its model output."
)

try:
    import shap
    import matplotlib.pyplot as plt

    available_targets = [
        target for target in [
            "WRI",
            "Exposure",
            "Vulnerability",
            "Susceptibility",
        ]
        if target in models
    ]

    if available_targets:
        shap_target = st.selectbox(
            "Select target to explain",
            available_targets,
            key="shap_target_selector",
        )

        st.caption(
            f"Country: {selected_country} | "
            f"Year: {selected_year} | Target: {shap_target}"
        )

        if st.button(
            "🔎 Explain Prediction with SHAP",
            use_container_width=True,
            key="run_shap",
        ):
            try:
                model = models[shap_target]

                X_shap = prepare_model_input(
                    selected_row,
                    model,
                )

                values, base_value, prediction = get_shap_values(
                    model,
                    X_shap,
                )

                feature_names = list(X_shap.columns)

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Feature Value": X_shap.iloc[0].to_numpy(),
                    "SHAP Value": values,
                    "Absolute SHAP": np.abs(values),
                }).sort_values(
                    "Absolute SHAP",
                    ascending=False,
                )

                # Save results so Streamlit reruns don't erase the chart.
                st.session_state["shap_result"] = {
                    "target": shap_target,
                    "country": selected_country,
                    "year": selected_year,
                    "table": importance_df,
                    "base_value": base_value,
                    "prediction": prediction,
                }

            except Exception as exc:
                st.error(f"SHAP calculation failed: {exc}")

    else:
        st.warning(
            "No trained models are available for SHAP analysis."
        )

    saved_shap = st.session_state.get("shap_result")

    if (
        saved_shap is not None
        and saved_shap["target"] in models
        and saved_shap["country"] == selected_country
        and saved_shap["year"] == selected_year
    ):
        target = saved_shap["target"]
        importance_df = saved_shap["table"]

        # No separate prediction metric/card in this section.
        st.markdown(
            f"### {target} — SHAP Feature Analysis"
        )

        st.caption(
            "Top features are ranked by absolute SHAP impact. "
            "Positive values push the explained model output higher; "
            "negative values push it lower."
        )

        # Top ten features, ordered by SHAP value.
        top_features = (
            importance_df.head(10)
            .sort_values("SHAP Value", ascending=True)
            .copy()
        )

        max_impact = max(
            float(top_features["SHAP Value"].abs().max()),
            0.01,
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=top_features["SHAP Value"],
                y=top_features["Feature"],
                orientation="h",
                marker=dict(
                    color=[
                        "#2878B5" if value >= 0 else "#E07A5F"
                        for value in top_features["SHAP Value"]
                    ]
                ),
                text=[
                    f"{value:.3f}"
                    for value in top_features["SHAP Value"]
                ],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "SHAP value: %{x:.4f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=f"Top 10 Feature Contributions — {target}",
            xaxis_title="SHAP value",
            yaxis_title="Feature",
            height=520,
            margin=dict(l=30, r=70, t=80, b=60),
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=13, color="#263238"),
            xaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="#555555",
                range=[
                    -max_impact * 1.25,
                    max_impact * 1.25,
                ],
                gridcolor="#E8EDF2",
            ),
            yaxis=dict(
                automargin=True,
                gridcolor="rgba(0,0,0,0)",
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"shap_chart_{target}",
        )

        with st.expander("View all SHAP feature values"):
            st.dataframe(
                importance_df[
                    [
                        "Feature",
                        "Feature Value",
                        "SHAP Value",
                        "Absolute SHAP",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "⬇️ Download SHAP Results CSV",
            importance_df.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"{selected_country}_{selected_year}_"
                f"{target}_shap.csv"
            ),
            mime="text/csv",
            key=f"shap_download_{target}",
        )

        # Optional waterfall plot.
        with st.expander("View SHAP Waterfall Explanation"):
            try:
                import shap

                model = models[target]
                X_shap = prepare_model_input(
                    selected_row,
                    model,
                )

                explanation = shap.TreeExplainer(model)(X_shap)
                raw_values = np.asarray(explanation.values)

                classes = list(getattr(model, "classes_", []))
                predicted = model.predict(X_shap)[0]
                class_index = (
                    classes.index(predicted)
                    if predicted in classes else 0
                )

                if raw_values.ndim == 3:
                    class_index = min(
                        class_index,
                        raw_values.shape[2] - 1,
                    )
                    waterfall_values = raw_values[
                        0, :, class_index
                    ]
                    base_array = np.asarray(
                        explanation.base_values
                    )

                    if base_array.ndim >= 2:
                        waterfall_base = float(
                            base_array[0].reshape(-1)[
                                min(
                                    class_index,
                                    base_array.shape[-1] - 1,
                                )
                            ]
                        )
                    else:
                        waterfall_base = float(
                            base_array.reshape(-1)[0]
                        )

                else:
                    waterfall_values = np.asarray(
                        explanation.values
                    ).reshape(-1, X_shap.shape[1])[0]

                    waterfall_base = float(
                        np.asarray(
                            explanation.base_values
                        ).reshape(-1)[0]
                    )

                waterfall_explanation = shap.Explanation(
                    values=waterfall_values,
                    base_values=waterfall_base,
                    data=X_shap.iloc[0].to_numpy(),
                    feature_names=list(X_shap.columns),
                )

                shap.plots.waterfall(
                    waterfall_explanation,
                    max_display=10,
                    show=False,
                )

                st.pyplot(
                    plt.gcf(),
                    use_container_width=True,
                )

                plt.close("all")

            except Exception as exc:
                st.warning(
                    f"Waterfall plot unavailable: {exc}"
                )

except ImportError:
    st.warning(
        "Install SHAP and Matplotlib with: "
        "pip install shap matplotlib"
    )

# =========================================================
# HISTORICAL TRENDS
# =========================================================

st.subheader("📈 Historical Country Trends")

trend_columns = [
    col for col in [
        "WRI",
        "Exposure",
        "Vulnerability",
        "Susceptibility",
    ]
    if col in history.columns
]

if trend_columns:
    selected_metrics = st.multiselect(
        "Choose historical indicators",
        trend_columns,
        default=trend_columns,
    )

    if selected_metrics:
        chart_data = history[
            ["Year"] + selected_metrics
        ].copy()

        chart_data = chart_data.melt(
            id_vars="Year",
            var_name="Indicator",
            value_name="Value",
        )

        chart_data["Value"] = pd.to_numeric(
            chart_data["Value"],
            errors="coerce",
        )

        chart_data = chart_data.dropna(subset=["Value"])

        if not chart_data.empty:
            fig = px.line(
                chart_data,
                x="Year",
                y="Value",
                color="Indicator",
                markers=True,
                title=f"{selected_country}: Historical Trends",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

# =========================================================
# FUTURE DISASTER RISK FORECAST
# =========================================================

st.subheader("🔮 Future Disaster Risk Forecast")

st.info(
    "The system learns feature trends from the selected country's "
    "available observations between 2000 and 2025. It then passes "
    "forecasted feature values to the existing trained models."
)

st.write(
    f"**Country:** {selected_country}  \n"
    f"**Historical learning window:** 2000–2025  \n"
    f"**Forecast window:** 2026–2030  \n"
    f"**Latest available year in the learning window:** "
    f"{int(history['Year'].max())}"
)

if st.button(
    "🔮 Generate Future Risk Forecast",
    use_container_width=True,
):
    with st.spinner("Generating future risk estimates..."):
        try:
            future_results = run_forecast(
                df,
                selected_country,
                models,
            )

            st.session_state["future_results"] = future_results
            st.session_state["future_country"] = selected_country

        except Exception as exc:
            st.error(f"Forecast failed: {exc}")

future_results = st.session_state.get("future_results")

if (
    future_results is not None
    and st.session_state.get("future_country") == selected_country
    and not future_results.empty
):
    st.markdown(
        f"### {selected_country} — Forecast for 2026–2030"
    )

    display_columns = [
        col for col in [
            "Country",
            "Year",
            "WRI Risk",
            "Exposure Risk",
            "Vulnerability Risk",
            "Susceptibility Risk",
            "Overall Score",
            "Overall Risk",
        ]
        if col in future_results.columns
    ]

    st.dataframe(
        future_results[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    valid_results = future_results.dropna(
        subset=["Overall Score"]
    )

    if not valid_results.empty:
        fig = px.line(
            valid_results,
            x="Year",
            y="Overall Score",
            markers=True,
            title="Projected Overall Disaster Risk",
        )

        fig.update_yaxes(
            range=[0, 4],
            tickvals=[0, 1, 2, 3, 4],
            ticktext=[
                "Very Low",
                "Low",
                "Medium",
                "High",
                "Very High",
            ],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # Risk dimensions across future years
    trend_rows = []

    for _, row in future_results.iterrows():
        for dimension in [
            "WRI",
            "Exposure",
            "Vulnerability",
            "Susceptibility",
        ]:
            column = f"{dimension} Class"

            if (
                column in future_results.columns
                and pd.notna(row[column])
            ):
                trend_rows.append({
                    "Year": int(row["Year"]),
                    "Risk Dimension": dimension,
                    "Risk Score": float(row[column]),
                })

    if trend_rows:
        trend_df = pd.DataFrame(trend_rows)

        fig = px.line(
            trend_df,
            x="Year",
            y="Risk Score",
            color="Risk Dimension",
            markers=True,
            title="Projected Risk Dimensions",
        )

        fig.update_yaxes(
            range=[0, 4],
            tickvals=[0, 1, 2, 3, 4],
            ticktext=[
                "Very Low",
                "Low",
                "Medium",
                "High",
                "Very High",
            ],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    confidence_columns = [
        col for col in future_results.columns
        if col.endswith("Confidence (%)")
    ]

    if confidence_columns:
        st.markdown("### Model Confidence")

        st.dataframe(
            future_results[
                ["Year"] + confidence_columns
            ],
            use_container_width=True,
            hide_index=True,
        )

    if "Prediction Notes" in future_results.columns:
        for _, row in future_results.iterrows():
            note = row.get("Prediction Notes")

            if isinstance(note, str) and note:
                st.warning(
                    f"{int(row['Year'])}: {note}"
                )

    st.download_button(
        "⬇️ Download Future Forecast CSV",
        future_results.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"{selected_country}_future_risk_forecast.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

# =========================================================
# MODEL INFORMATION
# =========================================================

st.subheader("🧠 Model Information")

model_info = []

for name, model in models.items():
    features = get_model_features(model)

    model_info.append({
        "Risk Dimension": name,
        "Model Type": type(model).__name__,
        "Expected Features": (
            len(features) if features else "Unknown"
        ),
    })

st.dataframe(
    pd.DataFrame(model_info),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

st.caption(
    "AI Disaster Risk Prediction System | "
    "Country-specific historical trends | "
    "SHAP explainability | Forecasts are estimates, "
    "not guaranteed outcomes."
)
