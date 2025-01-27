import os
from typing import Any, Optional, Tuple

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import BaseEstimator
from streamlit_extras.grid import grid
from streamlit_extras.metric_cards import style_metric_cards

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/yourrepo",
        "Report a bug": "https://github.com/yourusername/yourrepo/issues",
        "About": "# California Housing Price Predictor v1.0",
    },
)


# Styling
st.markdown(
    """
<style>
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
    }
    div[data-testid="stToolbar"] {
        visibility: hidden;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(ttl="1h", show_spinner="Loading model...")
def load_model() -> Optional[BaseEstimator]:
    """Load the trained model from disk."""
    model_path = os.path.join("models", "latest", "model.pkl")
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def create_prediction_columns() -> Tuple[Any, Any]:
    """Create and return column layouts for prediction inputs."""
    col1, col2 = st.columns(2)
    return col1, col2


def get_user_inputs(col1: Any, col2: Any) -> pd.DataFrame:
    """Collect and validate user inputs for prediction."""
    with col1:
        median_income = st.number_input(
            "Median Income ($10,000s)",
            min_value=0.0,
            max_value=20.0,
            value=8.3252,
            format="%.4f",
            help="Median income in block group",
        )
        house_age = st.slider(
            "House Age", min_value=0, max_value=100, value=41
        )
        ave_rooms = st.number_input("Average Rooms", value=6.984127, step=0.1)
        ave_bedrooms = st.number_input(
            "Average Bedrooms", value=1.023810, step=0.1
        )

    with col2:
        population = st.number_input("Population", value=322.0, step=1.0)
        ave_occup = st.number_input(
            "Average Occupancy", value=2.555556, step=0.1
        )
        latitude = st.slider(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=37.88,
        )
        longitude = st.slider(
            "Longitude",
            min_value=-124.0,
            max_value=-114.0,
            value=-122.23,
        )

    # Create features DataFrame
    features = [
        [
            median_income,
            house_age,
            ave_rooms,
            ave_bedrooms,
            population,
            ave_occup,
            latitude,
            longitude,
        ]
    ]
    columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    return pd.DataFrame(features, columns=columns)


def main() -> None:
    """Main function for the Streamlit application."""
    # Sidebar
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/"
            "Flag_of_California.svg/320px-Flag_of_California.svg.png"
        )
        st.divider()
        st.markdown("### Model Information")
        st.caption("Using Random Forest Regressor")

    # Main content
    st.title("🏠 California Housing Price Predictor")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Visualization", "ℹ️ Info"])

    with tab1:
        prediction_grid = grid(2, vertical_align="middle")

        with st.form("prediction_form", clear_on_submit=False, border=True):
            col1, col2 = create_prediction_columns()
            df = get_user_inputs(col1, col2)

            submitted = st.form_submit_button(
                "Predict Price",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            with st.status(
                "Processing prediction...", expanded=True
            ) as status:
                model = load_model()
                if model:
                    prediction = model.predict(df)
                    status.update(
                        label="Prediction complete!", state="complete"
                    )

                    metric_cols = st.columns(3)
                    predicted_price = prediction[0] * 100000
                    formatted_price = f"{predicted_price:,.2f}"
                    with metric_cols[1]:
                        st.metric(
                            "Predicted House Price",
                            "$" + formatted_price,
                            delta="USD",
                        )
                    style_metric_cards()

    with tab2:
        if "df" in locals():
            fig = px.scatter_mapbox(
                df,
                lat="Latitude",
                lon="Longitude",
                size=[1],
                zoom=6,
                mapbox_style="open-street-map",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        with st.expander("Feature Descriptions", expanded=True):
            feature_desc = pd.DataFrame(
                {
                    "Feature": [
                        "MedInc",
                        "HouseAge",
                        "AveRooms",
                        "AveBedrms",
                        "Population",
                        "AveOccup",
                        "Latitude",
                        "Longitude",
                    ],
                    "Description": [
                        "Median income in block group",
                        "Median house age in block group",
                        "Average number of rooms",
                        "Average number of bedrooms",
                        "Block group population",
                        "Average occupancy",
                        "Block group latitude",
                        "Block group longitude",
                    ],
                }
            )
            st.data_editor(feature_desc, hide_index=True, disabled=True)


if __name__ == "__main__":
    main()
