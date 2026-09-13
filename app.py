
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation Dashboard")
st.write(
    "Upload your transaction dataset and explore the data, build RFM features, "
    "evaluate K-Means, visualize clusters, and download the final segmentation."
)

# =========================================================
# SIDEBAR - UPLOAD
# =========================================================
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("👈 Upload your CSV or Excel dataset from the sidebar to start.")
    st.stop()


# =========================================================
# READ DATA
# =========================================================
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

# Work on a copy
df = df.copy()

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Overview",
    "📊 Visualizations",
    "🧮 RFM Analysis",
    "🤖 K-Means",
    "🗺️ PCA",
    "🔎 Customer Explorer",
    "📥 Export"
])


# =========================================================
# TAB 1 - OVERVIEW
# =========================================================
with tab1:
    st.header("🏠 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", f"{df.isna().sum().sum():,}")
    c4.metric("Duplicates", f"{df.duplicated().sum():,}")

    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("📋 Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing Values": [df[c].isna().sum() for c in df.columns],
        "Unique Values": [df[c].nunique() for c in df.columns]
    })

    st.dataframe(info_df, use_container_width=True)

    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)


# =========================================================
# TAB 2 - DYNAMIC VISUALIZATIONS
# =========================================================
with tab2:
    st.header("📊 Interactive Visualization Builder")

    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    chart_type = st.selectbox(
        "Choose Visualization Type",
        [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Histogram",
            "Box Plot",
            "Violin Plot",
            "Pie Chart",
            "Area Chart",
            "Count Plot",
            "Correlation Heatmap",
            "Scatter Matrix"
        ]
    )

    if chart_type == "Correlation Heatmap":
        if len(numeric_columns) < 2:
            st.warning("You need at least two numerical columns.")
        else:
            corr = df[numeric_columns].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap"
            )

            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Matrix":
        if len(numeric_columns) < 2:
            st.warning("You need at least two numerical columns.")
        else:
            selected = st.multiselect(
                "Select numerical columns",
                numeric_columns,
                default=numeric_columns[:min(4, len(numeric_columns))]
            )

            if len(selected) >= 2:
                fig = px.scatter_matrix(
                    df,
                    dimensions=selected,
                    title="Scatter Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Select at least two columns.")

    elif chart_type == "Histogram":
        if not numeric_columns:
            st.warning("No numerical columns found.")
        else:
            x_col = st.selectbox("Column", numeric_columns)
            bins = st.slider("Number of bins", 5, 100, 30)

            fig = px.histogram(
                df,
                x=x_col,
                nbins=bins,
                title=f"Distribution of {x_col}"
            )

            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        if not numeric_columns:
            st.warning("No numerical columns found.")
        else:
            y_col = st.selectbox("Numerical Column", numeric_columns)
            group_options = ["None"] + categorical_columns
            group_col = st.selectbox("Group By (optional)", group_options)

            if group_col == "None":
                fig = px.box(df, y=y_col, title=f"Box Plot - {y_col}")
            else:
                fig = px.box(
                    df,
                    x=group_col,
                    y=y_col,
                    title=f"{y_col} by {group_col}"
                )

            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Violin Plot":
        if not numeric_columns:
            st.warning("No numerical columns found.")
        else:
            y_col = st.selectbox("Numerical Column", numeric_columns)
            group_options = ["None"] + categorical_columns
            group_col = st.selectbox("Group By (optional)", group_options)

            if group_col == "None":
                fig = px.violin(
                    df,
                    y=y_col,
                    box=True,
                    points="all",
                    title=f"Violin Plot - {y_col}"
                )
            else:
                fig = px.violin(
                    df,
                    x=group_col,
                    y=y_col,
                    box=True,
                    points="all",
                    title=f"{y_col} by {group_col}"
                )

            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart":
        if not categorical_columns:
            st.warning("No categorical columns found.")
        else:
            col = st.selectbox("Category Column", categorical_columns)

            counts = df[col].value_counts().head(20).reset_index()
            counts.columns = [col, "Count"]

            fig = px.pie(
                counts,
                names=col,
                values="Count",
                title=f"Distribution of {col}"
            )

            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Count Plot":
        if not categorical_columns:
            st.warning("No categorical columns found.")
        else:
            col = st.selectbox("Category Column", categorical_columns)

            counts = df[col].value_counts().head(30).reset_index()
            counts.columns = [col, "Count"]

            fig = px.bar(
                counts,
                x=col,
                y="Count",
                title=f"Count Plot - {col}"
            )

            st.plotly_chart(fig, use_container_width=True)

    elif chart_type in ["Bar Chart", "Line Chart", "Area Chart"]:
        x_col = st.selectbox("X Column", columns)

        y_options = numeric_columns
        if not y_options:
            st.warning("No numerical columns found.")
        else:
            y_col = st.selectbox("Y Column", y_options)

            if chart_type == "Bar Chart":
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} by {x_col}"
                )

            elif chart_type == "Line Chart":
                plot_df = df.sort_values(x_col)
                fig = px.line(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over {x_col}"
                )

            else:
                plot_df = df.sort_values(x_col)
                fig = px.area(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} over {x_col}"
                )

            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        if len(numeric_columns) < 2:
            st.warning("You need at least two numerical columns.")
        else:
            x_col = st.selectbox("X Column", numeric_columns)
            y_col = st.selectbox("Y Column", numeric_columns)

            color_options = ["None"] + categorical_columns + numeric_columns
            color_col = st.selectbox("Color By (optional)", color_options)

            if color_col == "None":
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
            else:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{y_col} vs {x_col}"
                )

            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# RFM FUNCTION
# =========================================================
def build_rfm(data, customer_col, date_col, amount_col):
    work = data[[customer_col, date_col, amount_col]].copy()

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce")

    work = work.dropna(subset=[customer_col, date_col, amount_col])

    if work.empty:
        raise ValueError("No valid rows remain after date/amount conversion.")

    reference_date = work[date_col].max()

    rfm_result = (
        work.groupby(customer_col)
        .agg(
            Recency=(
                date_col,
                lambda x: (reference_date - x.max()).days
            ),
            Frequency=(date_col, "count"),
            Monetary=(amount_col, "sum")
        )
        .reset_index()
    )

    return rfm_result, reference_date


# =========================================================
# COLUMN SELECTION
# =========================================================
st.sidebar.header("🧮 RFM Column Settings")

all_columns = df.columns.tolist()

# Try to detect the project's original columns
def find_column(possible_names):
    lower_map = {str(c).lower(): c for c in all_columns}

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for c in all_columns:
        cl = str(c).lower()
        for name in possible_names:
            if name.lower() in cl:
                return c

    return all_columns[0] if all_columns else None


default_customer = find_column(
    ["customer_id", "customer id", "customer"]
)

default_date = find_column(
    ["transaction_date", "transaction date", "date"]
)

default_amount = find_column(
    ["amount", "sales", "revenue", "total_amount", "price"]
)

customer_index = all_columns.index(default_customer) if default_customer in all_columns else 0
date_index = all_columns.index(default_date) if default_date in all_columns else 0
amount_index = all_columns.index(default_amount) if default_amount in all_columns else 0

customer_col = st.sidebar.selectbox(
    "Customer ID Column",
    all_columns,
    index=customer_index
)

date_col = st.sidebar.selectbox(
    "Transaction Date Column",
    all_columns,
    index=date_index
)

amount_col = st.sidebar.selectbox(
    "Amount Column",
    all_columns,
    index=amount_index
)


# =========================================================
# TAB 3 - RFM
# =========================================================
with tab3:
    st.header("🧮 RFM Analysis")

    st.write(
        "RFM is calculated from the uploaded transaction data: "
        "Recency = days since the latest purchase, "
        "Frequency = number of transactions, "
        "Monetary = total spending."
    )

    try:
        rfm, reference_date = build_rfm(
            df,
            customer_col,
            date_col,
            amount_col
        )

        st.success(
            f"Reference Date: {reference_date.date()} | "
            f"Customers: {rfm.shape[0]:,}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Customers", f"{rfm.shape[0]:,}")
        c2.metric("Average Frequency", f"{rfm['Frequency'].mean():.2f}")
        c3.metric("Average Monetary", f"{rfm['Monetary'].mean():.2f}")

        st.subheader("RFM Feature Table")
        st.dataframe(rfm.head(20), use_container_width=True)

        st.subheader("RFM Distributions")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig = px.histogram(
                rfm,
                x="Recency",
                title="Recency Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                rfm,
                x="Frequency",
                title="Frequency Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = px.histogram(
                rfm,
                x="Monetary",
                title="Monetary Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"RFM Error: {e}")
        st.info(
            "Make sure the selected Customer ID, Date, and Amount columns "
            "match your transaction dataset."
        )


# =========================================================
# K-MEANS HELPER
# =========================================================
def run_segmentation(rfm_data, chosen_k):
    features = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_data[features])

    kmeans = KMeans(
        n_clusters=chosen_k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    result = rfm_data.copy()
    result["cluster"] = labels

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    result["PCA1"] = X_pca[:, 0]
    result["PCA2"] = X_pca[:, 1]

    return result, scaler, kmeans, pca, X_scaled


# =========================================================
# TAB 4 - K-MEANS
# =========================================================
with tab4:
    st.header("🤖 K-Means Customer Segmentation")

    try:
        rfm, reference_date = build_rfm(
            df,
            customer_col,
            date_col,
            amount_col
        )

        features = ["Recency", "Frequency", "Monetary"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(rfm[features])

        st.subheader("1️⃣ Feature Scaling")
        scaled_preview = pd.DataFrame(
            X_scaled,
            columns=features
        )

        st.dataframe(scaled_preview.head(10), use_container_width=True)

        # K evaluation
        st.subheader("2️⃣ Choosing K")

        k_values = list(range(2, 11))
        inertias = []
        silhouette_scores = []

        for k in k_values:
            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(X_scaled)

            inertias.append(model.inertia_)
            silhouette_scores.append(
                silhouette_score(X_scaled, labels)
            )

        k_results = pd.DataFrame({
            "k": k_values,
            "inertia": inertias,
            "silhouette_score": silhouette_scores
        })

        st.dataframe(k_results, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                k_results,
                x="k",
                y="inertia",
                markers=True,
                title="Elbow Method: Inertia vs K"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                k_results,
                x="k",
                y="silhouette_score",
                markers=True,
                title="Silhouette Score vs K"
            )
            st.plotly_chart(fig, use_container_width=True)

        best_silhouette_k = int(
            k_results.loc[
                k_results["silhouette_score"].idxmax(),
                "k"
            ]
        )

        st.info(
            f"Highest silhouette score occurs at k={best_silhouette_k}. "
            "For this project, k=4 is used to provide a more actionable "
            "business segmentation, matching the original project logic."
        )

        chosen_k = st.slider(
            "Choose number of clusters",
            min_value=2,
            max_value=10,
            value=4
        )

        # Final model
        segmented_rfm, scaler, kmeans, pca, X_scaled = run_segmentation(
            rfm,
            chosen_k
        )

        st.subheader(f"3️⃣ K-Means Result — k={chosen_k}")

        cluster_sizes = (
            segmented_rfm["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("cluster")
            .reset_index(name="customers")
        )

        fig = px.bar(
            cluster_sizes,
            x="cluster",
            y="customers",
            title="Customers per Cluster"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            segmented_rfm.head(20),
            use_container_width=True
        )

        # Cluster summary
        st.subheader("4️⃣ Cluster RFM Profile")

        summary = (
            segmented_rfm
            .groupby("cluster")[features]
            .mean()
            .round(2)
        )

        summary["Customers"] = (
            segmented_rfm["cluster"]
            .value_counts()
            .sort_index()
        )

        summary["Customer %"] = (
            summary["Customers"] /
            len(segmented_rfm) * 100
        ).round(1)

        st.dataframe(summary, use_container_width=True)

        # Persona assignment based on actual RFM
        ranked = summary.copy()

        ranked["activity_score"] = (
            ranked["Frequency"].rank(
                ascending=False,
                method="first"
            )
            + ranked["Monetary"].rank(
                ascending=False,
                method="first"
            )
            + ranked["Recency"].rank(
                ascending=True,
                method="first"
            )
        )

        vip_cluster = ranked["activity_score"].idxmin()

        remaining = ranked.drop(index=vip_cluster)

        at_risk_cluster = remaining["Recency"].idxmax()

        remaining = remaining.drop(index=at_risk_cluster)

        loyal_cluster = None

        if len(remaining) >= 2:
            loyal_cluster = (
                remaining["Frequency"].rank(
                    ascending=False,
                    method="first"
                )
                + remaining["Recency"].rank(
                    ascending=True,
                    method="first"
                )
            ).idxmin()

        persona_map = {
            vip_cluster: "VIP / High-Value Customers",
            at_risk_cluster: "At-Risk Customers"
        }

        if loyal_cluster is not None:
            persona_map[loyal_cluster] = "Loyal Active Customers"

        for cluster_id in summary.index:
            if cluster_id not in persona_map:
                persona_map[cluster_id] = (
                    "Occasional / Developing Customers"
                )

        final_profile = summary.copy()
        final_profile["Persona"] = final_profile.index.map(
            persona_map
        )

        final_profile = final_profile[
            [
                "Persona",
                "Customers",
                "Customer %",
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]

        st.subheader("5️⃣ Customer Personas")
        st.dataframe(final_profile, use_container_width=True)

        st.session_state["segmented_rfm"] = segmented_rfm
        st.session_state["final_profile"] = final_profile

    except Exception as e:
        st.error(f"K-Means Error: {e}")


# =========================================================
# TAB 5 - PCA
# =========================================================
with tab5:
    st.header("🗺️ PCA Cluster Visualization")

    if "segmented_rfm" not in st.session_state:
        st.warning("Run the K-Means tab first.")
    else:
        segmented_rfm = st.session_state["segmented_rfm"]

        fig = px.scatter(
            segmented_rfm,
            x="PCA1",
            y="PCA2",
            color="cluster",
            hover_data=[
                customer_col if customer_col in segmented_rfm.columns
                else segmented_rfm.columns[0],
                "Recency",
                "Frequency",
                "Monetary"
            ],
            title="Customer Segments — K-Means PCA Visualization"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(
            "PCA reduces the three scaled RFM features into two dimensions "
            "so the customer clusters can be visualized."
        )


# =========================================================
# TAB 6 - CUSTOMER EXPLORER
# =========================================================
with tab6:
    st.header("🔎 Customer Explorer")

    if "segmented_rfm" not in st.session_state:
        st.warning("Run the K-Means tab first.")
    else:
        segmented_rfm = st.session_state["segmented_rfm"]

        ids = segmented_rfm[customer_col].astype(str).tolist()

        selected_customer = st.selectbox(
            "Select Customer",
            ids
        )

        customer_row = segmented_rfm[
            segmented_rfm[customer_col].astype(str)
            == selected_customer
        ]

        if not customer_row.empty:
            row = customer_row.iloc[0]

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Recency", f"{row['Recency']:.0f} days")
            c2.metric("Frequency", f"{row['Frequency']:.0f}")
            c3.metric("Monetary", f"{row['Monetary']:.2f}")
            c4.metric("Cluster", str(row["cluster"]))

            st.dataframe(
                customer_row,
                use_container_width=True
            )


# =========================================================
# TAB 7 - EXPORT
# =========================================================
with tab7:
    st.header("📥 Export Results")

    if "segmented_rfm" not in st.session_state:
        st.warning("Run the K-Means tab first.")
    else:
        segmented_rfm = st.session_state["segmented_rfm"]

        csv_data = segmented_rfm.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Segmented Customers CSV",
            data=csv_data,
            file_name="customer_segmentation_results.csv",
            mime="text/csv"
        )

        if "final_profile" in st.session_state:
            profile_csv = (
                st.session_state["final_profile"]
                .to_csv(index=True)
                .encode("utf-8")
            )

            st.download_button(
                label="📥 Download Customer Personas CSV",
                data=profile_csv,
                file_name="customer_personas.csv",
                mime="text/csv"
            )

        st.success(
            "Your uploaded dataset is processed in the Streamlit session. "
            "The app does not require the original dataset to be saved inside the project."
        )
