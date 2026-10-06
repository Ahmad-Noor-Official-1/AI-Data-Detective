import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="AI Data Detective",
    page_icon="🕵️",
    layout="wide"
)

# =========================================================
# PROFESSIONAL AI / SAAS UI THEME
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(59, 130, 246, 0.10), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(139, 92, 246, 0.09), transparent 25%),
            #080d18;
        color: #e8eefc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(96, 165, 250, 0.20);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(20, 31, 55, 0.98), rgba(13, 20, 38, 0.98));
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -90px;
        top: -120px;
        border-radius: 50%;
        background: rgba(99, 102, 241, 0.16);
        filter: blur(35px);
    }

    .hero .eyebrow {
        position: relative;
        z-index: 1;
        color: #60a5fa;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        color: #f8fbff;
        font-size: 2.65rem;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -0.035em;
    }

    .hero p {
        position: relative;
        z-index: 1;
        max-width: 720px;
        margin: 0.75rem 0 0;
        color: #9eabc3;
        font-size: 1rem;
        line-height: 1.65;
    }

    h2 {
        color: #f1f5ff !important;
        font-weight: 750 !important;
        letter-spacing: -0.02em;
        margin-top: 1.6rem !important;
    }

    h3 {
        color: #dbe7ff !important;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.13) !important;
    }

    div[data-testid="stMetric"] {
        min-height: 118px;
        padding: 1.15rem 1.25rem;
        border: 1px solid rgba(148, 163, 184, 0.13);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(20, 30, 52, 0.96), rgba(14, 22, 39, 0.96));
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.20), inset 0 1px 0 rgba(255, 255, 255, 0.025);
        transition: transform 0.18s ease, border-color 0.18s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.32);
    }

    div[data-testid="stMetricLabel"] {
        color: #8fa0bd !important;
        font-size: 0.78rem !important;
        font-weight: 650 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fbff !important;
        font-weight: 750 !important;
    }

    section[data-testid="stFileUploaderDropzone"] {
        border: 1px dashed rgba(96, 165, 250, 0.40) !important;
        border-radius: 18px !important;
        background: rgba(16, 26, 46, 0.72) !important;
        padding: 1.25rem !important;
    }

    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(96, 165, 250, 0.75) !important;
        background: rgba(20, 34, 58, 0.82) !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border: 1px solid rgba(96, 165, 250, 0.32) !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
        color: white !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.20) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.30) !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
    }

    div[data-testid="stAlert"] {
        border-radius: 14px !important;
        border: 1px solid rgba(148, 163, 184, 0.13) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 12px !important;
        background-color: rgba(16, 25, 43, 0.90) !important;
        border-color: rgba(148, 163, 184, 0.18) !important;
    }

    .stCaption {
        color: #7f8da8 !important;
    }

    .footer-note {
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(148, 163, 184, 0.10);
        color: #66758f;
        text-align: center;
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Automated Data Investigation</div>
        <h1>🕵️ AI Data Detective</h1>
        <p>
            Turn messy CSV and Excel data into clear findings,
            visual insights, and a professional investigation report.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload your dataset",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    # =====================================================
    # READ DATASET
    # =====================================================

    try:

        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

    except Exception as e:

        st.error(f"❌ Could not read the file: {e}")
        st.stop()


    st.success("✅ Dataset loaded successfully!")


    # =====================================================
    # TRY TO CONVERT NUMERIC COLUMNS
    # =====================================================

    for column in df.columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.notna().sum() >= len(df) * 0.7:

            df[column] = converted


    # =====================================================
    # IDENTIFY NUMERICAL COLUMNS
    # =====================================================

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()


    # =====================================================
    # IDENTIFY DATE COLUMNS
    # =====================================================

    date_columns = []

    for column in df.columns:

        if column in numerical_columns:
            continue

        converted = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        valid_dates = converted.notna().sum()

        if valid_dates >= max(2, len(df) * 0.5):

            date_columns.append(column)


    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )

    col4.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )


    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.header("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # =====================================================
    # COLUMN INFORMATION
    # =====================================================

    st.header("🔎 Column Information")

    info = pd.DataFrame({

        "Column": df.columns,

        "Data Type": df.dtypes.astype(str),

        "Missing Values": df.isna().sum().values,

        "Unique Values": df.nunique().values

    })

    st.dataframe(
        info,
        use_container_width=True
    )


    # =====================================================
    # NUMERICAL STATISTICS
    # =====================================================

    st.header("📈 Numerical Statistics")

    if numerical_columns:

        statistics = (
            df[numerical_columns]
            .describe()
            .T
        )

        st.dataframe(
            statistics,
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns were found."
        )


    # =====================================================
    # ANOMALY DETECTION
    # =====================================================

    st.header("🚨 Anomaly Detection")

    if numerical_columns:

        selected_anomaly_column = st.selectbox(

            "Choose a numerical column:",

            numerical_columns,

            key="anomaly_column"

        )


        anomaly_data = df[
            [selected_anomaly_column]
        ].dropna()


        if len(anomaly_data) >= 5:

            model = IsolationForest(

                contamination=0.1,

                random_state=42

            )


            predictions = model.fit_predict(

                anomaly_data[
                    [selected_anomaly_column]
                ]

            )


            anomaly_data = anomaly_data.copy()

            anomaly_data["Result"] = predictions


            anomalies = anomaly_data[
                anomaly_data["Result"] == -1
            ]


            normal = anomaly_data[
                anomaly_data["Result"] == 1
            ]


            col1, col2 = st.columns(2)


            col1.metric(
                "✅ Normal Values",
                len(normal)
            )


            col2.metric(
                "⚠️ Anomalies",
                len(anomalies)
            )


            if len(anomalies) > 0:

                st.warning(

                    f"⚠️ Detected {len(anomalies)} "
                    f"unusual value(s) in "
                    f"**{selected_anomaly_column}**."

                )


                st.dataframe(

                    anomalies.drop(
                        columns=["Result"]
                    ),

                    use_container_width=True

                )

            else:

                st.success(
                    "✅ No significant anomalies detected."
                )


        else:

            st.info(
                "At least 5 numerical values "
                "are required for anomaly detection."
            )


    else:

        st.info(
            "No numerical columns are available."
        )


    # =====================================================
    # TREND DETECTIVE
    # =====================================================

    st.header("📅 Trend Detective")


    if date_columns:

        selected_date = st.selectbox(

            "📅 Select the date column:",

            date_columns,

            key="trend_date"

        )

    else:

        st.warning(

            "⚠️ I couldn't automatically identify "
            "a date column. Select it manually."

        )


        selected_date = st.selectbox(

            "📅 Select the column containing dates:",

            df.columns,

            key="manual_date"

        )


    if numerical_columns:

        selected_metric = st.selectbox(

            "📊 Select the value to analyze:",

            numerical_columns,

            key="trend_metric"

        )


        # Prepare trend data using separate Series. This avoids pandas
        # duplicate-key and dtype-assignment errors.
        if selected_date == selected_metric:

            st.warning(
                "⚠️ The date column and value column cannot be the same. "
                "Please choose a date column for the date selector."
            )

            trend_data = pd.DataFrame(
                columns=["Trend Date", "Trend Value"]
            )

        else:

            trend_data = pd.DataFrame({
                "Trend Date": pd.to_datetime(
                    df[selected_date],
                    errors="coerce"
                ),
                "Trend Value": pd.to_numeric(
                    df[selected_metric],
                    errors="coerce"
                )
            }).dropna()

            trend_data = trend_data.sort_values(
                "Trend Date"
            )

        if len(trend_data) >= 2:

            first_value = float(
                trend_data["Trend Value"].iloc[0]
            )

            last_value = float(
                trend_data["Trend Value"].iloc[-1]
            )

            if first_value != 0:
                percentage_change = (
                    (last_value - first_value)
                    / abs(first_value)
                ) * 100
            else:
                percentage_change = 0

            if percentage_change > 5:

                st.success(
                    f"📈 **Increasing trend detected!** "
                    f"{selected_metric} increased by "
                    f"**{percentage_change:.1f}%**."
                )

            elif percentage_change < -5:

                st.error(
                    f"📉 **Decreasing trend detected!** "
                    f"{selected_metric} decreased by "
                    f"**{abs(percentage_change):.1f}%**."
                )

            else:

                st.info(
                    f"➡️ **Relatively stable.** "
                    f"Overall change: "
                    f"**{percentage_change:+.1f}%**."
                )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Starting Value",
                f"{first_value:,.2f}"
            )

            col2.metric(
                "Latest Value",
                f"{last_value:,.2f}"
            )

            col3.metric(
                "Overall Change",
                f"{percentage_change:+.1f}%"
            )

            # ---------------------------------------------
            # TREND CHART
            # ---------------------------------------------

            st.subheader("📊 Trend Visualization")

            chart = px.line(
                trend_data,
                x="Trend Date",
                y="Trend Value",
                markers=True,
                title=f"{selected_metric} Over Time"
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

        else:

            st.info(

                "Not enough valid date/value pairs "
                "for trend analysis."

            )


    else:

        st.info(

            "No numerical columns are available "
            "for trend analysis."

        )


       # =====================================================
    # 🕵️ SMART DETECTIVE FINDINGS
    # =====================================================

    st.header("🕵️ Smart Detective Findings")

    findings = []
    recommendations = []


    # =====================================================
    # 1. DATA QUALITY
    # =====================================================

    total_missing = int(df.isna().sum().sum())

    duplicate_count = int(df.duplicated().sum())


    if total_missing == 0:

        findings.append(
            "✅ **Data Quality:** "
            "The dataset contains no missing values."
        )

    else:

        findings.append(
            f"⚠️ **Data Quality:** "
            f"{total_missing} missing value(s) were detected."
        )

        recommendations.append(
            "🧹 Consider cleaning or filling the missing values "
            "before making important decisions."
        )


    # =====================================================
    # 2. DUPLICATES
    # =====================================================

    if duplicate_count == 0:

        findings.append(
            "✅ **Duplicates:** "
            "No duplicate rows were detected."
        )

    else:

        findings.append(
            f"⚠️ **Duplicates:** "
            f"{duplicate_count} duplicate row(s) were detected."
        )

        recommendations.append(
            "🧹 Review duplicate rows because they could "
            "distort statistics and business results."
        )


    # =====================================================
    # 3. BEST PERFORMING CATEGORY
    # =====================================================

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()


    if categorical_columns and numerical_columns:

        for category in categorical_columns:

            if category in date_columns:
                continue

            try:

                metric = numerical_columns[0]

                grouped = (
                    df.groupby(category)[metric]
                    .sum()
                    .sort_values(ascending=False)
                )


                if len(grouped) >= 2:

                    best_category = grouped.index[0]
                    best_value = grouped.iloc[0]

                    second_value = grouped.iloc[1]


                    findings.append(
                        f"🏆 **Top {category}:** "
                        f"**{best_category}** generated the "
                        f"highest total {metric}: "
                        f"**{best_value:,.2f}**."
                    )


                    if second_value != 0:

                        difference = (
                            (
                                best_value - second_value
                            )
                            / abs(second_value)
                        ) * 100


                        if difference > 20:

                            recommendations.append(
                                f"💡 **{best_category}** is significantly "
                                f"ahead of the next {category}. "
                                f"Investigate what is driving its "
                                f"strong performance."
                            )

            except Exception:
                pass


    # =====================================================
    # 4. NUMERICAL COLUMN ANALYSIS
    # =====================================================

    for column in numerical_columns:

        try:

            average = df[column].mean()
            maximum = df[column].max()
            minimum = df[column].min()


            if pd.notna(average) and average != 0:

                max_difference = (
                    (maximum - average)
                    / abs(average)
                ) * 100


                if max_difference > 50:

                    findings.append(
                        f"🔎 **{column}:** "
                        f"The maximum value "
                        f"(**{maximum:,.2f}**) is "
                        f"significantly above the average "
                        f"(**{average:,.2f}**)."
                    )

        except Exception:
            pass


    # =====================================================
    # 5. CORRELATION ANALYSIS
    # =====================================================

    if len(numerical_columns) >= 2:

        try:

            correlation_matrix = df[
                numerical_columns
            ].corr()


            strongest_pair = None
            strongest_value = 0


            for i in range(
                len(numerical_columns)
            ):

                for j in range(
                    i + 1,
                    len(numerical_columns)
                ):

                    value = correlation_matrix.iloc[
                        i,
                        j
                    ]


                    if (
                        pd.notna(value)
                        and abs(value) > abs(strongest_value)
                    ):

                        strongest_value = value

                        strongest_pair = (
                            numerical_columns[i],
                            numerical_columns[j]
                        )


            if strongest_pair:

                if strongest_value >= 0.7:

                    findings.append(
                        f"🔗 **Strong Relationship:** "
                        f"**{strongest_pair[0]}** and "
                        f"**{strongest_pair[1]}** have a strong "
                        f"positive relationship "
                        f"(**{strongest_value:.2f}**)."
                    )

                    recommendations.append(
                        f"💡 Investigate whether changes in "
                        f"**{strongest_pair[0]}** are connected "
                        f"to changes in **{strongest_pair[1]}**."
                    )

                elif strongest_value <= -0.7:

                    findings.append(
                        f"🔗 **Strong Relationship:** "
                        f"**{strongest_pair[0]}** and "
                        f"**{strongest_pair[1]}** have a strong "
                        f"negative relationship "
                        f"(**{strongest_value:.2f}**)."
                    )

                else:

                    findings.append(
                        f"🔗 **Relationship:** "
                        f"The strongest relationship is between "
                        f"**{strongest_pair[0]}** and "
                        f"**{strongest_pair[1]}** "
                        f"(**{strongest_value:.2f}**)."
                    )

        except Exception:
            pass


    # =====================================================
    # DISPLAY FINDINGS
    # =====================================================

    st.subheader("🔍 What the Detective Found")

    if findings:

        for finding in findings:

            st.info(finding)

    else:

        st.info(
            "Not enough information to generate findings."
        )


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    st.subheader("💡 Detective Recommendations")

    if recommendations:

        for recommendation in recommendations:

            st.success(recommendation)

    else:

        st.success(
            "✅ No major issues were detected. "
            "The dataset appears relatively clean."
        )

    # =====================================================
    # 📊 VISUAL ANALYTICS
    # =====================================================

    st.header("📊 Visual Analytics")

    st.write(
        "Explore your dataset using interactive charts."
    )

    chart_types = [
        "📊 Bar Chart",
        "📈 Line Chart",
        "🥧 Pie Chart",
        "🔵 Scatter Plot"
    ]

    selected_chart = st.selectbox(
        "Choose a visualization:",
        chart_types
    )


    # =====================================================
    # BAR CHART
    # =====================================================

    if selected_chart == "📊 Bar Chart":

        categorical_columns = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        categorical_columns = [
            column
            for column in categorical_columns
            if column not in date_columns
        ]


        if categorical_columns and numerical_columns:

            selected_category = st.selectbox(
                "Choose a category:",
                categorical_columns,
                key="bar_category"
            )


            selected_value = st.selectbox(
                "Choose a numerical value:",
                numerical_columns,
                key="bar_value"
            )


            chart_data = (
                df.groupby(selected_category)[selected_value]
                .sum()
                .reset_index()
                .sort_values(
                    selected_value,
                    ascending=False
                )
            )


            chart = px.bar(
                chart_data,
                x=selected_category,
                y=selected_value,
                title=(
                    f"{selected_value} by "
                    f"{selected_category}"
                )
            )


            st.plotly_chart(
                chart,
                use_container_width=True,
                key="visual_bar_chart"
            )


        else:

            st.info(
                "A categorical and numerical column "
                "are required for a bar chart."
            )


    # =====================================================
    # LINE CHART
    # =====================================================

    elif selected_chart == "📈 Line Chart":

        if date_columns and numerical_columns:

            selected_date_chart = st.selectbox(
                "Choose a date column:",
                date_columns,
                key="line_date"
            )


            selected_line_value = st.selectbox(
                "Choose a numerical value:",
                numerical_columns,
                key="line_value"
            )


            line_data = df[
                [
                    selected_date_chart,
                    selected_line_value
                ]
            ].copy()


            line_data[selected_date_chart] = (
                pd.to_datetime(
                    line_data[selected_date_chart],
                    errors="coerce"
                )
            )


            line_data = line_data.dropna()


            line_data = line_data.sort_values(
                selected_date_chart
            )


            chart = px.line(
                line_data,
                x=selected_date_chart,
                y=selected_line_value,
                markers=True,
                title=(
                    f"{selected_line_value} Over Time"
                )
            )


            st.plotly_chart(
    chart,
    use_container_width=True,
    key="visual_line_chart"
)


        else:

            st.info(
                "A date column and numerical column "
                "are required for a line chart."
            )


    # =====================================================
    # PIE CHART
    # =====================================================

    elif selected_chart == "🥧 Pie Chart":

        categorical_columns = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()


        categorical_columns = [
            column
            for column in categorical_columns
            if column not in date_columns
        ]


        if categorical_columns and numerical_columns:

            selected_pie_category = st.selectbox(
                "Choose a category:",
                categorical_columns,
                key="pie_category"
            )


            selected_pie_value = st.selectbox(
                "Choose a numerical value:",
                numerical_columns,
                key="pie_value"
            )


            pie_data = (
                df.groupby(selected_pie_category)[
                    selected_pie_value
                ]
                .sum()
                .reset_index()
            )


            chart = px.pie(
                pie_data,
                names=selected_pie_category,
                values=selected_pie_value,
                title=(
                    f"{selected_pie_value} Distribution"
                )
            )


            st.plotly_chart(
    chart,
    use_container_width=True,
    key="visual_pie_chart"
)


        else:

            st.info(
                "A categorical and numerical column "
                "are required for a pie chart."
            )


    # =====================================================
    # SCATTER PLOT
    # =====================================================

    elif selected_chart == "🔵 Scatter Plot":

        if len(numerical_columns) >= 2:

            col1, col2 = st.columns(2)


            with col1:

                x_column = st.selectbox(
                    "Choose X-axis:",
                    numerical_columns,
                    key="scatter_x"
                )


            with col2:

                y_column = st.selectbox(
                    "Choose Y-axis:",
                    numerical_columns,
                    key="scatter_y"
                )


            # Make sure X and Y are different
            if x_column == y_column:

                st.warning(
                    "⚠️ Please choose two different numerical columns."
                )

            else:

                scatter_data = df[
                    [x_column, y_column]
                ].dropna()

                chart = px.scatter(
                    scatter_data,
                    x=x_column,
                    y=y_column,
                    title=f"{y_column} vs {x_column}"
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True,
                    key="visual_scatter_chart"
                )


        else:

            st.info(
                "At least two numerical columns "
                "are required for a scatter plot."
            )


       # =====================================================
    # 📄 PROFESSIONAL INVESTIGATION REPORT
    # =====================================================

    st.header("📄 Investigation Report")

    st.write(
        "Generate a professional data investigation report "
        "containing the dataset summary, findings, statistics, "
        "and recommendations."
    )

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.colors import HexColor
    from io import BytesIO
    from datetime import datetime


    def create_pdf_report():

        buffer = BytesIO()

        pdf = canvas.Canvas(
            buffer,
            pagesize=letter
        )

        width, height = letter

        # -------------------------------------------------
        # COLORS
        # -------------------------------------------------

        dark = HexColor("#172033")
        accent = HexColor("#2563EB")
        light = HexColor("#F3F6FA")
        green = HexColor("#15803D")
        orange = HexColor("#D97706")


        # =================================================
        # PAGE 1 — COVER
        # =================================================

        pdf.setFillColor(dark)

        pdf.rect(
            0,
            0,
            width,
            height,
            fill=1,
            stroke=0
        )


        # Accent block

        pdf.setFillColor(accent)

        pdf.rect(
            0,
            height - 12,
            width,
            12,
            fill=1,
            stroke=0
        )


        # Detective icon

        pdf.setFillColor(colors.white)

        pdf.setFont(
            "Helvetica-Bold",
            32
        )

        pdf.drawString(
            55,
            height - 100,
            "AI DATA"
        )

        pdf.setFillColor(
            HexColor("#60A5FA")
        )

        pdf.drawString(
            55,
            height - 140,
            "DETECTIVE"
        )


        pdf.setFillColor(
            colors.white
        )

        pdf.setFont(
            "Helvetica",
            14
        )

        pdf.drawString(
            57,
            height - 180,
            "Automated Data Investigation Report"
        )


        # Dataset card

        card_y = height - 330

        pdf.setFillColor(
            HexColor("#202B42")
        )

        pdf.roundRect(
            50,
            card_y,
            width - 100,
            105,
            12,
            fill=1,
            stroke=0
        )


        pdf.setFillColor(
            colors.white
        )

        pdf.setFont(
            "Helvetica-Bold",
            13
        )

        pdf.drawString(
            70,
            card_y + 72,
            "INVESTIGATED DATASET"
        )


        pdf.setFont(
            "Helvetica",
            12
        )

        dataset_name = uploaded_file.name

        pdf.drawString(
            70,
            card_y + 48,
            dataset_name[:70]
        )


        pdf.setFont(
            "Helvetica",
            9
        )

        pdf.setFillColor(
            HexColor("#CBD5E1")
        )

        pdf.drawString(
            70,
            card_y + 22,
            f"Generated {datetime.now().strftime('%d %B %Y')}"
        )


        # Bottom branding

        pdf.setFillColor(
            HexColor("#94A3B8")
        )

        pdf.setFont(
            "Helvetica",
            9
        )

        pdf.drawString(
            55,
            45,
            "AI Data Detective • Automated Data Analysis"
        )


        pdf.showPage()


        # =================================================
        # PAGE 2 — EXECUTIVE SUMMARY
        # =================================================

        pdf.setFillColor(
            colors.white
        )

        pdf.rect(
            0,
            0,
            width,
            height,
            fill=1,
            stroke=0
        )


        # Header

        pdf.setFillColor(dark)

        pdf.rect(
            0,
            height - 70,
            width,
            70,
            fill=1,
            stroke=0
        )


        pdf.setFillColor(colors.white)

        pdf.setFont(
            "Helvetica-Bold",
            20
        )

        pdf.drawString(
            50,
            height - 43,
            "Investigation Summary"
        )


        # KPI cards

        card_top = height - 125

        cards = [

            (
                "ROWS",
                str(df.shape[0]),
                accent
            ),

            (
                "COLUMNS",
                str(df.shape[1]),
                HexColor("#7C3AED")
            ),

            (
                "MISSING",
                str(int(df.isna().sum().sum())),
                orange
            ),

            (
                "DUPLICATES",
                str(int(df.duplicated().sum())),
                green
            )

        ]


        card_width = 115
        gap = 20
        start_x = 50


        for i, (label, value, color) in enumerate(cards):

            x = start_x + i * (card_width + gap)

            pdf.setFillColor(light)

            pdf.roundRect(
                x,
                card_top - 75,
                card_width,
                65,
                8,
                fill=1,
                stroke=0
            )


            pdf.setFillColor(color)

            pdf.setFont(
                "Helvetica-Bold",
                18
            )

            pdf.drawString(
                x + 12,
                card_top - 42,
                value
            )


            pdf.setFillColor(dark)

            pdf.setFont(
                "Helvetica",
                8
            )

            pdf.drawString(
                x + 12,
                card_top - 61,
                label
            )


        # Executive summary

        y = card_top - 120

        pdf.setFillColor(dark)

        pdf.setFont(
            "Helvetica-Bold",
            15
        )

        pdf.drawString(
            50,
            y,
            "Executive Assessment"
        )

        y -= 28


        pdf.setFillColor(
            HexColor("#334155")
        )

        pdf.setFont(
            "Helvetica",
            10
        )


        if (
            int(df.isna().sum().sum()) == 0
            and int(df.duplicated().sum()) == 0
        ):

            summary = (
                "The dataset appears structurally clean, "
                "with no missing values or duplicate rows "
                "detected during the investigation."
            )

        else:

            summary = (
                "The investigation identified data-quality "
                "issues that should be reviewed before the "
                "dataset is used for important decisions."
            )


        # Wrap summary

        words = summary.split()

        line = ""

        for word in words:

            test_line = (
                line + " " + word
            ).strip()

            if pdf.stringWidth(
                test_line,
                "Helvetica",
                10
            ) > 480:

                pdf.drawString(
                    55,
                    y,
                    line
                )

                y -= 17

                line = word

            else:

                line = test_line


        if line:

            pdf.drawString(
                55,
                y,
                line
            )

            y -= 25


        # =================================================
        # NUMERICAL ANALYSIS
        # =================================================

        pdf.setFillColor(dark)

        pdf.setFont(
            "Helvetica-Bold",
            15
        )

        pdf.drawString(
            50,
            y,
            "Numerical Analysis"
        )

        y -= 25


        pdf.setFont(
            "Helvetica",
            9
        )

        pdf.setFillColor(
            HexColor("#334155")
        )


        if numerical_columns:

            for column in numerical_columns:

                average = df[column].mean()

                maximum = df[column].max()

                minimum = df[column].min()


                text_line = (
                    f"{column}: "
                    f"minimum {minimum:,.2f} • "
                    f"average {average:,.2f} • "
                    f"maximum {maximum:,.2f}"
                )


                pdf.drawString(
                    60,
                    y,
                    text_line[:100]
                )

                y -= 18


                if y < 80:

                    pdf.showPage()

                    y = height - 60

                    pdf.setFont(
                        "Helvetica",
                        9
                    )


        else:

            pdf.drawString(
                60,
                y,
                "No numerical columns were detected."
            )

            y -= 18

        # =================================================
        # PAGE 3+ — FINDINGS & RECOMMENDATIONS
        # =================================================

        def clean_pdf_text(text):

            text = (
                text
                .replace("**", "")
                .replace("🕵️", "")
                .replace("📊", "")
                .replace("🏆", "")
                .replace("⚠️", "")
                .replace("✅", "")
                .replace("🔗", "")
                .replace("🔎", "")
                .replace("💡", "")
                .replace("🧹", "")
            )

            return (
                text
                .encode("ascii", "ignore")
                .decode()
            )


        def wrap_text(text, max_width, font_name, font_size):

            words = text.split()

            lines = []

            current_line = ""

            for word in words:

                test_line = (
                    word
                    if current_line == ""
                    else current_line + " " + word
                )

                # Normal line fits
                if pdf.stringWidth(
                    test_line,
                    font_name,
                    font_size
                ) <= max_width:

                    current_line = test_line

                else:

                    # Save previous line
                    if current_line:
                        lines.append(current_line)

                    # Handle extremely long words
                    if pdf.stringWidth(
                        word,
                        font_name,
                        font_size
                    ) <= max_width:

                        current_line = word

                    else:

                        # Break very long word safely
                        partial = ""

                        for char in word:

                            test_partial = partial + char

                            if pdf.stringWidth(
                                test_partial,
                                font_name,
                                font_size
                            ) <= max_width:

                                partial = test_partial

                            else:

                                if partial:
                                    lines.append(partial)

                                partial = char

                        current_line = partial

            if current_line:
                lines.append(current_line)

            return lines


        # =================================================
        # PAGE MANAGEMENT
        # =================================================

        bottom_margin = 75
        top_position = height - 105

        y = top_position


        def start_findings_page(title):

            global y

            pdf.showPage()

            pdf.setFillColor(colors.white)

            pdf.rect(
                0,
                0,
                width,
                height,
                fill=1,
                stroke=0
            )

            # Header

            pdf.setFillColor(dark)

            pdf.rect(
                0,
                height - 70,
                width,
                70,
                fill=1,
                stroke=0
            )

            pdf.setFillColor(colors.white)

            pdf.setFont(
                "Helvetica-Bold",
                20
            )

            pdf.drawString(
                50,
                height - 43,
                title
            )

            y = height - 105


        # Start first page

        start_findings_page(
            "Detective Findings"
        )


        # =================================================
        # DRAW FINDING CARD
        # =================================================

        def draw_finding_card(text):

            global y

            font_name = "Helvetica"
            font_size = 10

            text_width = width - 135

            lines = wrap_text(
                text,
                text_width,
                font_name,
                font_size
            )

            line_height = 15

            padding_top = 18
            padding_bottom = 16

            card_height = (
                len(lines) * line_height
                + padding_top
                + padding_bottom
            )

            # If card doesn't fit,
            # start a fresh page.

            if y - card_height < bottom_margin:

                start_findings_page(
                    "Detective Findings (continued)"
                )

            # Card

            pdf.setFillColor(light)

            pdf.roundRect(
                50,
                y - card_height,
                width - 100,
                card_height,
                8,
                fill=1,
                stroke=0
            )

            # Accent

            pdf.setFillColor(accent)

            pdf.roundRect(
                50,
                y - card_height,
                5,
                card_height,
                3,
                fill=1,
                stroke=0
            )

            # Text

            pdf.setFillColor(
                HexColor("#334155")
            )

            pdf.setFont(
                font_name,
                font_size
            )

            text_y = (
                y
                - padding_top
            )

            for line in lines:

                pdf.drawString(
                    68,
                    text_y,
                    line
                )

                text_y -= line_height

            y -= (
                card_height
                + 14
            )


        # =================================================
        # FINDINGS
        # =================================================

        for finding in findings:

            clean = clean_pdf_text(
                finding
            )

            draw_finding_card(
                clean
            )


        # =================================================
        # RECOMMENDATIONS TITLE
        # =================================================

        if recommendations:

            title_height = 35

            # Make sure title + first card
            # have enough space.

            if y - title_height < bottom_margin:

                start_findings_page(
                    "Recommendations"
                )

            else:

                pdf.setFillColor(dark)

                pdf.setFont(
                    "Helvetica-Bold",
                    18
                )

                pdf.drawString(
                    50,
                    y,
                    "Recommendations"
                )

                y -= 30


        # =================================================
        # DRAW RECOMMENDATION CARD
        # =================================================

        def draw_recommendation_card(text):

            global y

            font_name = "Helvetica"
            font_size = 10

            text_width = width - 135

            lines = wrap_text(
                text,
                text_width,
                font_name,
                font_size
            )

            line_height = 15

            padding_top = 18
            padding_bottom = 16

            card_height = (
                len(lines) * line_height
                + padding_top
                + padding_bottom
            )

            # Safe page break

            if y - card_height < bottom_margin:

                start_findings_page(
                    "Recommendations (continued)"
                )

            # Card

            pdf.setFillColor(
                HexColor("#ECFDF5")
            )

            pdf.roundRect(
                50,
                y - card_height,
                width - 100,
                card_height,
                8,
                fill=1,
                stroke=0
            )

            # Green accent

            pdf.setFillColor(
                HexColor("#16A34A")
            )

            pdf.roundRect(
                50,
                y - card_height,
                5,
                card_height,
                3,
                fill=1,
                stroke=0
            )

            # Text

            pdf.setFillColor(
                HexColor("#166534")
            )

            pdf.setFont(
                font_name,
                font_size
            )

            text_y = (
                y
                - padding_top
            )

            for line in lines:

                pdf.drawString(
                    68,
                    text_y,
                    line
                )

                text_y -= line_height

            y -= (
                card_height
                + 14
            )


        # =================================================
        # RECOMMENDATIONS
        # =================================================

        for recommendation in recommendations:

            clean = clean_pdf_text(
                recommendation
            )

            draw_recommendation_card(
                clean
            )


        # =================================================
        # FOOTER
        # =================================================

        pdf.setFillColor(
            HexColor("#64748B")
        )

        pdf.setFont(
            "Helvetica-Oblique",
            8
        )

        pdf.drawString(
            50,
            30,
            "Generated by AI Data Detective"
        )


        # =================================================
        # SAVE PDF
        # =================================================

        pdf.save()

        buffer.seek(0)

        return buffer.getvalue()
    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================

    pdf_data = create_pdf_report()


    st.download_button(

        label="📥 Download Professional Investigation Report",

        data=pdf_data,

        file_name="AI_Data_Detective_Report.pdf",

        mime="application/pdf",

        key="professional_report_download"

    )


    st.markdown(
        """
        <div class="footer-note">
            Built with Python • Streamlit • Pandas • NumPy • Scikit-learn • Plotly • ReportLab
        </div>
        """,
        unsafe_allow_html=True
    )
# =========================================================
# NO FILE UPLOADED
# =========================================================

else:

    st.info(

        "👆 Upload a CSV or Excel file to begin "
        "your investigation."

    )