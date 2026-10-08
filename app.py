
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Smart Campus AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Professional UI
# -----------------------------
st.markdown("""
<style>
    .stApp { background: #f6f8fc; }
    .block-container { padding-top: 1.5rem; max-width: 1450px; }
    [data-testid="stSidebar"] { background: #101828; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .hero {
        background: linear-gradient(135deg,#101828 0%,#173b73 55%,#2563eb 100%);
        padding: 28px 32px; border-radius: 20px; color: white;
        box-shadow: 0 12px 30px rgba(16,24,40,.18); margin-bottom: 22px;
    }
    .hero h1 { margin: 0; font-size: 34px; }
    .hero p { margin: 8px 0 0; color: #dbeafe; font-size: 16px; }
    .metric-card {
        background: white; border: 1px solid #e7ebf3; border-radius: 16px;
        padding: 18px; box-shadow: 0 4px 14px rgba(16,24,40,.05);
    }
    .metric-label { color:#667085; font-size:13px; }
    .metric-value { color:#101828; font-size:28px; font-weight:800; margin-top:5px; }
    .section {
        background:white; border:1px solid #e7ebf3; border-radius:18px;
        padding:20px; margin-bottom:18px;
    }
    .pill {
        display:inline-block; padding:5px 10px; border-radius:999px;
        font-size:12px; font-weight:700;
    }
    .high { background:#fee4e2; color:#b42318; }
    .medium { background:#fff0c2; color:#8a5a00; }
    .low { background:#dcfae6; color:#067647; }
    .small-note { color:#667085; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Demo data
# -----------------------------
@st.cache_data
def make_demo_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Student_ID": [f"STU{1000+i}" for i in range(n)],
        "Name": [f"Student {i+1}" for i in range(n)],
        "Department": rng.choice(["CSE","ECE","IT","ME","MBA"], n, p=[.30,.18,.20,.17,.15]),
        "Semester": rng.integers(1, 9, n),
        "CGPA": np.clip(rng.normal(7.1, 1.1, n), 4.0, 10.0),
        "Attendance": np.clip(rng.normal(76, 14, n), 35, 100),
        "LMS_Logins": np.clip(rng.normal(22, 9, n), 2, 50),
        "Assignment_Completion": np.clip(rng.normal(78, 17, n), 20, 100),
        "Engagement": np.clip(rng.normal(55, 22, n), 0, 100),
        "Placement_Readiness": np.clip(rng.normal(58, 22, n), 0, 100),
        "Skills": np.clip(rng.normal(62, 19, n), 0, 100),
        "Feedback": np.clip(rng.normal(72, 15, n), 25, 100),
        "Backlogs": np.clip(rng.poisson(1.1, n), 0, 8),
    })
    # Inject realistic relationships
    df["Student_Success_Score"] = (
        df.CGPA/10*30 +
        df.Attendance/100*15 +
        (df.LMS_Logins/50*7.5 + df.Assignment_Completion/100*7.5) +
        df.Engagement/100*10 +
        df.Placement_Readiness/100*10 +
        df.Skills/100*10 +
        df.Feedback/100*10 -
        df.Backlogs*2.0
    ).clip(0,100)

    df["Risk_Probability"] = (
        1 / (1 + np.exp(
            (df.Student_Success_Score - 55)/7
        ))
    ).clip(.01,.99)

    # Extra risk signals
    df.loc[df.Attendance < 60, "Risk_Probability"] += .10
    df.loc[df.Backlogs >= 3, "Risk_Probability"] += .10
    df["Risk_Probability"] = df.Risk_Probability.clip(0, .99)

    df["Risk_Level"] = pd.cut(
        df.Risk_Probability,
        bins=[-0.01,.35,.65,1],
        labels=["Low","Medium","High"]
    )
    return df

def normalize(v, lo, hi):
    return float(np.clip((v-lo)/(hi-lo)*100, 0, 100))

def calculate_score(row):
    # Transparent, explainable scoring
    parts = {
        "Academic": normalize(row["CGPA"], 4, 10) * .30,
        "Attendance": row["Attendance"] * .15,
        "LMS": (
            normalize(row["LMS_Logins"], 2, 50) * .075 +
            row["Assignment_Completion"] * .075
        ),
        "Engagement": row["Engagement"] * .10,
        "Placement": row["Placement_Readiness"] * .10,
        "Skills": row["Skills"] * .10,
        "Feedback": row["Feedback"] * .10,
    }
    score = sum(parts.values()) - min(float(row["Backlogs"]) * 2, 10)
    return max(0, min(100, score)), parts

def risk_level(row):
    score, _ = calculate_score(row)
    if score < 50 or row.Attendance < 55 or row.Backlogs >= 4:
        return "High"
    if score < 68 or row.Attendance < 70 or row.Backlogs >= 2:
        return "Medium"
    return "Low"

def recommendation(row):
    actions = []
    if row.Attendance < 70:
        actions.append("Attendance mentoring")
    if row.Assignment_Completion < 65:
        actions.append("Academic support / assignment plan")
    if row.CGPA < 6.5 or row.Backlogs >= 2:
        actions.append("Faculty academic intervention")
    if row.Placement_Readiness < 55:
        actions.append("Placement-readiness bootcamp")
    if row.Skills < 55:
        actions.append("Technical + soft-skill training")
    if row.Engagement < 40:
        actions.append("Clubs/events engagement")
    if not actions:
        actions.append("Continue current support and monitor monthly")
    return actions

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## 🎓 Smart Campus AI")
st.sidebar.caption("Student Success & Decision Intelligence")
uploaded = st.sidebar.file_uploader("Upload student CSV", type=["csv"])
st.sidebar.divider()
st.sidebar.markdown("### Demo controls")
dept_filter = st.sidebar.multiselect(
    "Department",
    ["CSE","ECE","IT","ME","MBA"],
    default=["CSE","ECE","IT","ME","MBA"]
)
risk_filter = st.sidebar.multiselect(
    "Risk level", ["High","Medium","Low"],
    default=["High","Medium","Low"]
)

if uploaded:
    df = pd.read_csv(uploaded)
    required = [
        "Student_ID","Name","Department","Semester","CGPA","Attendance",
        "LMS_Logins","Assignment_Completion","Engagement",
        "Placement_Readiness","Skills","Feedback","Backlogs"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("CSV is missing: " + ", ".join(missing))
        st.stop()
    df["Student_Success_Score"] = df.apply(lambda r: calculate_score(r)[0], axis=1)
    df["Risk_Level"] = df.apply(risk_level, axis=1)
    df["Risk_Probability"] = df.apply(
        lambda r: min(.99, max(.01, 1 - r["Student_Success_Score"]/100)),
        axis=1
    )
else:
    df = make_demo_data()

view = df[df.Department.isin(dept_filter) & df.Risk_Level.astype(str).isin(risk_filter)].copy()

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
  <h1>🎓 Smart Campus Analytics</h1>
  <p>AI-powered Student Success Platform — Predict, Optimize & Improve Student Outcomes</p>
</div>
""", unsafe_allow_html=True)

# KPIs
c1,c2,c3,c4,c5 = st.columns(5)
high = int((view.Risk_Level.astype(str)=="High").sum())
medium = int((view.Risk_Level.astype(str)=="Medium").sum())
avg_score = view.Student_Success_Score.mean()
avg_att = view.Attendance.mean()
placement = view.Placement_Readiness.mean()

cards = [
    ("Students", f"{len(view):,}"),
    ("Avg Success Score", f"{avg_score:.1f}/100"),
    ("High Risk", f"{high}"),
    ("Medium Risk", f"{medium}"),
    ("Placement Readiness", f"{placement:.1f}%"),
]
for col,(label,value) in zip([c1,c2,c3,c4,c5],cards):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.write("")

# -----------------------------
# Dashboard tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🚩 At-Risk Students", "🧠 Explainable AI", "👥 Segmentation"
])

with tab1:
    left,right = st.columns([1.15,1])
    with left:
        st.markdown('<div class="section"><h3>Risk Distribution</h3>', unsafe_allow_html=True)
        counts = view.Risk_Level.astype(str).value_counts().reindex(["Low","Medium","High"]).fillna(0)
        fig = px.bar(
            x=counts.index, y=counts.values,
            labels={"x":"Risk Level","y":"Students"},
            text=counts.values
        )
        fig.update_layout(height=330, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section"><h3>Success Score Distribution</h3>', unsafe_allow_html=True)
        fig2 = px.histogram(
            view, x="Student_Success_Score", nbins=20,
            labels={"Student_Success_Score":"Success Score"}
        )
        fig2.add_vline(x=50, line_dash="dash")
        fig2.update_layout(height=330, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><h3>Key Campus Trends</h3>', unsafe_allow_html=True)
    trend = view.groupby("Department", as_index=False).agg(
        Success_Score=("Student_Success_Score","mean"),
        Attendance=("Attendance","mean"),
        Placement_Readiness=("Placement_Readiness","mean"),
    )
    fig3 = px.bar(
        trend, x="Department", y=["Success_Score","Attendance","Placement_Readiness"],
        barmode="group"
    )
    fig3.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section"><h3>🚩 Students Requiring Attention</h3><p class="small-note">Prioritize intervention using transparent risk signals.</p>', unsafe_allow_html=True)
    risky = view[view.Risk_Level.astype(str).isin(["High","Medium"])].copy()
    risky = risky.sort_values(["Risk_Level","Student_Success_Score"], ascending=[True,True])
    display_cols = [
        "Student_ID","Name","Department","Semester","Student_Success_Score",
        "Risk_Level","Attendance","CGPA","Backlogs","Placement_Readiness"
    ]
    st.dataframe(
        risky[display_cols].round(1),
        use_container_width=True, hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if len(risky):
        selected = st.selectbox(
            "Select a student for an intervention plan",
            risky.Student_ID.tolist()
        )
        row = risky[risky.Student_ID==selected].iloc[0]
        score, parts = calculate_score(row)
        st.markdown('<div class="section"><h3>Recommended Intervention</h3>', unsafe_allow_html=True)
        for item in recommendation(row):
            st.write("•", item)
        st.info("Suggested approach: contact the student, validate the signal with faculty, assign the relevant support, and review progress after 2–4 weeks.")
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section"><h3>🧠 Explainable Student Success Score</h3><p class="small-note">The score is intentionally transparent: every component has a visible contribution.</p>', unsafe_allow_html=True)
    student_id = st.selectbox("Choose student", view.Student_ID.tolist())
    row = view[view.Student_ID==student_id].iloc[0]
    score, parts = calculate_score(row)

    a,b,c = st.columns(3)
    a.metric("Success Score", f"{score:.1f}/100")
    b.metric("Risk Level", risk_level(row))
    c.metric("Attendance", f"{row.Attendance:.1f}%")

    contrib = pd.DataFrame({
        "Indicator": list(parts.keys()),
        "Contribution": list(parts.values())
    }).sort_values("Contribution", ascending=False)

    fig = px.bar(contrib, x="Contribution", y="Indicator", orientation="h")
    fig.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Top positive drivers:**", ", ".join(contrib.head(3).Indicator.tolist()))
    st.write("**Potential risk signals:**")
    signals = []
    if row.Attendance < 70: signals.append(f"Attendance is {row.Attendance:.1f}%")
    if row.CGPA < 6.5: signals.append(f"CGPA is {row.CGPA:.1f}")
    if row.Backlogs >= 2: signals.append(f"{int(row.Backlogs)} backlogs")
    if row.Placement_Readiness < 55: signals.append(f"Placement readiness is {row.Placement_Readiness:.1f}%")
    if row.Engagement < 40: signals.append(f"Low engagement ({row.Engagement:.1f}%)")
    if not signals: signals.append("No major negative signal detected")
    for s in signals: st.write("•", s)
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section"><h3>👥 Student Segmentation</h3><p class="small-note">Meaningful groups enable targeted interventions instead of one-size-fits-all support.</p>', unsafe_allow_html=True)

    def segment(r):
        if r.CGPA >= 7.5 and r.Placement_Readiness < 55:
            return "High Academic / Low Placement Readiness"
        if r.CGPA < 6.5 and r.Attendance < 70:
            return "Academic + Attendance Support"
        if r.Engagement < 40:
            return "Low Engagement"
        if r.Placement_Readiness >= 75 and r.Skills >= 70:
            return "Placement Ready"
        if r.Student_Success_Score >= 75:
            return "High Overall Potential"
        return "General Monitoring"

    view["Segment"] = view.apply(segment, axis=1)
    seg = view.Segment.value_counts().reset_index()
    seg.columns = ["Segment","Students"]

    fig = px.pie(seg, names="Segment", values="Students", hole=.55)
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        view[["Student_ID","Name","Department","Student_Success_Score","Risk_Level","Segment"]]
        .sort_values("Student_Success_Score"),
        use_container_width=True, hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Smart Campus AI • Hackathon prototype • Scores are decision-support signals, not final judgments. Validate interventions with faculty and students.")
