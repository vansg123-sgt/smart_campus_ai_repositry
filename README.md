# Smart Campus AI — Hackathon Prototype

## What this implements
- Unified student data view
- Student Success Score (0–100)
- Risk identification
- Interactive dashboard
- At-risk student table
- Explainable score drivers
- Student segmentation
- Recommended interventions
- CSV upload for real/demo data

## Run
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## CSV columns
Student_ID, Name, Department, Semester, CGPA, Attendance, LMS_Logins,
Assignment_Completion, Engagement, Placement_Readiness, Skills, Feedback, Backlogs

The app also works without a CSV using generated demo data.

## Suggested hackathon demo
1. Open the dashboard.
2. Show overall KPIs and risk distribution.
3. Open At-Risk Students.
4. Select one student and show the intervention plan.
5. Open Explainable AI and show exactly why the score/risk was produced.
6. Open Segmentation to show targeted student groups.
