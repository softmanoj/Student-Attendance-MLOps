# Student Attendance Prediction — Feature-Enhanced MLOps

This update compares an IEEE-inspired baseline Random Forest model with a
feature-enhanced and tuned Random Forest model. The included 2,000-row dataset
is synthetic and reproducible; it is suitable for an academic prototype, not
for real institutional decision-making.

## Existing and proposed models

- Baseline features: Previous_Attendance, Timetable, Semester, Holidays,
  Internal_Exams.
- Proposed additions: Classes_Conducted, Classes_Attended,
  Current_Attendance_Rate, Approved_Leave_Days,
  Assignment_Completion_Rate, Study_Hours_Per_Week, Travel_Distance_Km.
- Metrics: MAE, RMSE and R².
- Comparison output: `reports/model_comparison.json`.

## Local execution

```bat
venv\Scripts\activate
pip install -r requirements.txt
python tools\generate_dataset.py
python train.py
mlflow ui --port 5000
```

In a second terminal:

```bat
venv\Scripts\activate
uvicorn app:app --reload --port 8000
```

Open:

- API: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- Metrics: http://127.0.0.1:8000/metrics
- MLflow: http://127.0.0.1:5000

## Docker

```bat
docker login
docker build -t softmanoj2/attendance-api:latest .
docker run --rm -p 8000:8000 softmanoj2/attendance-api:latest
docker push softmanoj2/attendance-api:latest
```

## Kubernetes

Start Docker Desktop and enable Kubernetes, then:

```bat
kubectl cluster-info
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl rollout status deployment/attendance-api
kubectl get pods
kubectl get svc
kubectl port-forward service/attendance-api-service 8000:8000
```

Keep the port-forward terminal running while testing.

## Prometheus and Grafana

The compose file starts the API, Prometheus, cAdvisor and Grafana together:

```bat
cd monitoring
docker compose -f docker-compose.monitoring.yml up -d
docker compose -f docker-compose.monitoring.yml ps
```

Open:

- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000
- cAdvisor: http://127.0.0.1:8080
- Grafana default login: admin / admin

The dashboard is automatically provisioned under the `MLOps` folder. API,
prediction, latency, CPU and memory panels are included. If Docker Desktop
prevents cAdvisor from reading container metrics, the API panels still work;
use the Prometheus/Grafana Kubernetes Helm stack for cluster-level CPU and
memory monitoring.

## Git and DVC update

If the old CSV is already tracked by Git:

```bat
git rm --cached data/student_attendance.csv
dvc add data/student_attendance.csv
git add data/student_attendance.csv.dvc data/.gitignore
git add .
git commit -m "Add feature-enhanced attendance model and MLOps pipeline"
git push origin main
```

Configure a DVC remote before expecting another machine or CI to run
`dvc pull`. Do not commit secrets or Docker Hub tokens.

## GitHub Actions secrets

Repository → Settings → Secrets and variables → Actions:

- `DOCKERHUB_USERNAME`: `softmanoj2`
- `DOCKERHUB_TOKEN`: a Docker Hub access token

## IEEE basis

T. Devasia, T. P. Vinushree, and V. Hegde, "Prediction of students
performance using Educational Data Mining," SAPIENCE, 2016,
doi: 10.1109/SAPIENCE.2016.7684167.
