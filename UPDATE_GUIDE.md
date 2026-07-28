# Update the Existing Windows Project

Open the downloaded ZIP and copy its contents into:

```text
C:\ML OPs Project
```

Allow Windows to replace the older project files. Keep any personal files that
are not part of this update.

## 1. Activate and install

```bat
cd "C:\ML OPs Project"
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Generate or inspect the dataset

The ZIP already contains the 2,000-row CSV. It can be regenerated identically:

```bat
python tools\generate_dataset.py
```

## 3. Train and compare

```bat
python train.py
type reports\model_comparison.json
mlflow ui --port 5000
```

Open http://127.0.0.1:5000.

## 4. Test FastAPI

```bat
pytest -q
uvicorn app:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs and execute `POST /predict`.

## 5. Update DVC

If Git currently tracks the CSV:

```bat
git rm --cached data\student_attendance.csv
```

Then:

```bat
dvc add data\student_attendance.csv
git add data\student_attendance.csv.dvc data\.gitignore dvc.yaml
```

For a classroom demonstration, show:

```bat
dvc status
dvc repro
```

## 6. Commit and push GitHub

```bat
git status
git add .
git commit -m "Add improved attendance prediction MLOps pipeline"
git push origin main
```

The GitHub repository must contain the two Actions secrets:

```text
DOCKERHUB_USERNAME = softmanoj2
DOCKERHUB_TOKEN = your Docker Hub access token
```

## 7. Build and push Docker manually

```bat
docker login
docker build -t softmanoj2/attendance-api:latest .
docker run --rm -p 8000:8000 softmanoj2/attendance-api:latest
```

After testing, press `Ctrl+C`, then:

```bat
docker push softmanoj2/attendance-api:latest
```

## 8. Deploy the new image to Kubernetes

Ensure Docker Desktop Kubernetes is running:

```bat
kubectl cluster-info
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl rollout restart deployment attendance-api
kubectl rollout status deployment attendance-api
kubectl get pods
kubectl get svc
kubectl port-forward service/attendance-api-service 8000:8000
```

Keep the last terminal open and test http://127.0.0.1:8000/docs.

## 9. Start Prometheus and Grafana

First inspect containers already using the monitoring ports:

```bat
docker ps -a
```

If an older `prometheus` or `grafana` container from this same project is using
ports 9090 or 3000, stop it before continuing:

```bat
docker stop prometheus grafana
```

```bat
cd monitoring
docker compose -f docker-compose.monitoring.yml up -d --build
docker compose -f docker-compose.monitoring.yml ps
```

Open:

```text
API:        http://127.0.0.1:8000/docs
Prometheus: http://127.0.0.1:9090
Grafana:    http://127.0.0.1:3000
cAdvisor:   http://127.0.0.1:8080
```

Generate several predictions in Swagger. Grafana's provisioned
`Student Attendance MLOps` dashboard will then display request count,
predictions per minute, latency, errors, CPU and memory.

## 10. Examiner comparison

Use the actual values in `reports/model_comparison.json`. For the supplied
dataset and validated model:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| IEEE-inspired baseline | 5.0423 | 6.2899 | 0.5902 |
| Proposed enhanced model | 2.7444 | 3.4572 | 0.8762 |

The measured RMSE reduction is 45.04%. These results apply only to the supplied
synthetic prototype dataset and must not be represented as results on real
institutional data.
