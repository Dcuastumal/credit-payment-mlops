// Agente Linux con Python 3.12. Requiere configurar el repositorio en Jenkins.
pipeline {
    agent any
    options { timestamps() }
    stages {
        stage('Entorno') {
            steps {
                sh 'python3.12 -m venv .venv'
                sh '.venv/bin/python -m pip install -r requirements.txt'
                sh '.venv/bin/python -m pip check'
            }
        }
        stage('Pipeline y pruebas') {
            steps { sh '.venv/bin/python run_pipeline.py' }
        }
    }
    post {
        success {
            archiveArtifacts artifacts: 'src/metrics.json,src/monitoring_report.json,src/evaluation.png,src/model.joblib,coverage.xml', fingerprint: true
        }
    }
}
