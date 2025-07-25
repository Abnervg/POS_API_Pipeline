from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models import Variable
from docker.types import Mount

# --- Define all host paths by reading from Airflow Variables at the top ---
# This makes the DAG clean and portable.
host_config_folder_path = Variable.get("HOST_CONFIG_FOLDER_PATH", default_var="/path/to/your/config")
host_aws_creds_path = Variable.get("HOST_AWS_CREDS_PATH", default_var="/path/to/your/.aws")


with DAG(
    dag_id="pos_etl_pipeline",
    start_date=pendulum.datetime(2025, 7, 25, tz="America/Mexico_City"),
    schedule="0 5 * * *",  # Runs daily at 5 AM UTC
    catchup=False,
    tags=["etl", "pos"],
) as dag:
    run_etl_task = DockerOperator(
        task_id="run_full_etl",
        image="pos-etl-pipeline:latest",
        api_version="auto",
        auto_remove=True,
        command="python main.py --step all",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        # --- Corrected Mounts Section ---
        mounts=[
            # Mount the entire config folder to support both .env and state.json files
            Mount(
                target="/app/config",          # Path inside the container
                source=host_config_folder_path, # Path on your host machine
                type="bind"
            ),
            Mount(
                target="/root/.aws",            # Path inside the container
                source=host_aws_creds_path,     # Path on your host machine
                type="bind"
            )
        ]
    )