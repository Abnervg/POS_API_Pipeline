from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import task, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from docker.types import Mount
from pathlib import Path
import json

# It's good practice to define constants for your image and paths
DOCKER_IMAGE = "pos-etl-pipeline:latest"
HOST_CONFIG_FOLDER_PATH = Variable.get("HOST_CONFIG_FOLDER_PATH", default_var="/path/to/config")
HOST_AWS_CREDS_PATH = Variable.get("HOST_AWS_CREDS_PATH", default_var="/path/to/.aws")
# --- New: Make the threshold configurable via an Airflow Variable ---
RECEIPT_THRESHOLD = Variable.get("RECEIPT_THRESHOLD", default_var=130, deserialize_json=True)


def _read_last_timestamp(state_file_path):
    """Helper function to read the last timestamp from the state file."""
    try:
        with open(state_file_path, 'r') as f:
            state = json.load(f)
        return state.get('last_successful_extraction_timestamp')
    except (FileNotFoundError, KeyError):
        # Dynamically calculate the start of the current month in the correct timezone
        now = pendulum.now("America/Mexico_City")
        start_of_month = now.start_of('month')
        # Convert to UTC and format for the API (e.g., "2025-07-01T06:00:00.000Z")
        return start_of_month.in_tz('UTC').to_iso8601_string().replace('+00:00', 'Z')

@task.branch(task_id="check_run_conditions")
def check_run_conditions_func(**kwargs):
    """
    Checks if the ETL should run based on a receipt threshold OR if it's the
    first day of the month.
    """
    import requests
    import os
    from dotenv import load_dotenv
    import pendulum

    # Get the path to the config folder on the host machine where Airflow is running
    config_folder = Path(HOST_CONFIG_FOLDER_PATH)
    load_dotenv(config_folder / "config.env")
    
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("POS_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}
    
    state_file = config_folder / "etl_state.json"
    last_timestamp = _read_last_timestamp(state_file)

    # --- Logic to check for new receipts ---
    # We fetch one page to get a count of new items.
    # Note: This assumes the API returns a full page if there are many new items.
    params = {'created_min': last_timestamp}
    receipts_url = f"{base_url}/receipts"
    response = requests.get(receipts_url, headers=headers, params=params)
    response.raise_for_status()
    new_receipt_count = len(response.json().get("receipts", []))

    # --- Logic to check the date ---
    # We use the logical date from Airflow for reliable scheduling
    logical_date = kwargs["logical_date"]
    is_first_of_month = logical_date.day == 1

    # --- The Decision Logic ---
    if new_receipt_count >= RECEIPT_THRESHOLD or is_first_of_month:
        print(f"Condition met. New receipts: {new_receipt_count}, Is first of month: {is_first_of_month}. Will run ETL.")
        return "run_full_etl_task"
    else:
        print(f"Condition not met. New receipts: {new_receipt_count}. Skipping ETL for today.")
        return "skip_etl_task"

with DAG(
    dag_id="incremental_pos_etl_pipeline",
    start_date=pendulum.datetime(2025, 7, 25, tz="America/Mexico_City"),
    schedule="0 5 * * *",  # Runs daily at 5 AM UTC
    catchup=False,
    tags=["etl", "pos", "incremental"],
) as dag:
    
    start_task = EmptyOperator(task_id="start")
    
    check_conditions = check_run_conditions_func()

    skip_etl_task = EmptyOperator(task_id="skip_etl_task")

    run_full_etl_task = DockerOperator(
        task_id="run_full_etl_task",
        image=DOCKER_IMAGE,
        api_version="auto",
        auto_remove=True,
        command="python main.py --step daily_run", # This will run your incremental logic
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(target="/app/config", source=HOST_CONFIG_FOLDER_PATH, type="bind"),
            Mount(target="/root/.aws", source=HOST_AWS_CREDS_PATH, type="bind")
        ]
    )

    end_task = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    # Define the workflow path
    start_task >> check_conditions >> [run_full_etl_task, skip_etl_task] >> end_task