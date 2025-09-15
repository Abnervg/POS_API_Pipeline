from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from docker.types import Mount

# --- Constants and Configuration ---
DOCKER_IMAGE = "pos-etl-pipeline:latest"
HOST_CONFIG_FOLDER_PATH = Variable.get("HOST_CONFIG_FOLDER_PATH", default_var="/path/to/config")
HOST_AWS_CREDS_PATH = Variable.get("HOST_AWS_CREDS_PATH", default_var="/path/to/.aws")

# --- Branching Function ---
def check_if_first_day_of_month(**kwargs):
    """
    Checks if the DAG is running on the first day of the month.
    """
    execution_date = kwargs["data_interval_end"]
    if execution_date.day == 1:
        # If it is, run the reporting tasks
        return ["run_monthly_report_task", "run_cumulative_report_task"]
    else:
        # Otherwise, skip to the end
        return "skip_reporting_task"

# --- DAG Definition ---
with DAG(
    dag_id="daily_etl_and_monthly_reporting",
    start_date=pendulum.datetime(2025, 7, 25, tz="America/Mexico_City"),
    schedule="0 7 * * *",  # Runs daily at 7 AM UTC (1 AM local time)
    catchup=False,
    tags=["production", "etl", "reporting"],
) as dag:

    start = EmptyOperator(task_id="start")

    # Task 1: Always run the daily incremental ETL
    run_daily_etl = DockerOperator(
        task_id="run_daily_incremental_etl",
        image=DOCKER_IMAGE,
        command="python main.py --step daily_run",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(target="/app/config", source=HOST_CONFIG_FOLDER_PATH, type="bind"),
            Mount(target="/root/.aws", source=HOST_AWS_CREDS_PATH, type="bind")
        ]
    )

    # Task 2: Check the date to decide if reports should run
    check_date = BranchPythonOperator(
        task_id="check_if_first_day_of_month",
        python_callable=check_if_first_day_of_month,
    )

    # --- Reporting Branch ---
    run_monthly_report = DockerOperator(
        task_id="run_monthly_report_task",
        image=DOCKER_IMAGE,
        command="python main.py --step monthly_report",
        # (All other DockerOperator parameters are the same)
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(target="/app/config", source=HOST_CONFIG_FOLDER_PATH, type="bind"),
            Mount(target="/root/.aws", source=HOST_AWS_CREDS_PATH, type="bind")
        ]
    )

    run_cumulative_report = DockerOperator(
        task_id="run_cumulative_report_task",
        image=DOCKER_IMAGE,
        command="python main.py --step cumulative_report",
        # (All other DockerOperator parameters are the same)
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(target="/app/config", source=HOST_CONFIG_FOLDER_PATH, type="bind"),
            Mount(target="/root/.aws", source=HOST_AWS_CREDS_PATH, type="bind")
        ]
    )

    # --- Skip Branch ---
    skip_reporting = EmptyOperator(task_id="skip_reporting_task")

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    # --- Define the Workflow ---
    start >> run_daily_etl >> check_date
    check_date >> [run_monthly_report, run_cumulative_report] >> end
    check_date >> skip_reporting >> end