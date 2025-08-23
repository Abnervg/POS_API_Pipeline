# 🧾 Automated POS Sales Data Pipeline & Reporting

## 🔍 Project Overview

This project implements a complete, production-grade data engineering pipeline that automates the extraction, transformation, and analysis of transactional data from a Point of Sale (POS) system.

Raw JSON data is fetched **incrementally** from the POS API, processed using a robust Python and Pandas-based ETL, and loaded into a **partitioned data lake** on AWS S3. The entire workflow is containerized with **Docker** and orchestrated with **Apache Airflow**. The pipeline runs on a daily schedule, and on the first of each month, it automatically generates and emails comprehensive business intelligence reports.

## ✨ Key Features

* **Incremental ETL**: Intelligently fetches only new receipts since the last successful run, using a persistent state file to track progress. This makes the daily runs fast and efficient.
* **Automated Orchestration**: The entire pipeline is managed by an Apache Airflow DAG, which handles scheduling, dependency management, branching logic, and error retries.
* **Containerized Environment**: The application is fully containerized with Docker, ensuring a consistent and portable environment for both development and production.
* **Partitioned Data Lake**: Curated data is saved in the efficient Parquet format and partitioned by year and month in an AWS S3 data lake, which is a best practice for optimized querying and scalability.
* **Automated Reporting Suite**:
    * **Monthly Report**: Compares the last full month's performance against the previous month.
    * **Cumulative Report**: Provides a complete historical analysis of all data.
    * **Automated Delivery**: Reports are automatically converted from Markdown to PDF and emailed to stakeholders.
* **Advanced Analytics**: Includes sophisticated data cleaning (e.g., exploding combo items for accurate counting) and analysis (e.g., Market Basket Analysis to find popular product combinations).

## 🛠️ Tech Stack

| **Layer** | **Tool/Service** |
| -------------------- | -------------------------------------------------------------------- |
| **Orchestration** | Apache Airflow                                                       |
| **Containerization** | Docker, Docker Compose                                               |
| **CI/CD & Scheduling**| GitHub Actions                                                      |
| **Language** | Python                                                               |
| **Data Processing** | Pandas                                                               |
| **Cloud Storage** | AWS S3 (Data Lake)                                                   |
| **Python Libraries** | Requests, Boto3, s3fs, PyArrow, Seaborn, Matplotlib, python-dotenv, mlxtend |
| **Testing** | Pytest                                                               |

## 🗂️ Project Structure

```
POS_API_Pipeline/
├── .github/
│   └── workflows/
│       └── monthly_etl.yml   # GitHub Actions workflow for scheduling
├── config/
│   ├── config.env          # For local configuration (ignored by Git)
│   └── etl_state.json        # Tracks last extraction timestamp (ignored by Git)
├── dags/
│   └── production_etl_dag.py # Airflow DAG definition
├── data/
│   └── ...                 # Local storage for raw/curated data (dev only)
├── etl/
│   ├── __init__.py
│   ├── extract.py
│   ├── transform.py
│   └── load.py
├── reporting/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── monthly_report.py
│   └── cumulative_report.py
├── tests/
│   └── ...                 # Unit tests for the pipeline
├── .gitignore
├── Dockerfile              # Defines the application's container image
├── docker-compose.yml      # For running the local Airflow environment
├── main.py                 # Main script orchestrator (callable from CLI)
└── requirements.txt        # Python dependencies
```

## 🚀 How to Run

### 1. Local Development

You can run any step of the pipeline manually using the command line.

```bash
# Example: Run the daily incremental ETL
python main.py --step daily_run

# Example: Generate the cumulative report
python main.py --step cumulative_report
```

### 2. Automated Run with Airflow

This project is designed to be run with Apache Airflow.

1.  **Build the Docker Image**:
    ```bash
    docker build -t pos-etl-pipeline .
    ```
2.  **Set Up Airflow Variables**: In the Airflow UI (`Admin` > `Variables`), create the necessary variables for your host machine's file paths (e.g., `HOST_CONFIG_FOLDER_PATH`, `HOST_AWS_CREDS_PATH`).
3.  **Start the Airflow Environment**:
    ```bash
    docker-compose up -d
    ```
4.  **Trigger the DAG**: Go to `http://localhost:8080`, un-pause the `daily_etl_and_monthly_reporting` DAG, and trigger it.