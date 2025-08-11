# **🧾 Automated POS Sales Data Pipeline & Reporting**

## **🔍 Project Overview**

This project implements a complete, production-grade data engineering pipeline that automates the extraction, transformation, and analysis of transactional data from a Point of Sale (POS) system.

Raw JSON data is fetched incrementally from the POS API, processed using a robust Python and Pandas-based ETL, and loaded into a partitioned data lake on AWS S3. The entire workflow is containerized with **Docker** and orchestrated with **Apache Airflow**, with automated daily runs scheduled via **GitHub Actions**. The pipeline culminates in the generation of automated monthly and cumulative business intelligence reports, providing actionable insights into sales trends, product performance, and customer behavior.

## **✨ Key Features**

* **Incremental Data Extraction**: Intelligently fetches only new receipts since the last successful run, using a persistent state file to track progress.  
* **Automated Orchestration**: The entire pipeline is managed by an Apache Airflow DAG, which handles scheduling, dependency management, and error retries.  
* **Containerized Environment**: The application is fully containerized with Docker, ensuring consistency and portability between development and production environments.  
* **Advanced Data Transformation**: Includes sophisticated cleaning logic to handle complex, nested JSON data, such as "exploding" combo meal items into individual rows for accurate sales counting.  
* **Partitioned Data Lake**: Curated data is saved in the efficient Parquet format and partitioned by year and month in an AWS S3 data lake for optimized querying.  
* **Automated Reporting**: Generates monthly and cumulative reports in Markdown format, complete with embedded visualizations that highlight key business metrics and trends.  
* **CI/CD Automation**: A GitHub Actions workflow automates the daily execution of the ETL pipeline, providing a serverless and reliable scheduling solution.

## **🛠️ Tech Stack**

| Layer | Tool/Service |
| :---- | :---- |
| **Orchestration** | Apache Airflow |
| **Containerization** | Docker, Docker Compose |
| **CI/CD & Scheduling** | GitHub Actions |
| **Language** | Python |
| **Data Processing** | Pandas |
| **Cloud Storage** | AWS S3 (Data Lake) |
| **Python Libraries** | Requests, Boto3, s3fs, PyArrow, Seaborn, Matplotlib, python-dotenv |
| **Testing** | Pytest |

## **🗂️ Project Structure**

POS\_API\_Pipeline/  
├── .github/  
│   └── workflows/  
│       └── monthly\_etl.yml   \# GitHub Actions workflow for scheduling  
├── config/  
│   ├── config.env          \# For local configuration (ignored by Git)  
│   └── etl\_state.json        \# Tracks last extraction timestamp (ignored by Git)  
├── dags/  
│   └── incremental\_etl\_dag.py \# Airflow DAG definition  
├── data/  
│   ├── raw/                \# Stores raw extracted JSON data locally  
│   └── curated/            \# Stores final processed data locally  
├── etl/  
│   ├── \_\_init\_\_.py  
│   ├── extract.py  
│   ├── transform.py  
│   └── load.py  
├── reporting/  
│   ├── \_\_init\_\_.py  
│   ├── data\_preparation.py  
│   ├── monthly\_report.py  
│   └── cumulative\_report.py  
├── tests/  
│   └── ...                 \# Unit tests for the pipeline  
├── .gitignore  
├── Dockerfile              \# Defines the application's container image  
├── docker-compose.yml      \# For running the local Airflow environment  
├── main.py                 \# Main script orchestrator (callable from CLI)  
└── requirements.txt        \# Python dependencies  
