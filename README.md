# 🧾 POS API Data Pipeline – Serverless Data Lake on AWS

## 🔍 Project Overview

This project builds a lightweight, serverless data engineering pipeline that extracts transactional data from a Point of Sale (POS) system via its official API. The extracted data is cleaned and stored in an AWS S3-based data lake, showcasing real-world data ingestion and processing using only free-tier services.

---

## 🎯 Objectives

- Integrate securely with a real-world POS API
- Extract, transform, and load transactional data to a cloud storage layer
- Use AWS S3 to create a simple, tier-free data lake
- Demonstrate modular and reproducible ETL architecture
- Prepare for downstream analytics and reporting

---

## 🛠️ Tech Stack

| Layer         | Tool/Service            |
|---------------|--------------------------|
| Language       | Python (Requests, Pandas, Boto3) |
| API Handling   | RESTful HTTP (JSON)       |
| Storage        | AWS S3 (Free Tier)        |
| Scheduling     | Cron or AWS Lambda (optional) |
| Secrets        | `.env` file (dotenv) or AWS Secrets Manager |
| Documentation  | Markdown (README), Diagrams |
| Data Format    | CSV / Parquet             |

---

## 🗂️ Project Structure

