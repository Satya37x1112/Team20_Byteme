## Smart Identity Verification System

**Overview**

-The Insider Threat Detection and Smart Identity Verification System is a security-focused platform designed to monitor user behavior, prevent identity misuse, and detect anomalous activities in real time.

-The system combines behavioral analytics, trust-based scoring, and controlled identity verification to simulate how modern Security Operations Centers (SOC) protect corporate environments from internal threats and unauthorized access.

It demonstrates enterprise security principles such as:

-Baseline-driven behavioral analysis

-Zero Trust philosophy

-Role-aware monitoring

-False-positive minimization

-Real-time risk evaluation

This project is structured as a modular, scalable platform that can be extended into a production-grade security solution.

# Problem Statement

-Organizations face increasing risks from insider threats, credential misuse, and unauthorized access.

-Traditional monitoring systems rely on static thresholds that often generate false alerts or fail to detect subtle behavioral drift.

This system addresses those limitations by:

-Learning normal behavior per user

-Detecting deviations instead of relying on global averages

-Dynamically adjusting trust scores

-Providing real-time SOC-style monitoring dashboards

# Key Features

-Behavioral Trust Engine

-Evaluates each user relative to their personal baseline rather than comparing users globally.

-Insider Threat Detection

**Identifies suspicious behavior patterns such as:**

-abnormal file access

-repeated login failures

-location changes

-after-hours activity

-unauthorized device usage

-Visitor Monitoring

-Supports role-based behavior modeling to ensure visitors are monitored differently from employees while avoiding unnecessary alerts.

**Random Challenge Liveness Verification**

-Prevents spoofing attacks such as video replay or photo-based impersonation during identity registration.

-Real-Time SOC Dashboard

-Displays trust score trends, behavioral metrics, deviations, and risk classification through a monitoring interface.

**Multi-Dashboard Architecture**

-Each user profile launches an isolated monitoring environment, simulating enterprise SOC workflows.

-False Positive Reduction

-Designed to demonstrate that effective security systems protect legitimate users from incorrect suspicion.

## System Architecture

-The platform follows a modular architecture separating frontend visualization from backend behavioral simulation.

Frontend (SOC Dashboard)
        |
        | REST API
        |
Backend Behavioral Engine
        |
Trust Score Calculator
        |
Baseline Profiles
        |
Simulation Engine

# Technology Stack
Frontend

HTML5

CSS3

Vanilla JavaScript

Chart.js

Backend

Python

FastAPI

Async background tasks

JSON-based API responses

Security Concepts Modeled

Zero Trust

Behavioral Drift Detection

Role-Based Risk Modeling

Trust Scoring

Liveness Verification

Project Structure
insider-threat-system/
│
├── backend/
│   ├── main.py
│   ├── trust_engine.py
│   ├── behavior_simulator.py
│   ├── baseline_profiles.py
│   └── config.py
│
├── frontend/
│   ├── index.html
│   ├── dashboard.html
│   ├── styles.css
│   └── scripts.js
│
├── liveness-verification/
│   ├── main.py
│   ├── blink_detector.py
│   ├── head_pose_estimator.py
│   └── face_storage.pkl
│
└── README.md

Behavioral Modeling Approach
Personalized Baselines

Each user is assigned a behavioral profile defining what constitutes normal activity.

Example:

Trusted Employee

predictable login patterns

moderate file access

minimal policy violations

Elevated Risk Employee

increasing irregularity

repeated deviations

behavioral drift

Visitor

limited system interaction

restricted access

high trust stability

Trust Score Methodology

Trust is dynamically evaluated on a scale from 0 to 100.

Risk Classification
Trust Score	Risk Level
85 – 100	Low Risk
70 – 84	Guarded
50 – 69	High Risk
Below 50	Critical
Example Penalties

multiple failed logins

unusual file access spikes

after-hours activity

unauthorized USB usage

location anomalies

The system detects behavioral drift when multiple deviations occur within a short window.

Frontend Overview

The frontend simulates a SOC monitoring dashboard capable of displaying:

trust score visualization

behavioral metrics

risk classification

deviation logs

trust history charts

baseline comparisons

Each registered user is clickable and redirects to their respective monitoring instance.

This models isolated security environments commonly used in enterprise SOC platforms.

Backend Overview

The backend exposes a REST endpoint:

GET /api/status


This endpoint returns structured behavioral data including:

user identity

trust score

risk level

behavioral indicators

deviation analysis

drift detection

A background simulation updates values at fixed intervals to mimic real-time monitoring.

Installation Guide
Clone Repository
git clone https://github.com/your-repo/insider-threat-system.git
cd insider-threat-system

Backend Setup

Create virtual environment:

python -m venv venv


Activate:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate


Install dependencies:

pip install fastapi uvicorn


Start server:

uvicorn main:app --reload


Backend runs on:

http://localhost:8000

Frontend Setup

Simply open:

index.html


or serve via:

python -m http.server

Running Multiple Monitoring Instances

Each user dashboard can operate on separate ports to simulate enterprise monitoring environments.

Example mapping:

Employee Dashboard → localhost:8081

Visitor Dashboard → localhost:8000

Risk Monitoring → localhost:5500

This design demonstrates segmented monitoring.

Liveness Verification Module

The system includes a controlled face registration process that requires a randomized live action.

Purpose:

prevent replay attacks

block photo spoofing

ensure real human presence

Embeddings are stored instead of raw images to support privacy-aware design.

Security Principles Demonstrated

Trust must be continuously evaluated

Behavior should be analyzed relative to personal baselines

Security systems must minimize operational friction

Effective monitoring prevents threats without disrupting legitimate users

Future Enhancements

Machine learning-based anomaly detection

SIEM integration

CCTV pipeline connectivity

Access control hardware integration

Multi-tenant SOC support

Alert orchestration

Audit trail storage

Use Cases

Corporate office security

Research labs

Financial institutions

Government facilities

Smart campuses
