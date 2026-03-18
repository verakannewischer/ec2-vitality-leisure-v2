# Vitality Leisure Park — EC2 Deployment (v2)

This repository contains the Flask/HTML version of the Vitality Leisure Park Capacity Intelligence Viewer, deployed on AWS EC2.

## What it does

A web application combining a Gradient Boosting visitor forecast model with a RAG-powered Wellness Coach chatbot. Built with Flask and HTML/CSS/JavaScript for a production-ready frontend.

**Pages:**
- **Landing page** — overview with live weather and navigation
- **Manager Dashboard** — 7-day forecast, capacity tracking, heatmap, year-over-year trends
- **Plan My Visit** — preference-based day recommendation with crowd and weather filtering
- **Wellness Coach** — conversational chatbot using Retrieval-Augmented Generation (RAG) over the real restaurant menu and fitness schedule PDFs

## Tech stack
- Flask + Gunicorn + Nginx on AWS EC2 (Ubuntu 24.04)
- Gradient Boosting Regressor (scikit-learn)
- Cohere API — `embed-english-v3.0` for RAG embeddings, `command-a-03-2025` for chat
- Open-Meteo API for live and historical weather
- NRW public and school holidays via Python `holidays` library

## Live deployment
http://13.53.214.124:5000

## Setup
See `DEPLOY.md` for full EC2 deployment instructions.

## Note
This is a design and deployment iteration of the Streamlit prototype (Assignment 2). The Streamlit version is available at https://vitality-leisure-v2.streamlit.app
