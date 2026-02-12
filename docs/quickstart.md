# Quickstart

## Prerequisites
- Python 3.11+
- Node.js 20+
- Redis 7+

## 1) Install dependencies
```bash
cd /Users/rajattiwari/microgpt
python3 -m pip install -r requirements.txt
npm install
```

## 2) Configure environment
```bash
cp .env.example .env
```

## 3) Start Redis
```bash
redis-server
```

## 4) Start API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 5) Start worker
```bash
python -m worker.main
```

## 6) Start frontend
```bash
npm run frontend:dev
```

Open `http://localhost:5173`.
