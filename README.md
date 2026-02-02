# Smart Parking Access Control System

A production-ready smart parking surveillance system that uses ML-based number plate detection and OCR to authorize vehicle access against a whitelist database.

## Features

- 🚗 **Plate Detection**: YOLO-based number plate detection with pre-trained weights
- 🔤 **OCR Extraction**: EasyOCR with image preprocessing for optimal accuracy
- ✅ **Whitelist Verification**: Real-time plate validation against database
- 🚨 **Alert System**: Automatic alerts for unauthorized/unknown vehicles
- 👮 **Guard Dashboard**: React app for viewing and acknowledging alerts
- 📊 **Access Logging**: Complete audit trail of all verification attempts

## Architecture

```
backend/
├── app/
│   ├── core/           # Config, logging, security
│   ├── domain/         # Business rules & models
│   ├── application/    # Use cases
│   ├── infrastructure/ # DB, ML, storage
│   ├── api/            # FastAPI routes
│   └── tests/          # Test suite
```

## Quick Start

### Prerequisites

- Python 3.10+
- MySQL 8.0+
- Node.js 18+ (for guard app)

### 1. Clone & Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy and edit environment file
cp .env.example .env

# Edit .env with your settings:
# - DATABASE_URL: Your MySQL connection string
# - SECRET_KEY: Generate a secure key
# - ML_MODEL_PATH: Path to plate detection weights
```

### 3. Setup Database

```bash
# Create database in MySQL
mysql -u root -p -e "CREATE DATABASE parking_db;"

# Tables are created automatically on first run
```

### 4. Add Model Weights (Optional)

Place your pre-trained plate detection model at `./models/plate_detector.pt`.

If no weights are provided, a mock detector is used for development.

### 5. Run the Server

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### 6. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready
```

## API Documentation

### Interactive Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints

#### POST /api/v1/plates/verify

Verify a vehicle plate from camera image.

**Request:**
- `image`: Multipart file (JPEG/PNG)
- `camera_id`: Camera identifier
- `timestamp`: Optional capture time

**Headers:**
- `X-API-Key`: Your API key

**Response:**
```json
{
  "plate_number": "MH12AB1234",
  "status": "authorized",
  "action": "allow",
  "confidence": 0.92
}
```

#### GET /api/v1/alerts

List pending alerts (requires Basic Auth).

#### POST /api/v1/alerts/{id}/acknowledge

Acknowledge an alert (requires Basic Auth).

## Testing

```bash
# Run all tests with coverage
pytest --cov=app --cov-report=term-missing

# Run specific test categories
pytest app/tests/unit/ -v
pytest app/tests/integration/ -v
pytest app/tests/e2e/ -v

# Generate HTML coverage report
pytest --cov=app --cov-report=html
```

## Model Loading

The system loads plate detection weights at startup:

1. Checks `ML_MODEL_PATH` from environment
2. If weights exist, loads YOLO model with warm-up inference
3. If weights missing, uses mock detector (development mode)

**To use your own model:**
- Provide YOLO-compatible `.pt` weights
- Update `ML_MODEL_PATH` in `.env`

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | MySQL connection string | Required |
| `SECRET_KEY` | JWT signing key | Required |
| `API_KEY` | Service authentication | Required |
| `ML_MODEL_PATH` | Path to model weights | `./models/plate_detector.pt` |
| `OCR_CONFIDENCE_THRESHOLD` | Min OCR confidence | `0.70` |
| `DETECTOR_CONFIDENCE_THRESHOLD` | Min detector confidence | `0.50` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Development

```bash
# Format code
black app/

# Lint
ruff check app/

# Type check
mypy app/
```

## License

MIT
