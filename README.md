# HTRex Backend Service

A GPU-accelerated Handwriting Recognition service using PaddleOCR and Gemini AI for text correction and summarization.

## Features
- Handwriting Recognition using PaddleOCR
- Text error correction using Gemini AI
- Text summarization
- GPU acceleration support
- REST API endpoints

## Tech Stack
- Python 3.10
- Flask
- PaddleOCR
- PyTorch
- Google Gemini AI
- CUDA 11.8

## API Endpoints

### POST /process_image
Process an image containing handwritten text.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - image: Image file (PNG, JPG, JPEG)

**Response:**
```json
{
    "text": "Original recognized text",
    "corrected_text": "Corrected text",
    "summary": "AI-generated summary",
    "visualization": "Base64 encoded visualization image"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
    "status": "healthy"
}
```

## Deployment

### Prerequisites
1. Render.com account
2. Google Cloud API key for Gemini AI
3. Git

### Environment Variables
- `PORT`: 8080 (default)
- `GOOGLE_API_KEY`: Your Google API key for Gemini AI

### Deployment Steps
1. Fork/clone this repository
2. Create a new Web Service on Render
3. Choose "GPU Standard" instance
4. Set environment variables
5. Deploy

## Local Development
1. Install Docker
2. Build: `docker build -t htrex-backend .`
3. Run: `docker run -p 8080:8080 -e GOOGLE_API_KEY=your_key htrex-backend`

## License
MIT 