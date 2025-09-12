# Flask Face Recognition Attendance System

A modern web-based face recognition attendance system built with Flask, JavaScript, and advanced AI models.

## Features

- **Web-based Interface**: Modern, responsive UI built with Bootstrap
- **Real-time Face Recognition**: Live camera feed with face detection and recognition
- **Multiple AI Models**: Support for FaceNet, ArcFace, ResNet50, and MobileNetV2
- **Quality Control**: Automatic quality checks for face enrollment
- **Performance Monitoring**: Real-time FPS and processing time tracking
- **Configurable Settings**: Adjustable thresholds and similarity metrics
- **Auto-attendance**: Automatic attendance marking with duplicate prevention
- **CSV Export**: Attendance records saved to CSV files

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd attendance_proto
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements_flask.txt
   ```

4. **Optional: Install advanced models for better accuracy**:
   ```bash
   # For MTCNN face detection
   pip install mtcnn

   # For FaceNet model
   pip install facenet-pytorch

   # For ArcFace model (requires additional setup)
   pip install insightface
   ```

## Usage

1. **Start the Flask application**:
   ```bash
   python flask_app.py
   ```

2. **Open your web browser** and go to:
   ```
   http://localhost:5000
   ```

3. **System Setup**:
   - Go to Configuration page
   - Select your preferred AI model
   - Adjust similarity threshold if needed
   - Click "Initialize Model"

4. **Enroll Students**:
   - Go to Enrollment page
   - Enter student ID
   - Start camera
   - Position face clearly in frame
   - Click "Capture & Enroll"

5. **Mark Attendance**:
   - Go to Attendance page
   - Start recognition
   - Students will be automatically detected
   - Click "Mark Attendance" or enable auto-marking

## File Structure

```
attendance_proto/
├── flask_app.py              # Main Flask application
├── templates/                # HTML templates
│   ├── base.html            # Base template with common elements
│   ├── index.html           # Home page
│   ├── enrollment.html      # Student enrollment page
│   ├── attendance.html      # Attendance marking page
│   └── config.html          # Configuration page
├── data/                    # Data storage directory
│   ├── embeddings.pkl       # Face embeddings database
│   ├── attendance.csv       # Attendance records
│   └── config.pkl           # System configuration
├── requirements_flask.txt   # Python dependencies
└── README_FLASK.md         # This file
```

## API Endpoints

- `POST /api/initialize_model` - Initialize face recognition model
- `POST /api/enroll` - Enroll a new student
- `POST /api/recognize` - Recognize faces in image
- `POST /api/mark_attendance` - Mark attendance for students
- `POST /api/save_config` - Save system configuration
- `GET /api/students` - Get list of enrolled students

## Model Comparison

| Model | Accuracy | Speed | Embedding Size | Requirements |
|-------|----------|-------|----------------|--------------|
| FaceNet | High | Fast | 512D | facenet-pytorch |
| ArcFace | Very High | Fast | 512D | insightface |
| ResNet50 | Medium | Slow | 2048D | tensorflow |
| MobileNetV2 | Medium | Very Fast | 1280D | tensorflow |

## Configuration Options

### Recognition Settings
- **Similarity Threshold**: Controls matching strictness (0.5-0.95)
- **Similarity Metric**: Cosine similarity or Euclidean distance
- **Recognition Interval**: How often to process frames
- **Auto-marking**: Automatically mark attendance when faces are detected

### Advanced Settings
- **Face Detection**: Minimum face size and confidence thresholds
- **Performance**: Frame skipping for better performance
- **Quality Control**: Blur detection and face size validation

## Troubleshooting

### Camera Issues
- **Permission denied**: Grant camera permissions in browser
- **Camera not found**: Check if camera is connected and not used by other apps
- **Poor quality**: Ensure good lighting and clear face visibility

### Model Issues
- **Model loading failed**: Check if required packages are installed
- **Low accuracy**: Try different models or adjust thresholds
- **Slow performance**: Use MobileNetV2 or enable frame skipping

### Installation Issues
- **Package conflicts**: Use virtual environment
- **Missing dependencies**: Install optional packages for advanced features
- **Platform issues**: Some models may not work on all platforms

## Development

### Adding New Models
1. Add model configuration to `MODEL_CONFIGS`
2. Implement loading logic in `FaceRecognitionModel.load_model()`
3. Add preprocessing logic in `preprocess_face()`
4. Update embedding extraction in `get_embedding()`

### Customizing UI
- Templates use Bootstrap 5 for styling
- JavaScript functions are in `base.html`
- Each page has its own JavaScript in `extra_js` block

### Database Integration
- Current version uses pickle files for simplicity
- Can be extended to use SQLite, PostgreSQL, or MongoDB
- Implement database models for students and attendance records

## Security Considerations

- Add user authentication for production use
- Implement CSRF protection
- Use HTTPS for secure camera access
- Add rate limiting for API endpoints
- Sanitize user inputs
- Consider data privacy regulations (GDPR, etc.)

## Performance Optimization

- Use frame skipping to reduce processing load
- Implement caching for frequently accessed data
- Consider using GPU acceleration for models
- Optimize image preprocessing pipeline
- Use WebSockets for real-time updates

## License

This project is provided as-is for educational and research purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration options
3. Check the browser console for JavaScript errors
4. Ensure all dependencies are properly installed