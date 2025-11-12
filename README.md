# KCF Object Tracker

A high-performance object tracking application using Kernelized Correlation Filter (KCF) algorithm with real-time video tracking capabilities. This implementation features fullscreen display, high-quality video rendering, and comprehensive tracking metrics.

---

## 🎯 Features

- **KCF Algorithm**: Advanced Kernelized Correlation Filter for robust object tracking
- **Fullscreen Display**: Automatic fullscreen mode with aspect ratio preservation
- **High-Quality Rendering**: LANCZOS4 interpolation for crisp video quality
- **Real-Time Metrics**: 
  - PSR (Peak-to-Sidelobe Ratio) - tracking confidence
  - NCC (Normalized Cross-Correlation) - appearance similarity
  - FPS (Frames Per Second) - performance monitoring
- **Green Tracking Box**: Visual bounding box around tracked objects
- **Multiple Input Sources**: Support for video files and webcam
- **Adaptive Scaling**: Automatic video scaling to fit screen while maintaining aspect ratio

## 📋 Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```
opencv-python>=4.5.0
numpy>=1.19.0
scipy>=1.5.0
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Tracker.git
   cd Tracker
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python numpy scipy
   ```
   
   Or using requirements.txt (create one if needed):
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Track objects in a video file:
```bash
python main.py path/to/your/video.mp4
```

### Track objects using webcam:
```bash
python main.py
```

### Controls:
- **ROI Selection**: When the video starts, select the Region of Interest (ROI) by dragging a rectangle around the object you want to track
- **ESC Key**: Press ESC to exit the application

## 📖 How It Works

1. **Initialization**: 
   - Load video or start webcam
   - Select ROI (Region of Interest) on the first frame
   - Initialize KCF tracker with the selected region

2. **Tracking**:
   - The tracker uses Gaussian kernel correlation in frequency domain
   - Updates position based on correlation response
   - Computes PSR and appearance similarity for tracking quality

3. **Display**:
   - Video is scaled to fit screen with letterboxing
   - Green bounding box shows tracked object
   - Real-time metrics displayed in top-left corner

## 🔧 Technical Details

### Algorithm
- **KCF Tracker**: Kernelized Correlation Filter implementation
- **Gaussian Kernel**: Used for correlation computation
- **FFT-based**: Fast correlation using Fast Fourier Transform
- **Appearance Model**: Combines NCC and color histogram similarity

### Performance
- Real-time tracking at 30+ FPS (depending on video resolution)
- Efficient FFT-based correlation computation
- Adaptive model update with interpolation factor

## 📊 Tracking Metrics

- **PSR (Peak-to-Sidelobe Ratio)**: Measures tracking confidence. Higher values indicate better tracking.
- **NCC (Normalized Cross-Correlation)**: Measures appearance similarity between current frame and template. Range: 0-1, higher is better.
- **FPS (Frames Per Second)**: Real-time performance indicator.

## 🎨 Display Features

- **Fullscreen Mode**: Automatic fullscreen with proper aspect ratio
- **Letterboxing**: Black borders added when needed to maintain aspect ratio
- **High-Quality Scaling**: LANCZOS4 interpolation for smooth, sharp video
- **White Text Overlay**: Clean, readable metrics display

## 📝 Code Structure

```
main.py
├── Helper Functions
│   ├── hann_safe() - Safe Hann window generation
│   ├── get_subwindow() - Extract subwindow from image
│   ├── gaussian_correlation() - KCF kernel correlation
│   ├── compute_psr() - Peak-to-Sidelobe Ratio calculation
│   ├── get_color_hist() - Color histogram computation
│   ├── appearance_similarity() - NCC + histogram similarity
│   └── resize_to_fit_screen() - Screen fitting with letterboxing
├── KCFTracker Class
│   ├── __init__() - Initialize tracker with ROI
│   └── update() - Update tracking for new frame
└── Main Function
    └── kcf_tracker_roi() - Main tracking loop
```

## 🐛 Troubleshooting

### Video not loading?
- Check if the video file path is correct
- Ensure the video format is supported by OpenCV (MP4, AVI, MOV, etc.)

### Webcam not working?
- Make sure your webcam is connected and not being used by another application
- Try specifying the camera index: `python main.py 0` (0 is default)

### Poor tracking performance?
- Ensure good lighting conditions
- Select a clear, distinctive ROI
- Avoid fast camera movements or occlusions

### Import errors?
- Make sure all dependencies are installed: `pip install opencv-python numpy scipy`
- Check Python version compatibility

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- KCF algorithm based on the work by João F. Henriques et al.
- OpenCV community for excellent computer vision tools

---

⭐ If you find this project helpful, please consider giving it a star!

