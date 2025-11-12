import cv2
import numpy as np
import time
from scipy.signal.windows import hann
from scipy.spatial.distance import cosine

# ------------------- Helper Functions -------------------

def hann_safe(N):
    return hann(N, sym=False) if N > 1 else np.ones(1)

def get_subwindow(im, pos, sz):
    """Crop subwindow from image, replicating border if needed."""
    h, w = im.shape[:2]
    ys = np.floor(pos[0]) + np.arange(sz[0]) - sz[0]//2
    xs = np.floor(pos[1]) + np.arange(sz[1]) - sz[1]//2
    ys = np.clip(ys, 0, h-1).astype(int)
    xs = np.clip(xs, 0, w-1).astype(int)
    return im[np.ix_(ys, xs)]

def gaussian_correlation(x1, x2, sigma):
    """Gaussian kernel correlation for KCF."""
    xf = np.fft.fft2(x1)
    yf = np.fft.fft2(x2)
    xx1 = np.sum(x1**2)
    xx2 = np.sum(x2**2)
    xy = np.real(np.fft.ifft2(xf * np.conj(yf)))
    d = np.maximum(0, xx1 + xx2 - 2*xy)/x1.size
    return np.exp(-1/sigma**2 * d)

def compute_psr(resp, excl=11):
    """Compute Peak-to-Sidelobe Ratio (PSR)."""
    peak_idx = np.unravel_index(np.argmax(resp), resp.shape)
    peak_val = resp[peak_idx]
    m, n = resp.shape
    half = excl//2
    mask = np.ones(resp.shape, bool)
    r1 = max(0, peak_idx[0]-half)
    r2 = min(m, peak_idx[0]+half+1)
    c1 = max(0, peak_idx[1]-half)
    c2 = min(n, peak_idx[1]+half+1)
    mask[r1:r2, c1:c2] = False
    side = resp[mask]
    mu, sigma = np.mean(side), np.std(side)
    return (peak_val - mu)/sigma if sigma>0 else float('inf')

def get_color_hist(img, bins=16):
    """Compute normalized HSV histogram."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [bins]*3, [0,180,0,256,0,256])
    hist = hist.flatten()
    hist /= (np.sum(hist)+1e-6)
    return hist

def appearance_similarity(img1, img2):
    """Combine NCC (grayscale) + histogram similarity."""
    # Resize images to same size for NCC computation
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        img1 = cv2.resize(img1, (w2, h2))
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)/255.0
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)/255.0
    ncc = np.sum((gray1-np.mean(gray1))*(gray2-np.mean(gray2))) / (
        np.sqrt(np.sum((gray1-np.mean(gray1))**2)*np.sum((gray2-np.mean(gray2))**2))+1e-6)
    hist1 = get_color_hist(img1)
    hist2 = get_color_hist(img2)
    hist_sim = 1 - cosine(hist1, hist2)
    return 0.4*hist_sim + 0.6*ncc

def resize_to_fit_screen(frame, screen_width, screen_height):
    """Resize frame to fit screen while maintaining aspect ratio, with letterboxing."""
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate scaling factor to fit screen
    scale_w = screen_width / frame_width
    scale_h = screen_height / frame_height
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)
    
    # Resize frame with high-quality interpolation for better video quality
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Create black background
    display_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Calculate position to center the video
    y_offset = (screen_height - new_height) // 2
    x_offset = (screen_width - new_width) // 2
    
    # Place resized frame in center
    display_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return display_frame, scale, (x_offset, y_offset)

# ------------------- KCF Tracker Class -------------------

class KCFTracker:
    def __init__(self, frame, roi, kernel_sigma=0.3, lambda_=1e-4, interp_factor=0.04, padding=2.5):
        self.pos = [roi[1]+roi[3]/2, roi[0]+roi[2]/2]
        self.target_sz = [roi[3], roi[2]]
        self.padding = padding
        self.lambda_ = lambda_
        self.interp_factor = interp_factor
        self.kernel_sigma = kernel_sigma

        self.window_sz = np.floor(np.array(self.target_sz)*(1+self.padding)).astype(int)
        self.cos_window = np.outer(hann_safe(self.window_sz[0]), hann_safe(self.window_sz[1]))
        
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.0
        self.x = get_subwindow(self.gray, self.pos, self.window_sz) * self.cos_window
        kxx = gaussian_correlation(self.x, self.x, self.kernel_sigma)
        output_sigma = np.sqrt(np.prod(self.target_sz))*1/16
        y = np.exp(-0.5*((np.arange(self.window_sz[0])[:,None]-self.window_sz[0]//2)**2 +
                         (np.arange(self.window_sz[1])-self.window_sz[1]//2)**2)/(output_sigma**2))
        self.yf = np.fft.fft2(y)
        self.alphaf = self.yf/(np.fft.fft2(kxx)+self.lambda_)
        self.template = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]].copy()

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.0
        z = get_subwindow(gray, self.pos, self.window_sz) * self.cos_window
        k = gaussian_correlation(z, self.x, self.kernel_sigma)
        resp = np.real(np.fft.ifft2(self.alphaf * np.fft.fft2(k)))
        r, c = np.unravel_index(np.argmax(resp), resp.shape)
        self.pos[0] += r - self.window_sz[0]//2
        self.pos[1] += c - self.window_sz[1]//2
        psr = compute_psr(resp)

        # Update model
        new_x = get_subwindow(gray, self.pos, self.window_sz) * self.cos_window
        kxx = gaussian_correlation(new_x, new_x, self.kernel_sigma)
        alphaf_new = self.yf/(np.fft.fft2(kxx)+self.lambda_)
        self.x = (1-self.interp_factor)*self.x + self.interp_factor*new_x
        self.alphaf = (1-self.interp_factor)*self.alphaf + self.interp_factor*alphaf_new

        # Appearance similarity
        x1, y1 = int(self.pos[1]-self.target_sz[1]/2), int(self.pos[0]-self.target_sz[0]/2)
        x2, y2 = x1+self.target_sz[1], y1+self.target_sz[0]
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(frame.shape[1],x2), min(frame.shape[0],y2)
        crop = frame[y1:y2, x1:x2]
        app_sim = appearance_similarity(crop, self.template)

        return (int(x1), int(y1), int(x2-x1), int(y2-y1)), psr, app_sim

# ------------------- Main Function -------------------

def kcf_tracker_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Cannot read video")
        return

    # Get screen resolution
    screen_width = 1920  # Default, will be updated
    screen_height = 1080  # Default, will be updated
    try:
        # Try to get actual screen resolution
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except:
        # Fallback to common resolutions
        pass

    # Select ROI on original frame
    roi = cv2.selectROI("Select ROI", frame, False, False)
    cv2.destroyWindow("Select ROI")

    # Create window and set to fullscreen
    cv2.namedWindow("KCF Tracker", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("KCF Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    tracker = KCFTracker(frame, roi)
    
    # Get video properties for FPS and frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default FPS if not available
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # FPS calculation
    frame_count = 0
    prev_time = time.time()
    
    # Get initial scale and offset for first frame
    display_frame, scale, (x_offset, y_offset) = resize_to_fit_screen(frame, screen_width, screen_height)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()
        elapsed = current_time - prev_time
        prev_time = current_time
        
        # Calculate FPS (average over last second)
        if elapsed > 0:
            current_fps = 1.0 / elapsed
        else:
            current_fps = fps

        bbox, psr, app_sim = tracker.update(frame)
        x, y, w, h = bbox

        # Resize frame to fit screen
        display_frame, scale, (x_offset, y_offset) = resize_to_fit_screen(frame, screen_width, screen_height)
        
        # Adjust bounding box coordinates to match scaled and offset frame
        scaled_x = int(x * scale + x_offset)
        scaled_y = int(y * scale + y_offset)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)

        # Always use green color for border
        color = (0, 255, 0)

        # Draw rectangle on display frame
        cv2.rectangle(display_frame, (scaled_x, scaled_y), (scaled_x+scaled_w, scaled_y+scaled_h), color, 2)
        
        # Draw text on display frame (adjust position based on scale)
        text_scale = max(0.5, scale * 0.5)
        text_thickness = max(1, int(scale))
        
        # White color for text
        text_color = (255, 255, 255)
        
        # Display all metrics on one line: PSR NCC FPS
        metrics_text = f"PSR:{psr:.2f} NCC:{app_sim:.2f} FPS:{current_fps:.2f}"
        cv2.putText(display_frame, metrics_text, 
                    (int(10*scale), int(30*scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)

        cv2.imshow("KCF Tracker", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key==27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------- Run Example -------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default to webcam if no argument provided
        video_path = 0  # Use webcam
        print("No video path provided. Using webcam (press ESC to exit).")
    
    kcf_tracker_roi(video_path)
