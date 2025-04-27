import cv2
import numpy as np

class Enhancer:
    def __init__(self, method, background_enhancement, upscale):
        self.method = method
        self.background_enhancement = background_enhancement
        self.upscale = upscale

    def enhance(self, image_array):
        # Convert the image array to a format OpenCV can understand
        image = cv2.imdecode(np.frombuffer(image_array, np.uint8), cv2.IMREAD_COLOR)

        # Apply Gaussian blur removal to reduce blurriness
        kernel_size = (5, 5)
        sigma = 1.0
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        unsharp_mask = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        
        # Apply sharpening filter to enhance details
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(unsharp_mask, -1, kernel_sharpen)

        # Convert the image back to bytes
        _, buffer = cv2.imencode('.jpg', sharpened)
        enhanced_image_bytes = buffer.tobytes()
        return enhanced_image_bytes
