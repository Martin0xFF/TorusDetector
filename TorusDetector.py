import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


def percent_to_u8(percent):
    return int((float(percent) / 100.0) * 255)


@dataclass
class TorusDetectorOptions:
    """Class for specifying options for TorusDetector."""

    # Note that HSV will have the following ranges:
    # Hue - will be between [0, 180]
    # Saturation - will be betwee [0, 100]
    # Value - will be betwee [0, 100]

    hsv_start: Tuple[int, int, int] = (0, percent_to_u8(35), percent_to_u8(85))
    hsv_end: Tuple[int, int, int] = (15, percent_to_u8(100), percent_to_u8(100))

    # Must be an odd number, larger kernel -> more blurring
    blur_kernel_size: int = 7


class TorusDetector:
    def __init__(self, config: Optional[TorusDetectorOptions] = None):
        if config is not None:
            self.config = config
        else:
            self.config = TorusDetectorOptions()

    def generate_color_mask(self, image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.hsv_start, self.config.hsv_end)
        return mask

    def __call__(self, input_image):
        mask = self.generate_color_mask(input_image)
        result = cv2.bitwise_and(input_image, input_image, mask=mask)
        result = cv2.medianBlur(
            cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), self.config.blur_kernel_size
        )

        # TODO: Add Support for configuring these values.
        circles = cv2.HoughCircles(
            result,
            cv2.HOUGH_GRADIENT,
            1,
            result.shape[0] / 8,
            param1=100,
            param2=30,
            minRadius=0,
            maxRadius=0,
        )
        return circles


if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    td = TorusDetector()
    while True:
        ret, image = vid.read()
        circles = td(image)

        # Visualize the Circles.
        blob_color = (255, 0, 0)
        blobs = image[:]
        count = 0
        # If circles are present, draw on input image.
        if circles is not None:
            count = len(circles)
            # Convert to unsigned int since pixel locations are whole numbers.
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv2.circle(blobs, center, 1, blob_color, 3)
                radius = i[2]
                cv2.circle(blobs, center, radius, blob_color, 3)

        # Write debug text to original image.
        text = "Number of Detected Circles: " + str(count)
        cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, blob_color, 2)

        # Show image with circles.
        cv2.imshow("Filtering Circular Blobs Only", blobs)
        if cv2.waitKey(10) == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()
