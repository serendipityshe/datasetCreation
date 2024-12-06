import numpy as np
from skimage.io import imread
import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class imageCropMerge:
    def __init__(self, img_path: str, patch_size: tuple, stride: int):
        """
        Initialize the imageCropMerge class.

        Parameters:
        - img_path (str): Path to the input image.
        - patch_size (tuple): Size of each image patch (height, width).
        - stride (int): Stride for cropping patches.
        """
        self.img_path = img_path
        self.patch_size = patch_size
        self.stride = stride
        self.image = None

    def _image_to_blocks(self, image: np.ndarray) -> np.ndarray:
        """
        Convert an image into blocks.

        Parameters:
        - image (np.ndarray): Input image.

        Returns:
        - np.ndarray: Blocks of the image.
        """
        if len(image.shape) == 2:
            # Grayscale image
            imhigh, imwidth = image.shape
        elif len(image.shape) == 3:
            # RGB image
            imhigh, imwidth, imch = image.shape
        else:
            raise ValueError("Unsupported image dimensions")

        range_y = np.arange(0, imhigh - self.patch_size[0], self.stride)
        if range_y[-1] != imhigh - self.patch_size[0]:
            range_y = np.append(range_y, imhigh - self.patch_size[0])

        range_x = np.arange(0, imwidth - self.patch_size[1], self.stride)
        if range_x[-1] != imwidth - self.patch_size[1]:
            range_x = np.append(range_x, imwidth - self.patch_size[1])

        sz = len(range_y) * len(range_x)  # Number of blocks
        if len(image.shape) == 2:
            # Initialize grayscale image blocks
            res = np.zeros((sz, self.patch_size[0], self.patch_size[1]))
        elif len(image.shape) == 3:
            # Initialize RGB image blocks
            res = np.zeros((sz, self.patch_size[0], self.patch_size[1], imch))

        index = 0
        for y in range_y:
            for x in range_x:
                patch = image[y:y + self.patch_size[0], x:x + self.patch_size[1]]
                res[index] = patch
                index += 1

        return res

    def crop_image(self, output_dir: str):
        """
        Crop the image into blocks and save them to the specified directory.

        Parameters:
        - output_dir (str): Directory to save the cropped image blocks.
        """
        # Read and normalize the image
        self.image = imread(self.img_path) / 255.0

        # Print image shape
        print(f"Image shape: {self.image.shape}")

        # Convert image to blocks
        image_blocks = self._image_to_blocks(self.image)

        # Print image blocks shape
        print(f"Image blocks shape: {image_blocks.shape}")

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save image blocks to the specified directory
        for i, block in enumerate(image_blocks):
            # If the image is RGB, reverse the channel order to match OpenCV's BGR order
            if block.ndim == 3:
                block = block[:, :, ::-1]

            # Save the image block as a PNG file
            cv2.imwrite(os.path.join(output_dir, f"{i}.png"), (block * 255).astype(np.uint8))

    def merge_patches(self, cache_dir: str, output_path: str = None):
        """
        Merge the cropped image patches back into a complete image.

        Parameters:
        - cache_dir (str): Directory containing the cropped image patches.
        - output_path (str, optional): Path to save the merged image. If not specified,
                                       it will be saved in the same directory as the input image.
        """
        # Read the original image
        image = imread(self.img_path)
        imhigh, imwidth, imch = image.shape

        # Initialize the output image and weight matrix
        res = np.zeros_like(image, dtype=np.float32)
        weights = np.zeros((imhigh, imwidth), dtype=np.float32)

        # Calculate the coordinate ranges for cropping
        range_y = np.arange(0, imhigh - self.patch_size[0] + 1, self.stride)
        if range_y[-1] != imhigh - self.patch_size[0]:
            range_y = np.append(range_y, imhigh - self.patch_size[0])

        range_x = np.arange(0, imwidth - self.patch_size[1] + 1, self.stride)
        if range_x[-1] != imwidth - self.patch_size[1]:
            range_x = np.append(range_x, imwidth - self.patch_size[1])

        # Define a function to process a single image patch
        def process_patch(index):
            image_patch_path = os.path.join(cache_dir, f"{index}.png")
            image_patch = cv2.imread(image_patch_path)

            if image_patch is None:
                print(f"Warning: Unable to load image patch at {image_patch_path}")
                return

            if len(image_patch.shape) == 2:  # If grayscale, convert to RGB
                image_patch = cv2.cvtColor(image_patch, cv2.COLOR_GRAY2RGB)

            # Calculate the position of the patch in the full image
            y_start = index // (imwidth // self.stride) * self.stride
            x_start = (index % (imwidth // self.stride)) * self.stride

            # Accumulate the image patch and weights
            image_patch_gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
            res[y_start:y_start + self.patch_size[0], x_start:x_start + self.patch_size[1]] += image_patch_gray.astype(np.float32)
            weights[y_start:y_start + self.patch_size[0], x_start:x_start + self.patch_size[1]] += 1

        # Use a thread pool to process image patches in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_patch, i) for i in range(len(os.listdir(cache_dir)))]
            for future in as_completed(futures):
                future.result()  # Check for exceptions

        # Perform weighted average in overlapping regions
        res = res / (weights[:, :, np.newaxis] + 1e-8)  # Add a small epsilon to prevent division by zero

        # Convert to uint8 type
        res = np.uint8(np.clip(res, 0, 255))

        # If output path is not specified, default to the same directory as the input image
        if output_path is None:
            output_path = os.path.join(os.path.dirname(self.img_path), "whole_result_fused.tif")

        # Save the result, noting that channel order is BGR for OpenCV
        cv2.imwrite(output_path, res[:, :, ::-1])

        # Print some statistics
        print(f"Saved merged image to {output_path}")
        print(f"Min value in res: {np.min(res)}, Max value in res: {np.max(res)}, Mean value in res: {np.mean(res)}")

# Example usage
if __name__ == '__main__':
    img_crop_merge = imageCropMerge(img_path="E:/xunlei/TDOM/0802-1028/tif1.tif",
                                    patch_size=(256, 256),
                                    stride=160)
    # img_crop_merge.crop_image(output_dir="/home/ljh/Desktop/test/data/image/cache/")
    img_crop_merge.merge_patches(cache_dir="D:/PROJECT/AI-Project/ChangeDetect/open-cd/OUTPUT_TDOM_CHANGEFORMER_256s50_dsifn-nanning/vis",
                                 output_path="E:/xunlei/TDOM/0802-1028/merged_output1.tif")