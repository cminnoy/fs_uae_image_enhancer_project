# cache.py

import os
import warnings
from diskcache import Cache

DEFAULT_TRAIN_CACHE_FILE = '.train_scan_cache'
DEFAULT_TEST_CACHE_FILE = '.test_scan_cache'

class ScanCache:
    """
    Manages caching of ground truth image scan results for individual crop locations
    using diskcache for efficient disk-based caching.
    """
    def __init__(self, cache_path=None, verbose=0):
        """
        Initializes the ScanCache.

        Args:
            cache_path (str, optional): The directory path where the cache data will be stored.
                                        If None, caching will be in-memory only.
            verbose (int): Verbosity level (0=quiet, 1=progress, 2=debug).
        """
        self.cache_path = cache_path or DEFAULT_TRAIN_CACHE_FILE
        self.verbose = verbose
        self.cache = Cache(self.cache_path)  # Initialize diskcache

        if self.verbose >= 2:
            print(f"Cache initialized at {self.cache_path}")

    def get_image_cache(self, img_path):
        """
        Retrieves cached data for a specific image.

        Args:
            img_path (str): Path to the image file.

        Returns:
            dict or None: The cached data for the image, or None if not found.
        """
        return self.cache.get(img_path)

    def update_image_cache(self, img_path, data):
        """
        Updates the cache for a specific image.

        Args:
            img_path (str): Path to the image file.
            data (dict): The data to cache for the image.
        """
        self.cache[img_path] = data
        if self.verbose >= 2:
            print(f"Cache updated for {img_path}")

    def clear(self):
        """
        Clears the entire cache.
        """
        self.cache.clear()
        if self.verbose >= 1:
            print("Cache cleared.")

    def close(self):
        """
        Closes the cache and performs cleanup.
        """
        self.cache.close()
        if self.verbose >= 2:
            print("Cache closed.")

    # Implement context manager protocol for easy cache management
    def __enter__(self):
        """
        Context manager entry point.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point.
        """
        self.close()
        return False