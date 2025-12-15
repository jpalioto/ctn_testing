"""Loader protocol - interface for document loading.

All dataset loaders implement this protocol, allowing uniform
handling of different document formats and sources.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from ..document import Document
from ..ground_truth import DocumentWithGroundTruth, GroundTruth


class DocumentLoader(ABC):
    """Abstract base for document loaders.
    
    Implementations handle specific formats:
    - TextLoader: .txt, .json files
    - DocILELoader: DocILE dataset format
    - FUNSDLoader: FUNSD dataset format
    - etc.
    """
    
    @abstractmethod
    def load(self, path: Path) -> Document:
        """Load a single document from path.
        
        Args:
            path: Path to document file
            
        Returns:
            Document instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is invalid
        """
        ...
    
    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Check if this loader supports the given path.
        
        Args:
            path: Path to check
            
        Returns:
            True if this loader can handle the path
        """
        ...


class DatasetLoader(ABC):
    """Abstract base for dataset loaders.
    
    Loads documents with their ground truth labels for evaluation.
    """
    
    @abstractmethod
    def load_all(self, dataset_path: Path) -> list[DocumentWithGroundTruth]:
        """Load all documents from dataset.
        
        Args:
            dataset_path: Root path of dataset
            
        Returns:
            List of documents with ground truth
        """
        ...
    
    @abstractmethod
    def iter_documents(self, dataset_path: Path) -> Iterator[DocumentWithGroundTruth]:
        """Iterate over dataset documents.
        
        For large datasets, avoids loading everything into memory.
        
        Args:
            dataset_path: Root path of dataset
            
        Yields:
            DocumentWithGroundTruth instances
        """
        ...
    
    def load_sample(self, dataset_path: Path, n: int) -> list[DocumentWithGroundTruth]:
        """Load first n documents from dataset.
        
        Args:
            dataset_path: Root path of dataset
            n: Number of documents to load
            
        Returns:
            List of up to n documents
        """
        results = []
        for doc in self.iter_documents(dataset_path):
            results.append(doc)
            if len(results) >= n:
                break
        return results


class GroundTruthLoader(ABC):
    """Abstract base for ground truth loaders.
    
    Separated from document loading to allow mixing:
    - Same documents, different annotation schemes
    - Different ground truth formats (YAML, JSON, custom)
    """
    
    @abstractmethod
    def load(self, path: Path) -> dict[str, GroundTruth]:
        """Load ground truth from path.
        
        Args:
            path: Path to ground truth file
            
        Returns:
            Dict mapping field names to GroundTruth
        """
        ...
    
    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Check if this loader supports the given path."""
        ...
