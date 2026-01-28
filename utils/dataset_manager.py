"""Dataset History & Management Module"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class DatasetManager:
    """Manage dataset history, versions, and metadata"""

    def __init__(self, workspace_dir: str = "data/workspace"):
        """
        Initialize dataset manager
        
        Args:
            workspace_dir: Directory to store dataset metadata
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.workspace_dir / "datasets.json"
        self.datasets = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"datasets": []}

    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.datasets, f, indent=2)

    def add_dataset(self, filename: str, filepath: str, rows: int, cols: int, 
                    file_size: float, description: str = "") -> Dict:
        """
        Add a new dataset to history
        
        Args:
            filename: Dataset name
            filepath: Full path to file
            rows: Number of rows
            cols: Number of columns
            file_size: File size in MB
            description: Dataset description
            
        Returns:
            Dataset metadata
        """
        dataset = {
            "id": len(self.datasets["datasets"]) + 1,
            "filename": filename,
            "filepath": filepath,
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "rows": rows,
            "cols": cols,
            "size_mb": round(file_size, 2),
            "description": description,
            "liked": False,
            "version": 1,
            "tags": []
        }
        
        self.datasets["datasets"].append(dataset)
        self._save_metadata()
        
        return dataset

    def get_recent_datasets(self, limit: int = 10) -> List[Dict]:
        """Get recent datasets"""
        sorted_datasets = sorted(
            self.datasets["datasets"],
            key=lambda x: x['modified'],
            reverse=True
        )
        return sorted_datasets[:limit]

    def get_liked_datasets(self) -> List[Dict]:
        """Get liked datasets"""
        return [d for d in self.datasets["datasets"] if d.get('liked', False)]

    def toggle_like(self, dataset_id: int) -> bool:
        """Toggle like status for a dataset"""
        for dataset in self.datasets["datasets"]:
            if dataset["id"] == dataset_id:
                dataset["liked"] = not dataset["liked"]
                self._save_metadata()
                return dataset["liked"]
        return False

    def rename_dataset(self, dataset_id: int, new_name: str) -> Optional[Dict]:
        """Rename a dataset"""
        for dataset in self.datasets["datasets"]:
            if dataset["id"] == dataset_id:
                dataset["filename"] = new_name
                dataset["modified"] = datetime.now().isoformat()
                self._save_metadata()
                return dataset
        return None

    def update_description(self, dataset_id: int, description: str) -> Optional[Dict]:
        """Update dataset description"""
        for dataset in self.datasets["datasets"]:
            if dataset["id"] == dataset_id:
                dataset["description"] = description
                dataset["modified"] = datetime.now().isoformat()
                self._save_metadata()
                return dataset
        return None

    def add_tags(self, dataset_id: int, tags: List[str]) -> Optional[Dict]:
        """Add tags to dataset"""
        for dataset in self.datasets["datasets"]:
            if dataset["id"] == dataset_id:
                dataset["tags"] = list(set(dataset.get("tags", []) + tags))
                dataset["modified"] = datetime.now().isoformat()
                self._save_metadata()
                return dataset
        return None

    def create_version(self, dataset_id: int, version_name: str) -> Optional[Dict]:
        """Create a new version of dataset"""
        for dataset in self.datasets["datasets"]:
            if dataset["id"] == dataset_id:
                new_version = {
                    "id": len(self.datasets["datasets"]) + 1,
                    "filename": f"{dataset['filename']}_v{dataset['version']+1}",
                    "filepath": dataset["filepath"],
                    "created": datetime.now().isoformat(),
                    "modified": datetime.now().isoformat(),
                    "rows": dataset["rows"],
                    "cols": dataset["cols"],
                    "size_mb": dataset["size_mb"],
                    "description": f"{version_name} (from v{dataset['version']})",
                    "liked": False,
                    "version": dataset['version'] + 1,
                    "parent_id": dataset["id"],
                    "tags": dataset.get("tags", [])
                }
                self.datasets["datasets"].append(new_version)
                self._save_metadata()
                return new_version
        return None

    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset"""
        original_count = len(self.datasets["datasets"])
        self.datasets["datasets"] = [d for d in self.datasets["datasets"] if d["id"] != dataset_id]
        
        if len(self.datasets["datasets"]) < original_count:
            self._save_metadata()
            return True
        return False

    def get_dataset_by_id(self, dataset_id: int) -> Optional[Dict]:
        """Get dataset by ID"""
        for dataset in self.datasets["datasets"]:
            if dataset["id"] == dataset_id:
                return dataset
        return None

    def search_datasets(self, query: str) -> List[Dict]:
        """Search datasets by name or tags"""
        query_lower = query.lower()
        results = []
        
        for dataset in self.datasets["datasets"]:
            if (query_lower in dataset["filename"].lower() or 
                query_lower in dataset["description"].lower() or
                any(query_lower in tag.lower() for tag in dataset.get("tags", []))):
                results.append(dataset)
        
        return results

    def get_statistics(self) -> Dict:
        """Get workspace statistics"""
        datasets = self.datasets["datasets"]
        
        return {
            "total_datasets": len(datasets),
            "total_size_mb": round(sum(d["size_mb"] for d in datasets), 2),
            "total_rows": sum(d["rows"] for d in datasets),
            "average_size_mb": round(sum(d["size_mb"] for d in datasets) / len(datasets) if datasets else 0, 2),
            "liked_count": len([d for d in datasets if d.get("liked", False)]),
            "oldest_dataset": min([d["created"] for d in datasets]) if datasets else None,
            "newest_dataset": max([d["created"] for d in datasets]) if datasets else None
        }
