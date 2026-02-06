"""AutoWork processor for the email service."""

import subprocess
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from processing a single PDF."""
    filename: str
    success: bool
    items: int = 0
    matched_items: int = 0
    estimate: float = 0.0
    output_dir: Optional[Path] = None
    csv_path: Optional[Path] = None
    json_path: Optional[Path] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class AutoWorkProcessor:
    """Runs AutoWork on PDF files."""
    
    def __init__(
        self,
        autowork_dir: Optional[Path] = None,
        timeout_seconds: int = 1800,  # 30 min default
        parallel: bool = False,
    ):
        """
        Initialize processor.
        
        Args:
            autowork_dir: Path to AutoWork code directory
            timeout_seconds: Max processing time per PDF
            parallel: Enable parallel OCR
        """
        if autowork_dir is None:
            autowork_dir = Path(__file__).parent.parent
        
        self.autowork_dir = Path(autowork_dir)
        self.timeout = timeout_seconds
        self.parallel = parallel
        
        # Verify AutoWork is available
        self.python = self.autowork_dir / 'venv' / 'bin' / 'python'
        self.main = self.autowork_dir / 'main.py'
        
        if not self.python.exists():
            raise FileNotFoundError(f"Python venv not found: {self.python}")
        if not self.main.exists():
            raise FileNotFoundError(f"main.py not found: {self.main}")
    
    def process(self, pdf_path: Path) -> ProcessingResult:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ProcessingResult with outputs and status
        """
        filename = pdf_path.name
        start_time = time.time()
        
        # Create output directory
        output_dir = Path(tempfile.mkdtemp(prefix='autowork_out_'))
        
        try:
            # Build command
            cmd = [
                str(self.python),
                str(self.main),
                str(pdf_path),
                str(output_dir),
                '--mode', 'hybrid',  # Use hybrid mode for better accuracy
                '--vision-budget', '10',  # Up to 10 pages per PDF for Vision API
            ]
            
            if self.parallel:
                cmd.append('--parallel')
            
            # Set environment to skip model source check
            env = {
                'PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK': 'True',
                'PATH': '/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin',
            }
            
            logger.info(f"Processing: {filename}")
            
            # Run AutoWork
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.autowork_dir),
                env={**subprocess.os.environ, **env},
            )
            
            processing_time = time.time() - start_time
            
            if result.returncode != 0:
                logger.error(f"AutoWork failed for {filename}: {result.stderr}")
                return ProcessingResult(
                    filename=filename,
                    success=False,
                    error=result.stderr[:500] if result.stderr else "Unknown error",
                    processing_time=processing_time,
                )
            
            # Find output files
            pdf_name = pdf_path.stem
            output_subdir = output_dir / pdf_name
            
            if not output_subdir.exists():
                # Try without subdirectory
                output_subdir = output_dir
            
            csv_files = list(output_subdir.glob('*_takeoff.csv'))
            json_files = list(output_subdir.glob('*_takeoff.json'))
            
            csv_path = csv_files[0] if csv_files else None
            json_path = json_files[0] if json_files else None
            
            # Parse results from JSON
            items = 0
            estimate = 0.0
            
            if json_path and json_path.exists():
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                    # pay_items contains all detected materials
                    items = len(data.get('pay_items', []))
                    # total_cost and matched count are in the summary
                    summary = data.get('summary', {})
                    estimate = summary.get('total_cost', 0) or 0
                    matched_items = summary.get('matched_items', 0) or 0
                except Exception as e:
                    logger.warning(f"Could not parse JSON results: {e}")
                    matched_items = 0
            
            logger.info(f"Completed: {filename} - {items} materials found, {matched_items} priced, ${estimate:,.2f}")
            
            return ProcessingResult(
                filename=filename,
                success=True,
                items=items,
                matched_items=matched_items,
                estimate=estimate,
                output_dir=output_subdir,
                csv_path=csv_path,
                json_path=json_path,
                processing_time=processing_time,
            )
            
        except subprocess.TimeoutExpired:
            processing_time = time.time() - start_time
            logger.error(f"Timeout processing {filename} after {self.timeout}s")
            return ProcessingResult(
                filename=filename,
                success=False,
                error=f"Processing timed out after {self.timeout}s",
                processing_time=processing_time,
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.exception(f"Error processing {filename}: {e}")
            return ProcessingResult(
                filename=filename,
                success=False,
                error=str(e),
                processing_time=processing_time,
            )
    
    def process_batch(self, pdf_paths: List[Path]) -> List[ProcessingResult]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_paths: List of PDF paths
            
        Returns:
            List of ProcessingResults
        """
        results = []
        for pdf_path in pdf_paths:
            result = self.process(pdf_path)
            results.append(result)
        return results


def cleanup_output(result: ProcessingResult):
    """Clean up temporary output directory."""
    if result.output_dir and result.output_dir.exists():
        try:
            # Move up one level if in subdirectory
            parent = result.output_dir.parent
            if parent.name.startswith('autowork_out_'):
                shutil.rmtree(parent, ignore_errors=True)
            else:
                shutil.rmtree(result.output_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Could not cleanup {result.output_dir}: {e}")
