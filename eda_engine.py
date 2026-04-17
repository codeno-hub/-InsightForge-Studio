from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class EDAReportResult:
    report_path: Path


def generate_eda_report(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    title: str = "Automated EDA Report",
    minimal: bool = True,
    filename: str = "eda_report.html",
) -> EDAReportResult:
    """
    Generate an automated EDA HTML report and write it to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / filename

    from ydata_profiling import ProfileReport

    profile = ProfileReport(
        df,
        title=title,
        minimal=minimal,
        explorative=not minimal,
    )
    profile.to_file(str(report_path))
    return EDAReportResult(report_path=report_path)


def generate_eda_report_html(
    df: pd.DataFrame,
    *,
    title: str = "Automated EDA Report",
    minimal: bool = True,
) -> bytes:
    """
    Generate an automated EDA HTML report and return it as bytes.
    """
    from ydata_profiling import ProfileReport

    profile = ProfileReport(
        df,
        title=title,
        minimal=minimal,
        explorative=not minimal,
    )
    return profile.to_html().encode("utf-8")


def try_generate_eda_report(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    title: str = "Automated EDA Report",
    minimal: bool = True,
    filename: str = "eda_report.html",
) -> Optional[EDAReportResult]:
    """
    Best-effort EDA report generation. Returns None if dependencies are missing.
    """
    try:
        return generate_eda_report(
            df,
            output_dir=output_dir,
            title=title,
            minimal=minimal,
            filename=filename,
        )
    except Exception:
        return None

