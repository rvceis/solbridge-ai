#!/usr/bin/env python3
"""Simple Tkinter UI to ingest CSV data and run the preprocessing pipeline.

Features:
- Select an input CSV (solar or consumption).
- Choose data type and system capacity (for solar).
- Optional validation before processing.
- Choose output directory; auto-names output with timestamp.
- Shows processing logs and summary.
- Opens the output folder on completion.
- Auto-detects CSV delimiter and encoding.
- Skips bad rows gracefully.
"""
import sys
import threading
import time
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

# Third-party
import pandas as pd

# Local imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.pipeline import DataPreprocessingPipeline  # noqa: E402


def detect_csv_delimiter(file_path: Path, sample_lines: int = 5) -> str:
    """Auto-detect CSV delimiter by sampling file lines."""
    delimiters = [',', ';', '\t', '|']
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                sample = ''.join([f.readline() for _ in range(sample_lines)])
            sniffer = csv.Sniffer()
            try:
                delimiter = sniffer.sniff(sample, delimiters=''.join(delimiters)).delimiter
                return delimiter
            except csv.Error:
                continue
        except Exception:
            continue
    
    return ','  # default to comma


class DataApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Solar Sharing - Data Prep")
        root.geometry("750x580")

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(ROOT / "data/processed"))
        self.data_type = tk.StringVar(value="solar")
        self.capacity_kw = tk.DoubleVar(value=5.0)
        self.validate_flag = tk.BooleanVar(value=True)
        self.skip_rows = tk.IntVar(value=0)  # NEW: skip metadata rows

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # Input file picker
        ttk.Label(self.root, text="Input CSV").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(self.root, textvariable=self.input_path, width=60).grid(row=0, column=1, **pad)
        ttk.Button(self.root, text="Browse", command=self._choose_input).grid(row=0, column=2, **pad)

        # Output directory picker
        ttk.Label(self.root, text="Output Folder").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(self.root, textvariable=self.output_dir, width=60).grid(row=1, column=1, **pad)
        ttk.Button(self.root, text="Browse", command=self._choose_output).grid(row=1, column=2, **pad)

        # Data type and capacity
        ttk.Label(self.root, text="Data Type").grid(row=2, column=0, sticky="w", **pad)
        type_cb = ttk.Combobox(self.root, textvariable=self.data_type, values=["solar", "consumption"], state="readonly")
        type_cb.grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(self.root, text="System Capacity (kW)").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(self.root, textvariable=self.capacity_kw, width=10).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(self.root, text="Skip Header Rows").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(self.root, textvariable=self.skip_rows, width=10).grid(row=4, column=1, sticky="w", **pad)
        ttk.Label(self.root, text="(e.g., 2 for NSRDB metadata rows)", font=("", 8)).grid(row=4, column=2, sticky="w")

        # Validation toggle
        ttk.Checkbutton(self.root, text="Validate before processing", variable=self.validate_flag).grid(row=5, column=1, sticky="w", **pad)

        # Action buttons
        ttk.Button(self.root, text="Run Preprocessing", command=self._run_async).grid(row=6, column=1, sticky="w", **pad)
        ttk.Button(self.root, text="Open Output Folder", command=self._open_output_folder).grid(row=6, column=2, sticky="w", **pad)

        # Log box
        ttk.Label(self.root, text="Logs").grid(row=7, column=0, sticky="nw", **pad)
        self.log_box = tk.Text(self.root, height=15, width=100, state="disabled", wrap="word")
        self.log_box.grid(row=7, column=1, columnspan=2, sticky="nsew", **pad)

        # Grid weights
        self.root.grid_rowconfigure(7, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def _choose_input(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.input_path.set(path)

    def _choose_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def _log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state="disabled")

    def _run_async(self):
        threading.Thread(target=self._run_preprocess, daemon=True).start()

    def _run_preprocess(self):
        input_path = Path(self.input_path.get()).expanduser()
        output_dir = Path(self.output_dir.get()).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            messagebox.showerror("Input missing", "Please select a valid input CSV.")
            return

        data_type = self.data_type.get()
        capacity = self.capacity_kw.get()
        validate = self.validate_flag.get()
        skip_rows = self.skip_rows.get()

        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{data_type}_processed_{ts}.csv"

        self._log(f"Input: {input_path}")
        self._log(f"Output: {output_path}")
        self._log(f"Type: {data_type}, capacity: {capacity} kW, validate: {validate}, skip_rows: {skip_rows}")

        try:
            # Auto-detect delimiter and encoding
            delimiter = detect_csv_delimiter(input_path)
            self._log(f"Detected delimiter: {repr(delimiter)}")
            
            # Try different encodings
            df = None
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    self._log(f"Trying encoding: {enc} with skiprows={skip_rows}")
                    df = pd.read_csv(
                        input_path, 
                        delimiter=delimiter,
                        encoding=enc,
                        on_bad_lines='skip',  # Skip problematic rows
                        engine='python',      # More robust parser
                        skiprows=skip_rows    # Skip metadata header rows
                    )
                    self._log(f"✓ Successfully read with encoding: {enc}")
                    break
                except Exception as e:
                    self._log(f"  ✗ Failed with {enc}: {str(e)[:60]}")
                    continue
            
            if df is None or len(df) == 0:
                raise ValueError("Could not read CSV with any encoding. Check file format and skip_rows value.")
            
            self._log(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            cols_preview = ', '.join(df.columns.tolist()[:10])
            if len(df.columns) > 10:
                cols_preview += ", ..."
            self._log(f"Columns: {cols_preview}")

            pipeline = DataPreprocessingPipeline()
            if data_type == "solar":
                processed = pipeline.preprocess_solar_data(df, system_capacity_kw=capacity, validate=validate)
            else:
                processed = pipeline.preprocess_consumption_data(df)

            processed.to_csv(output_path, index=False)
            removed = len(df) - len(processed)
            pct_removed = (removed / len(df) * 100) if len(df) else 0.0

            self._log(f"Saved processed data: {len(processed)} rows, {len(processed.columns)} columns")
            self._log(f"Rows removed: {removed} ({pct_removed:.1f}%)")
            self._log("✓ Done. You can open the output folder.")
            messagebox.showinfo("Success", f"Processing complete. Saved to:\n{output_path}")
        except Exception as exc:  # pragma: no cover - UI path
            self._log(f"✗ Error: {exc}")
            messagebox.showerror("Processing failed", str(exc))

    def _open_output_folder(self):
        path = Path(self.output_dir.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("linux"):
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(path)])
        elif sys.platform == "win32":
            import os
            os.startfile(str(path))
        else:
            messagebox.showinfo("Path", str(path))


def main():
    root = tk.Tk()
    app = DataApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
