import os
import io
import sys
import uuid
import base64
import traceback
import contextlib
import tempfile
import subprocess
import sqlite3
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil

class CodeInterpreter:
    def __init__(self, allowed_modules=None, max_execution_time=30, working_directory=None):
        """Initialize the code interpreter with safety measures."""
        self.allowed_modules = allowed_modules or [
            "numpy", "pandas", "matplotlib", "scipy", "sklearn", 
            "math", "random", "statistics", "datetime", "collections",
            "itertools", "functools", "operator", "re", "json",
            "sympy", "networkx", "nltk", "PIL", "pytesseract", 
            "cmath", "uuid", "tempfile", "requests", "urllib"
        ]
        self.max_execution_time = max_execution_time

        # Define base path for all interpreter-specific temporary files
        self.base_temp_path = os.path.join(os.path.dirname(__file__), "code_interpreter_files")

        # Configure working directory for Python script outputs (e.g., plots)
        self.working_directory = working_directory or os.path.join(self.base_temp_path, "executions")
        
        # Configure path for the temporary SQLite database
        self.sqlite_db_dir = os.path.join(self.base_temp_path, "db")
        self.temp_sqlite_db = os.path.join(self.sqlite_db_dir, "code_exec.db")

        # Configure directory for C/Java compilation temporary files
        self.compile_temp_dir = os.path.join(self.base_temp_path, "compile_temp")

        # Ensure all necessary directories exist
        os.makedirs(self.working_directory, exist_ok=True)
        os.makedirs(self.sqlite_db_dir, exist_ok=True)
        os.makedirs(self.compile_temp_dir, exist_ok=True)
        
        self.globals = {
            "__builtins__": __builtins__,
            "np": np,
            "pd": pd,
            "plt": plt,
            "Image": Image,
        }

    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute the provided code in the selected programming language."""
        language = language.lower()
        execution_id = str(uuid.uuid4())
        
        result = {
            "execution_id": execution_id,
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
            "plots": [],
            "dataframes": []
        }
        
        try:
            if language == "python":
                return self._execute_python(code, execution_id)
            elif language == "bash":
                return self._execute_bash(code, execution_id)
            elif language == "sql":
                return self._execute_sql(code, execution_id)
            elif language == "c":
                return self._execute_c(code, execution_id)
            elif language == "java":
                return self._execute_java(code, execution_id)
            else:
                result["stderr"] = f"Unsupported language: {language}"
        except Exception as e:
            result["stderr"] = str(e)
        
        return result

    def _execute_python(self, code: str, execution_id: str) -> dict:
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        result = {
            "execution_id": execution_id,
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
            "plots": [],
            "dataframes": []
        }
        
        try:
            exec_dir = os.path.join(self.working_directory, execution_id)
            os.makedirs(exec_dir, exist_ok=True)
            plt.switch_backend('Agg')
            
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                exec_result = exec(code, self.globals)

                if plt.get_fignums():
                    for i, fig_num in enumerate(plt.get_fignums()):
                        fig = plt.figure(fig_num)
                        img_path = os.path.join(exec_dir, f"plot_{i}.png")
                        fig.savefig(img_path)
                        with open(img_path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            result["plots"].append({
                                "figure_number": fig_num,
                                "data": img_data
                            })

                for var_name, var_value in self.globals.items():
                    if isinstance(var_value, pd.DataFrame) and len(var_value) > 0:
                        result["dataframes"].append({
                            "name": var_name,
                            "head": var_value.head().to_dict(),
                            "shape": var_value.shape,
                            "dtypes": str(var_value.dtypes)
                        })
                
            result["status"] = "success"
            result["stdout"] = output_buffer.getvalue()
            result["result"] = exec_result
            
        except Exception as e:
            result["status"] = "error"
            result["stderr"] = f"{error_buffer.getvalue()}\n{traceback.format_exc()}"
        
        return result

    def _execute_bash(self, code: str, execution_id: str) -> dict:
        try:
            completed = subprocess.run(
                code, shell=True, capture_output=True, text=True, timeout=self.max_execution_time
            )
            return {
                "execution_id": execution_id,
                "status": "success" if completed.returncode == 0 else "error",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "result": None,
                "plots": [],
                "dataframes": []
            }
        except subprocess.TimeoutExpired:
            return {
                "execution_id": execution_id,
                "status": "error",
                "stdout": "",
                "stderr": "Execution timed out.",
                "result": None,
                "plots": [],
                "dataframes": []
            }

    def _execute_sql(self, code: str, execution_id: str) -> dict:
        result = {
            "execution_id": execution_id,
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
            "plots": [],
            "dataframes": []
        }
        try:
            conn = sqlite3.connect(self.temp_sqlite_db)
            cur = conn.cursor()
            cur.execute(code)
            if code.strip().lower().startswith("select"):
                columns = [description[0] for description in cur.description]
                rows = cur.fetchall()
                df = pd.DataFrame(rows, columns=columns)
                result["dataframes"].append({
                    "name": "query_result",
                    "head": df.head().to_dict(),
                    "shape": df.shape,
                    "dtypes": str(df.dtypes)
                })
            else:
                conn.commit()

            result["status"] = "success"
            result["stdout"] = "Query executed successfully."

        except Exception as e:
            result["stderr"] = str(e)
        finally:
            conn.close()

        return result

    def _execute_c(self, code: str, execution_id: str) -> dict:
        temp_dir = tempfile.mkdtemp(dir=self.compile_temp_dir)
        source_path = os.path.join(temp_dir, "program.c")
        binary_path = os.path.join(temp_dir, "program")

        try:
            with open(source_path, "w") as f:
                f.write(code)

            compile_proc = subprocess.run(
                ["gcc", source_path, "-o", binary_path],
                capture_output=True, text=True, timeout=self.max_execution_time
            )
            if compile_proc.returncode != 0:
                return {
                    "execution_id": execution_id,
                    "status": "error",
                    "stdout": compile_proc.stdout,
                    "stderr": compile_proc.stderr,
                    "result": None,
                    "plots": [],
                    "dataframes": []
                }

            run_proc = subprocess.run(
                [binary_path],
                capture_output=True, text=True, timeout=self.max_execution_time
            )
            return {
                "execution_id": execution_id,
                "status": "success" if run_proc.returncode == 0 else "error",
                "stdout": run_proc.stdout,
                "stderr": run_proc.stderr,
                "result": None,
                "plots": [],
                "dataframes": []
            }
        except Exception as e:
            return {
                "execution_id": execution_id,
                "status": "error",
                "stdout": "",
                "stderr": str(e),
                "result": None,
                "plots": [],
                "dataframes": []
            }

    def _execute_java(self, code: str, execution_id: str) -> dict:
        temp_dir = tempfile.mkdtemp(dir=self.compile_temp_dir)
        source_path = os.path.join(temp_dir, "Main.java")

        try:
            with open(source_path, "w") as f:
                f.write(code)

            compile_proc = subprocess.run(
                ["javac", source_path],
                capture_output=True, text=True, timeout=self.max_execution_time
            )
            if compile_proc.returncode != 0:
                return {
                    "execution_id": execution_id,
                    "status": "error",
                    "stdout": compile_proc.stdout,
                    "stderr": compile_proc.stderr,
                    "result": None,
                    "plots": [],
                    "dataframes": []
                }

            run_proc = subprocess.run(
                ["java", "-cp", temp_dir, "Main"],
                capture_output=True, text=True, timeout=self.max_execution_time
            )
            return {
                "execution_id": execution_id,
                "status": "success" if run_proc.returncode == 0 else "error",
                "stdout": run_proc.stdout,
                "stderr": run_proc.stderr,
                "result": None,
                "plots": [],
                "dataframes": []
            }
        except Exception as e:
            return {
                "execution_id": execution_id,
                "status": "error",
                "stdout": "",
                "stderr": str(e),
                "result": None,
                "plots": [],
                "dataframes": []
            }

if __name__ == "__main__":
    print("Executing code_interpreter.py directly...")

    # Define the base path for temporary files, consistent with CodeInterpreter class
    main_base_temp_path = os.path.join(os.path.dirname(__file__), "code_interpreter_files")

    # Clear the existing base temporary directory if it exists
    if os.path.exists(main_base_temp_path):
        print(f"Found existing directory: {main_base_temp_path}. Clearing it...")
        shutil.rmtree(main_base_temp_path)
        print(f"Successfully cleared directory: {main_base_temp_path}")
    else:
        print(f"Directory {main_base_temp_path} not found. It will be created by CodeInterpreter.")

    print("\nInitializing CodeInterpreter for testing...")
    # Instantiating CodeInterpreter will now create the base_temp_path and its subdirectories
    # (executions, db, compile_temp) because __init__ calls os.makedirs.
    interpreter = CodeInterpreter()

    print(f"CodeInterpreter instance created.")
    print(f"  Base temporary path: {interpreter.base_temp_path}")
    print(f"  Working directory: {interpreter.working_directory}")
    print(f"  SQLite DB directory: {interpreter.sqlite_db_dir}")
    print(f"  SQLite DB file: {interpreter.temp_sqlite_db}")
    print(f"  Compilation temp directory: {interpreter.compile_temp_dir}")

    # Verify that the directories were created by __init__
    print("\nVerifying directory creation...")
    expected_dirs_to_check = [
        interpreter.working_directory,
        interpreter.sqlite_db_dir,
        interpreter.compile_temp_dir
    ]
    all_dirs_exist = True
    for dir_path in expected_dirs_to_check:
        if os.path.exists(dir_path):
            print(f"  [OK] Directory exists: {dir_path}")
        else:
            print(f"  [ERROR] Directory NOT FOUND: {dir_path}")
            all_dirs_exist = False
    
    if all_dirs_exist:
        print("All expected directories verified successfully.")
    else:
        print("One or more directories were not created as expected. Please check __init__.")

    # Example of how C/Java compilation temp files would be handled:
    # Simulating mkdtemp usage within the specified compile_temp_dir
    if os.path.exists(interpreter.compile_temp_dir):
        test_compile_subdir = tempfile.mkdtemp(dir=interpreter.compile_temp_dir)
        print(f"\nTest: Created a temporary compilation subdirectory: {test_compile_subdir}")
        if os.path.exists(test_compile_subdir) and test_compile_subdir.startswith(interpreter.compile_temp_dir):
            print("  [OK] Test subdirectory is within the configured compile_temp_dir.")
            try:
                os.rmdir(test_compile_subdir) # Clean up empty test dir
                print(f"  Cleaned up test subdirectory: {test_compile_subdir}")
            except OSError as e:
                print(f"  Could not remove test subdirectory {test_compile_subdir}: {e} (might not be empty if used by other process)")
        else:
            print("  [ERROR] Test subdirectory was not created correctly or not in the expected location.")
    else:
        print("\nCompile temp directory does not exist, skipping mkdtemp test.")
