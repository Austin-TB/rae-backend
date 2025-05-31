import subprocess
import sys
import os

def run_python_script(script_path):
    # Check if the file exists
    if not os.path.isfile(script_path):
        return f"Error: File not found -> {script_path}"

    try:
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True
        )
        output = result.stdout
        error = result.stderr

        if result.returncode == 0:
            return f"Output:\n{output}"
        else:
            return f"Script exited with errors (code {result.returncode}):\n{error}"

    except Exception as e:
        return f"Exception occurred: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python runner.py <path_to_target_script.py>")
    else:
        script_to_run = sys.argv[1]
        result = run_python_script(script_to_run)
        print(result)
