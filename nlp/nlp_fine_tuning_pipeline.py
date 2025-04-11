import subprocess
import os

def run_command(command, working_dir=None):
    """
    Executes a shell command in a subprocess, capturing and printing its real-time output and error streams.
    Args:
        command (str): The shell command to execute.
        working_dir (str, optional): The directory in which to execute the command. Defaults to None.
    Returns:
        bool: True if the command executes successfully, False otherwise.
    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero return code.
    Notes:
        - The function uses `subprocess.Popen` to execute the command and capture its output and error streams.
        - Output and error messages are printed in real-time.
        - If the command fails, an error message is printed, and the function returns False.
    """
    try:
        # use Popen to get real-time output
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=working_dir
        )

        # print output in real-time
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            
            print(output.strip())
            print(error.strip())
            
            # Check if process has finished
            if output == '' and error == '' and process.poll() is not None:
                break

        return_code = process.poll()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        return False


if __name__ == "__main__":
    model = "Llama-3.2-1B-Instruct-4bit"
    iters = 1000

    model_dir = f"nlp/fine_tuning/models/{model}"
    os.makedirs(model_dir, exist_ok=True)
    model_dir = os.path.abspath(model_dir)

    fine_tune_text = f"""mlx_lm.lora \\
        --model mlx-community/{model} \\
        --train \\
        --data ../../data/json \\
        --iters {iters}"""
    run_command(fine_tune_text, model_dir)

    fuse_model_text = f"mlx_lm.fuse --model mlx-community/{model}"
    run_command(fuse_model_text, model_dir)