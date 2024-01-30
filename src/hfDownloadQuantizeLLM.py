import subprocess
import sys
import os
from huggingface_hub import login

def hf_login():
    """
    Logs into Hugging Face using a token from an environment variable.

    This function retrieves the Hugging Face API token from the environment variable 'HF_TOKEN'.
    If the token exists, it attempts to log in to Hugging Face. If the token is absent or login
    fails, it prints an appropriate message.
    """
    # Retrieve Hugging Face token from environment variable
    hf_token = os.getenv('HF_TOKEN')

    # Check if the token is available
    if hf_token:
        try:
            # Attempt to log in with the provided token
            login(token=hf_token)
            print("Logged in to Hugging Face successfully.")
        except Exception as e:
            # Handle exceptions during login and print error message
            print(f"Failed to log in to Hugging Face: {e}")
    else:
        # Token not found in environment variables
        print("Hugging Face token not found in environment variables.")

def clone_repository(repository_url):
    """
    Clones a Git repository at the specified URL.

    This function first initializes Git LFS and then clones the repository from the provided URL.
    It handles subprocesses for running Git commands and prints relevant output or error messages.

    Parameters:
    repository_url (str): URL of the Git repository to be cloned.
    """
    try:
        # Initialize Git LFS before cloning
        lfs_process = subprocess.run(["git", "lfs", "install"], check=True, text=True, capture_output=True)
        print("Git LFS initialized.\n" + lfs_process.stdout)

        # Run git clone with the provided URL
        with subprocess.Popen(["git", "clone", repository_url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            for line in proc.stderr:  # Git clone progress is sent to stderr
                print(line, end='')

        # Check if the cloning was successful
        if proc.returncode == 0:
            print("Repository cloned successfully.")
        else:
            print("Error occurred while cloning the repository.")
    except subprocess.CalledProcessError as e:
        # Handle errors during Git LFS initialization
        print("Error during Git LFS initialization:")
        print(e.stderr)
    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")

def extract_model_id(repository_url):
    """
    Extracts the model ID from a Hugging Face repository URL.

    This function splits the provided repository URL to extract the last two parts, 
    which constitute the model ID in the standard Hugging Face repository URL format.

    Parameters:
    repository_url (str): The full URL of the Hugging Face repository.

    Returns:
    str: The extracted model ID.
    """
    # Split the URL by '/'
    parts = repository_url.split('/')

    # Extract the model ID (last two parts of the URL)
    model_id = '/'.join(parts[-2:])

    return model_id

def convert_model_to_fp16(model_name, fp16_output_file):
    """
    Converts a model to fp16 format using a Python script.

    This function constructs and executes a command to run a Python script that converts
    a model to fp16 format. The script is assumed to be located in 'llama.cpp' directory.

    Parameters:
    model_name (str): The name of the model to be converted.
    fp16_output_file (str): The file path for the converted model output.
    """
    # Construct the command
    command = ["python", "llama.cpp/convert.py", model_name, "--outtype", "f16", "--outfile", fp16_output_file]

    # Execute the command
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Model conversion to fp16 successful. Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("An error occurred during model conversion:")
        print(e.stderr)

QUANTIZATION_METHODS = {
    "q2_k": "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l": "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m": "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s": "Uses Q3_K for all tensors",
    "q4_0": "Original quant method, 4-bit.",
    "q4_1": "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_m": "Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q4_k_s": "Uses Q4_K for all tensors",
    "q5_0": "Higher accuracy, higher resource usage and slower inference.",
    "q5_1": "Even higher accuracy, resource usage and slower inference.",
    "q5_k_m": "Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q5_k_s": "Uses Q5_K for all tensors",
    "q6_k": "Uses Q8_K for all tensors",
    "q8_0": "Almost indistinguishable from float16. High resource use and slow. Not recommended for most users."
}

def main():
    """
    Main function to execute the script.

    This function orchestrates the script execution by first logging into Hugging Face and
    then cloning a specified Git repository. It extracts the model ID from the repository URL,
    converts the model to fp16 format, and then performs quantization based on the user-specified
    method or prompts for it if not provided.
    """
    # Log into Hugging Face
    hf_login()

    # Get repository URL from command-line arguments or prompt for it
    if len(sys.argv) > 1:
        repository_url = sys.argv[1]
    else:
        repository_url = input("Enter the Git repository URL: ")

    # Extract the model ID and prepare for model conversion
    model_id = extract_model_id(repository_url)
    MODEL_NAME = model_id.split('/')[-1]
    fp16_output_file = f"{MODEL_NAME}/{MODEL_NAME.lower()}.fp16.bin"

    # Clone the specified repository
    clone_repository(repository_url)

    # Convert the model to fp16 format
    convert_model_to_fp16(MODEL_NAME, fp16_output_file)

    # Obtain the quantization method
    if len(sys.argv) > 2:
        method = sys.argv[2]
    else:
        print("Please choose a quantization method from the following options:")
        for key, description in QUANTIZATION_METHODS.items():
            print(f"{key}: {description}")
        method = input("Enter your chosen quantization method: ")

    # Prepare quantization command
    qtype = f"{MODEL_NAME}/{MODEL_NAME.lower()}.{method.upper()}.gguf"
    quantize_command = f"./llama.cpp/quantize {fp16_output_file} {qtype} {method}"

    # Execute quantization command
    try:
        subprocess.run(quantize_command, shell=True, check=True)
        print(f"Quantization with method '{method}' completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during quantization: {e}")

if __name__ == "__main__":
    main()