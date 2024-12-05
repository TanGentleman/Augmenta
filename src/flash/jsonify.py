import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
FLASH_DATA_FOLDER = ROOT / "data"
INPUTS_FOLDER = FLASH_DATA_FOLDER / "inputs"
OUTPUTS_FOLDER = FLASH_DATA_FOLDER / "outputs"

from langchain.output_parsers import JsonOutputParser

def is_valid_flashcard_list(obj: any) -> bool:
    """
    Validates that an object is a list of dictionaries suitable for flashcards.

    Args:
        obj (any): The object to validate.

    Returns:
        bool: True if the object is a valid list of dictionaries.
    """
    return isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict)

def validate_flashcard(card: dict) -> bool:
    """
    Validates that a dictionary has the required fields to be a flashcard.

    Args:
        card (dict): The dictionary to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    # required_fields = ['front', 'back']
    # return all(field in card for field in required_fields)
    # Check that the card has at least one field
    if not card:
        return False
    return True

def parse_flashcards(json_string: str) -> list[dict]:
    """
    Parses a JSON string into a validated list of flashcard dictionaries using JsonOutputParser.

    Args:
        json_string (str): The JSON string to parse.

    Returns:
        list[dict]: A list of validated flashcard dictionaries.
        
    Raises:
        ValueError: If parsing fails or flashcards are invalid.
    """
    try:
        # Use JsonOutputParser to parse the string
        parser = JsonOutputParser()
        flashcards = parser.parse(json_string)
        
        # Validate the overall structure
        if not is_valid_flashcard_list(flashcards):
            raise ValueError("Parsed JSON is not a valid list of dictionaries")
            
        # Validate each flashcard
        invalid_cards = [i for i, card in enumerate(flashcards) if not validate_flashcard(card)]
        if invalid_cards:
            raise ValueError(f"Invalid flashcards at indices: {invalid_cards}")
            
        return flashcards
        
    except Exception as e:
        raise ValueError(f"Failed to parse flashcards: {str(e)}")



def fix_filename(filename: str | Path) -> str:
    """
    Fixes the filename by ensuring it is an appropriate string format.

    Args:
        filename (str | Path): The filename to be fixed.

    Returns:
        str: The fixed filename.
    """
    if isinstance(filename, Path):
        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        print("Warning: Converting path to just the filename")
        return str(filename.name)
    return filename


def copy_file_to_inputs(file_path: str | Path) -> bool:
    """
    Copies a file from the given filepath to the inputs folder.

    Args:
        file_path (str | Path): The path to the file to be copied.

    Returns:
        bool: True if the copy operation was successful, False otherwise.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        new_file_path = INPUTS_FOLDER / file_path.name
        new_file_path.write_bytes(file_path.read_bytes())
        return True
    except Exception as e:
        print(f"Error copying file to inputs folder: {e}")
        return False


def write_jsonl(data, filename: str | Path):
    """
    Writes a list of dictionaries to a .jsonl file.

    Args:
        data (list of dict): A list of dictionaries where each dictionary represents a JSON object.
        filename (str | Path): The path to the .jsonl file where the data will be written.

    Returns:
        bool: True if the write operation was successful, False otherwise.
    """
    file_path = OUTPUTS_FOLDER / fix_filename(filename)
    try:
        with open(file_path, 'w') as file:
            for entry in data:
                if not isinstance(entry, dict):
                    raise ValueError(
                        "Each entry in the data list must be a dictionary.")
                json.dump(entry, file)
                file.write('\n')
        return True
    except Exception as e:
        print(f"Error writing to .jsonl file: {e}")
        return False


def read_jsonl(filename: str | Path):
    """
    Reads a .jsonl file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the .jsonl file to be read.

    Returns:
        list of dict: A list of dictionaries where each dictionary represents a JSON object from the file.
                      Returns an empty list if an error occurs.
    """
    file_path = INPUTS_FOLDER / fix_filename(filename)

    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    entry = json.loads(line)
                    if not isinstance(entry, dict):
                        raise ValueError(
                            "Each line in the .jsonl file must be a valid JSON object (dictionary).")
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {e}")
    except Exception as e:
        print(f"Error reading .jsonl file: {e}")
        return None
    return data


def write_json(data, filename: str | Path):
    """
    Writes a list of dictionaries to a .json file.

    Args:
        data (list of dict): A list of dictionaries where each dictionary represents a JSON object.
        file_path (str): The path to the .json file where the data will be written.

    Returns:
        bool: True if the write operation was successful, False otherwise.
    """
    file_path = OUTPUTS_FOLDER / fix_filename(filename)

    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        return True
    except Exception as e:
        print(f"Error writing to .json file: {e}")
        return False


def read_json(filename: str | Path):
    """
    Reads a .json file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the .json file to be read.

    Returns:
        list of dict: A list of dictionaries where each dictionary represents a JSON object from the file.
                      Returns an empty list if an error occurs.
    """
    file_path = INPUTS_FOLDER / fix_filename(filename)

    data = []
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if not isinstance(data, list):
                raise ValueError(
                    "The JSON file must contain a list of dictionaries.")
            for entry in data:
                if not isinstance(entry, dict):
                    raise ValueError(
                        "Each entry in the JSON file must be a dictionary.")
    except Exception as e:
        print(f"Error reading .json file: {e}")
    return data


def run_experiment():
    # Read data from flashcards.json
    filename = "flashcards.json"
    loaded_data = read_json(filename)
    if not loaded_data:
        print(f"Error reading data from {filename}!")
        return
    print(f"Data read from {filename}!")

    # Write it to flashcards_converted.jsonl
    new_filename = "flashcards_converted.jsonl"
    if write_jsonl(loaded_data, new_filename):
        print(f"Data written to {new_filename} successfully!")
    else:
        raise ValueError(f"Error writing data to {new_filename}!")

    # Make a copy of the file in the inputs folder
    if copy_file_to_inputs(OUTPUTS_FOLDER / new_filename):
        print(f"File copied to inputs folder successfully!")
    else:
        raise ValueError(f"Error copying file to inputs folder!")

    # Print the data read from flashcards_converted.jsonl
    loaded_data_jsonl = read_jsonl(new_filename)
    if not loaded_data_jsonl:
        print(f"Error reading data from {new_filename}!")
        raise ValueError(f"Error reading data from {new_filename}!")
    print(f"Item of {len(loaded_data_jsonl)} from {new_filename}:",
          loaded_data_jsonl[:2])


def main():
    run_experiment()


# Example usage:
if __name__ == "__main__":
    run_experiment()
    exit()
    # Sample data
    sample_data = [
        {"text": "Buy groceries", "isCompleted": True},
        {"text": "Go for a swim", "isCompleted": True},
        {"text": "Integrate Convex", "isCompleted": False}
    ]

    # File paths
    jsonl_file_path = "tasks.jsonl"
    json_file_path = "tasks.json"

    # Write data to .jsonl file
    if write_jsonl(sample_data, jsonl_file_path):
        print("Data written successfully to .jsonl file.")
    else:
        print("Failed to write data to .jsonl file.")

    # Read data from .jsonl file
    loaded_data_jsonl = read_jsonl(jsonl_file_path)
    print("Data read from .jsonl file:", loaded_data_jsonl)

    # Write data to .json file
    if write_json(sample_data, json_file_path):
        print("Data written successfully to .json file.")
    else:
        print("Failed to write data to .json file.")

    # Read data from .json file
    loaded_data_json = read_json(json_file_path)
    print("Data read from .json file:", loaded_data_json)
