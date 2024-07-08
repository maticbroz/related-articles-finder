# Related Articles Finder

This project finds and lists related articles based on their titles and categories using SentenceTransformers.

## Features

- Finds top-k similar articles based on cosine similarity.
- Handles data from a JSON input file.
- Outputs results to a text file.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/related-articles-finder.git
    cd related-articles-finder
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your `input.json` file in the project directory.
2. Run the script:
    ```bash
    python main.py
    ```
3. The results will be saved in `output.txt`.

## Example

An example `input.json` file is provided for testing purposes.

## Dependencies

- pandas
- sentence-transformers
- numpy

## License

This project is licensed under the MIT License.
