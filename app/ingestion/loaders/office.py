import logfire
from unstructured.partition.auto import partition

def parse_office(file_path:str):
    try:
        elements = partition(filename = file_path)
        full_text = "\n".join([str(e) for e in elements])

        if not full_text.strip():
            logfire.warning(f"Unstructured return empty text for {file_path}")
        else:
            logfire.info(f"Successfully parsed {len{full_text}} characters")

        return full_text

    except Exception as e:
        logfire.error(f"Office parse failed: {e}")
        raise e