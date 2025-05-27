import json

def format_to_sft(example):
    """
    Formats a single example into the SFT prompt-target pair for Qwen2.5 NER fine-tuning.

    Args:
        example (dict): A dict with keys:
            - title (str): the title of the text.
            - text (str): the input text to label.
            - entities (List[str]): list of company name entities in the text.

    Returns:
        str: A string containing the full chat-style prompt and the JSON array of entities.
    """
    # System prompt defining the NER task
    SYSTEM_PROMPT = (
        "You are a named-entity recognition assistant. Identify all company names "
        "in the following title and text, and output a JSON array of strings, "
        "each string being one company name."
    )

    # ChatML instruction delimiters
    inst_start = "<|im_start|>[INST] <<SYS>>\n"
    inst_end = "<</SYS>>\nTitle: \"{title}\"\nText: \"{text}\"\n[/INST]\n"
    inst_close = "<|im_end|>"

    # Build the JSON array of entities
    entities = example.get("entities", [])
    output_json = json.dumps(entities, ensure_ascii=False)

    # Compose final SFT example including title and text
    prompt = (
        f"{inst_start}"
        f"{SYSTEM_PROMPT}\n"
        f"{inst_end.format(title=example['title'], text=example['text'])}"
        # f"{output_json}\n"
        f"{inst_close}"
    )
    return prompt


# Example usage:
if __name__ == "__main__":
    example = {
        "title": "Microsoft Acquires GitHub",
        "text": "Microsoft announced today that it has acquired GitHub, and will integrate GitHub Copilot into its Azure cloud platform.",
        "entities": ["Microsoft", "GitHub", "GitHub Copilot", "Azure"]
    }
    print(format_to_sft(example))
