from typing import Any

def pretty_str_dict(d: dict[str, Any]) -> str:
    """
    Print dictionary with one `key: value` pair per line.
    Additionally pad the `key:` on the right such that starts
    of `value` align.

    Args:
        d (dict[str, Any]): The dictionary to format.
    Returns:
        str: The formatted string.
    """
    if not d:
        return ""
    
    # Find length of the longest key
    max_key_len = max(len(k) for k in d.keys())
    
    lines = []
    for key, value in d.items():
        # Pad key so values align, ensure at least one space after colon
        lines.append(f"{key}:{' ' * (max_key_len - len(key) + 1)}{value}")
    
    return "\n".join(lines)