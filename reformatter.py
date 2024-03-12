from pyperclip import copy, paste

def reformat(content: str) -> str:
    # Split the content into lines
    lines = content.split('\n')
    letter = "A"
    new_lines = []
    #
    for line in lines:
        if "•" in line:
            line = line.replace("•", f"{letter}.")
            # Increment the letter
            letter = chr(ord(letter) + 1)
        new_lines.append(line)
    # Join the lines into a single string
    return '\n'.join(new_lines)

def main():
    # Get the clipboard's content
    content = paste().strip()

    # Reformat the content
    reformatted_content = reformat(content)

    # Copy the reformatted content to the clipboard
    copy(reformatted_content)

main()