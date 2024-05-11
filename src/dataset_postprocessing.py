import re

def escape_markdown_links(text) -> str: 
    pattern = r'(\[[^\]]+\]\([^)]+\))'
    matches = re.findall(pattern, text)
    replacements = {'[': '', ']': '', "(": " ", ")": ""}
    if len(matches):
        for match in matches:
            trans_table = str.maketrans(replacements)
            cleaned_match = match.translate(trans_table)
            text = text.replace(match, cleaned_match)
        return text
    return text

def remove_emoji(text) -> str: 
    try:
        emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                        "]+", re.UNICODE)
        return re.sub(emoj, '', text)
    except: return text

def remove_empty_lines(text) -> str: 
    try:
        return text.replace('\n', " ").replace('\r', '').replace("  ", " ")
    except: return str(text)

def remove_links(text):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    try:
        clean_text = re.sub(pattern, '', text)
    except: clean_text=text
    return clean_text

def remove_trash_phrases(text): 
    phrases = [
        "*"
        "(____",
        "____", 
        "__", 
        "Підпишіться на NV", 
        "Підпишіться на NV ",
        "Радіо Свобода. Підписуйтесь ", 
        "Радіо Свобода. Підписуйтесь"
        "Підписуйтесь ",
        "Підписуйтесь", 
        "Еспресо | Долучайтесь ", 
        "Еспресо | Долучайтесь", 
        "hromadske | підписатися", 
        "підписатися",
        "Підписатися", 
        "Підписуйтеся на наші соцмережі, щоб не пропустити головні новиниFacebook | Twitter | Instagram | Viber | WhatsApp", 
        "Try unlimited access Only 1 € for 4 weeksThen 69 € per month.Complete digital access to quality FT journalism on any device. Cancel anytime during your trial.", 

    ]
    for p in phrases: 
        text = text.replace(p, "")
    return text
