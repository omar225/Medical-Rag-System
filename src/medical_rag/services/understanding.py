"""Small conversational grammar and conservative source-vocabulary spelling checks.

No medication relationships, benchmark answers, or retrieval configuration live here.
"""
import re

from medical_rag.retrieval.normalization import normalize_text

GREETING = r"(?:hello|hi|hey|good\s*morning|good\s*afternoon|good\s*evening|صباح الخير|مساء الخير|اهلا|السلام عليكم)"
PERMISSION = r"(?:can|may|could) i ask (?:you )?(?:a question|something)|ممكن اسالك سوال|ممكن اسالك سؤال"


def strip_greeting(question):
    normalized = normalize_text(question)
    return re.sub(r"^" + GREETING + r"(?:[\s,.!،؟?]+|$)", "", normalized).strip()


def conversational_reply(question, language):
    """Full-string recognition: trailing medical questions must never disappear."""
    text = strip_greeting(question).strip(" .,!؟?،")
    arabic = language != "en"
    if not text and re.fullmatch(GREETING + r"[\s,.!،؟?]*", normalize_text(question)):
        return "أهلاً! تحب تسأل عن إيه؟" if arabic else "Hello! What would you like to know?"
    if re.fullmatch(PERMISSION, text):
        return "أكيد، اتفضل! تحب تعرف إيه؟" if arabic else "Of course! What would you like to know?"
    if re.fullmatch(r"thanks(?: a lot)?|thank you(?: very much)?|شكرا|شكراً|متشكر|تسلم", text):
        return "العفو! لو عندك سؤال تاني، اتفضل." if arabic else "You're welcome! Feel free to ask another question."
    return None


def one_edit(left, right):
    """Exactly one insertion, deletion, substitution, or adjacent transposition."""
    if left == right or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        changed = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
        return len(changed) == 1 or (len(changed) == 2 and changed[1] == changed[0] + 1
            and left[changed[0]] == right[changed[1]] and left[changed[1]] == right[changed[0]])
    short, long = sorted((left, right), key=len)
    return any(long[:i] + long[i+1:] == short for i in range(len(long)))


class QuestionUnderstanding:
    def __init__(self, checks):
        self.checks = checks
        self.names = {}
        # Only explicitly headed brand/ingredient tables, never side-effect tables.
        for chunk in checks.corpus.chunks:
            active = False
            for line in chunk['text'].splitlines():
                if '|' not in line:
                    active = False
                    continue
                left, right = (part.strip() for part in line.split('|', 1))
                if 'ingredient' in normalize_text(right) and re.search(r'brand|trade|تجاري', normalize_text(left)):
                    active = True
                    continue
                if active:
                    for name in (left, right):
                        if re.fullmatch(r'[A-Za-z]{5,}', name):
                            self.names[normalize_text(name)] = name
        for entry in checks.aliases.entries:
            if entry['status'] == 'confirmed' and entry['id'] in checks.available and entry['id'].startswith(('drug-', 'brand-')):
                for name in [entry['canonical'], *entry['variants']]:
                    if len(name) >= 5 and ' ' not in name:
                        self.names[normalize_text(name)] = entry['canonical']

    def clarify_topic(self, question, language):
        text = strip_greeting(question).strip(' .,!؟?،')
        match = re.fullmatch(r'(?:(?:i (?:need|want|would like) to ask (?:a question )?about)|'
                             r'(?:عايز|عاوز|حابب) اسال عن)\s+(.+)', text)
        topic = match[1] if match else text
        # Clarify only a bare, verified name; arbitrary residual text stays unsupported.
        viewed = self.checks.view(topic)
        if viewed not in self.names:
            return None
        name = self.names[viewed]
        return (f'أهلاً! تحب تعرف إيه عن {name}؟ اكتب اسم الدواء مع سؤالك.' if language != 'en'
                else f'Hello! What would you like to know about {name}? Include the medicine name in your question.')

    def spelling(self, question):
        """Return a proposed correction, never overwrite known names or qualifiers.

        Restrict automatic recovery to a single unknown subject token and a unique
        long name with a one-letter INSERTION. Substitutions/short names are only
        suggestions: similar medicine names can be clinically distinct.
        """
        subjects = self.checks.subjects(question)
        if len(subjects) != 1 or self.checks.intent(question) is None:
            return question, None, []
        word = next(iter(subjects))
        if word in self.names:
            return question, None, []
        candidates = sorted({target for name, target in self.names.items() if one_edit(word, name)})
        if len(candidates) == 1:
            target = candidates[0]
            normalized = normalize_text(target)
            if len(word) >= 7 and len(normalized) == len(word) + 1 and one_edit(word, normalized):
                # Preserve all question text outside the candidate token.
                corrected = re.sub(r'(?<!\w)' + re.escape(word) + r'(?!\w)', target,
                                   self.checks.view(question))
                return corrected, (word, target), []
        return question, None, candidates[:4]
