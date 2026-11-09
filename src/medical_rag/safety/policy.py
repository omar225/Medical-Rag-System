"""Rule-based triage, not diagnosis. Categories intentionally prefer escalation."""

import re

from medical_rag.retrieval.normalization import normalize_text


def answer_language(text: str) -> str:
    arabic = bool(re.search(r"[\u0621-\u064a]", text))
    latin = bool(re.search(r"[a-zA-Z]", text))
    return "mixed" if arabic and latin else "ar" if arabic else "en"


def preferred_language(question: str) -> str:
    language = answer_language(question)
    if language == "en" and re.search(
        r"\b(eh|ezay|ezzay|yenfa3|a3rad|ganbeya|fa3ala|mada|ashrab|akhod|a5od|ko7ol|byetkawen|dawa|7obob|el|fe|ma3|mish|mesh|atnafas)\b",
        normalize_text(question),
    ):
        return "ar"
    return language


# Normalize first: Arabic diacritics, letter variants and Arabic-script digits are handled.
EMERGENCY = (
    r"\b(can(?:not|'t|’t) breathe|struggling to breathe|gasping|choking|blue lips|lips.{0,25}(?:blue|grey)|"
    r"severe (?:breathlessness|shortness of breath|chest pain)|crushing chest|unconscious|unresponsive|"
    r"overdos(?:e|ed)|poison(?:ed|ing)|coughing up blood|suicid\w*|kill myself|hurt myself|"
    r"anaphylaxis|severe bleeding|bleeding (?:will not|won't) stop|face drooping|slurred speech|seizures?)\b",
    r"مش قادر.{0,12}(?:اتنفس|اخد نفس)|مش عارف.{0,12}اتنفس|صعوب[هة] شديد[هة].{0,12}التنفس|"
    r"ضيق (?:تنفس|نفس) شديد|(?:شفايف|شفاه|شفت).{0,20}(?:زرق|زرقاء)|اختناق|فاقد الوعي|"
    r"الم شديد.{0,10}الصدر|نزيف شديد|تشنجات|تورم (?:اللسان|الحلق)|جرع[هة] زائد[هة]|انتحار|اقتل نفسي",
    r"\b(mesh|mish) (?:ader|2ader|3aref) (?:atnafas|akhod nafas)\b|\bshafayef\w* zar2\w*\b",
)
DOSING = (
    r"\b(dos(?:e|es|age|ing)|how (?:many|much).{0,45}(?:take|give|tablet|pill|mg|ml)|"
    r"(?:take|give).{0,35}(?:daily|per day|mg|ml)|(?:tablet|pill)s? (?:a|per) day|titrate|titration)\b",
    r"جرع|كام.{0,25}(?:قرص|حبايه|حباية|حبه|حبة|مل|مجم|مر[هة])|اخد كام|كم.{0,20}(?:اتناول|اعطي)|"
    r"\b(?:gor3a|gora3|dose|kam 7aba|kam 2ors|akhod kam|a5od kam)\b",
)
PERSONAL = (
    r"\b(should (?:i|we)|can i (?:take|use|start|stop|combine|switch)|what (?:should i|medicine should)|"
    r"diagnos\w*|prescribe|what should i take|which (?:drug|medicine|treatment).{0,20}(?:me|my)|"
    r"(?:i have|my child has|my symptoms|i am pregnant|i'm pregnant)|treat (?:me|my))\b",
    r"(?:ينفع|ممكن).{0,25}(?:اخد|اخذ|استخدم|استعمل|اوقف|اجمع)|(?:اخد|اخذ|استخدم|استعمل).{0,15}(?:ولا|ايه)|"
    r"(?:عندي|ابني عنده|بنتي عندها|انا حامل|شخصني|تشخيصي|عالجني)|"
    r"\b(?:yenfa3|momken).{0,25}(?:akhod|a5od|astakhdem)|\b3andy\b",
)
DANGEROUS = (
    r"\b(lethal|fatal dose|poison someone|harm someone|drug (?:someone|a person)|"
    r"get high|recreational|without (?:them|their) knowing|hide.{0,15}(?:medicine|drug))\b",
    r"(?:اسَمّم|اسمم|اقتل|اؤذي|اخدر حد|من غير ما يعرف|اتسطّل|اتسطل)|\b(?:atsetel|asmem)\b",
)


def safety_category(question: str) -> str | None:
    text = normalize_text(question)
    for category, patterns in (("emergency", EMERGENCY), ("dosing", DOSING),
                               ("dangerous", DANGEROUS), ("personal", PERSONAL)):
        if any(re.search(pattern, text) for pattern in patterns):
            return category
    return None


def refusal_message(language: str, *, emergency: bool = False, safety: bool = False) -> str:
    if emergency:
        return ("قد تكون هذه حالة طارئة. اتصل بخدمات الطوارئ المحلية الآن واطلب مساعدة طبية عاجلة. لا تنتظر إجابة من الدردشة."
                if language != "en" else
                "This may be an emergency. Contact local emergency services now and seek urgent medical help. Do not wait for a chat response.")
    if safety:
        return ("هذا الطلب يحتاج تقييماً من طبيب أو صيدلي. لا أستطيع تحديد تشخيص أو جرعة أو قرار علاجي شخصي."
                if language != "en" else
                "This request needs a doctor or pharmacist's assessment. I cannot provide a personal diagnosis, medication dose, or treatment decision.")
    return ("لا تتوفر في المقاطع المسترجعة من قاعدة المعرفة معلومات كافية وموثوقة للإجابة عن هذا السؤال."
            if language != "en" else
            "The retrieved knowledge-base passages do not provide enough verified information to answer this question.")
