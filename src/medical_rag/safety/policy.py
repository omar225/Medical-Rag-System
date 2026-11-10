"""Deterministic safety screen, language helpers and fixed user-facing messages.

The screen runs before retrieval and the LLM. It is deliberately small and tuned
for *recall* on red flags: a false emergency costs one extra escalation, a missed
one can cost a life. Language understanding (is this personal advice? is it in
scope?) is left to the LLM; this module only catches what must never wait for it.

Categories:
  emergency  - possible emergency now (breathing, consciousness, bleeding, seizure,
               severe allergy, chest pain, very low sugar, self-harm, overdose/ingestion)
  dangerous  - intent to harm others or misuse medicines
  dosing     - explicit request for a dose/amount/frequency (the KB contains no doses)
"""

import re

from medical_rag.retrieval.normalization import normalize_text


# ---------------------------------------------------------------- language

_ARABIC_WORD = re.compile(r"[ء-ي]+")
_LATIN_WORD = re.compile(r"[a-zA-Z]+")
# Common Franco-Arabic (Arabizi) words; a question using them gets an Arabic answer.
_FRANCO = re.compile(
    r"\b(?:eh|ezay|ezzay|yenfa3|a3rad|ganbeya|ganbia|fa3ala|mada|ashrab|akhod|a5od|akhadt|a5adt|"
    r"bala3t|ko7ol|byetkawen|dawa|7obob|mish|mesh|atnafas|3ayez|3awez|momken|leh|ana|enta|"
    r"el|fel|feeh|bta3|btaa3|emta|mamno3|badayel|ma3a|m3a)\b")


def answer_language(text: str) -> str:
    """Language of a written answer: mostly-Arabic prose with English drug names is "ar"."""
    arabic = len(_ARABIC_WORD.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    if not arabic:
        return "en"
    if not latin or arabic / (arabic + latin) >= 0.4:
        return "ar"
    return "mixed"


def preferred_language(question: str) -> str:
    """Language to answer in: Arabic for Arabic script or Franco-Arabic, else English."""
    if _ARABIC_WORD.search(question):
        return "ar"
    return "ar" if _FRANCO.search(normalize_text(question)) else "en"


# ---------------------------------------------------------------- red flags
# All patterns run on normalize_text(): casefolded, Arabic letter variants unified,
# Arabic-Indic digits converted. Apostrophes may be ' or ’.

_A = r"['’]?"  # optional apostrophe

SELF_HARM = re.compile(
    r"\bsuicid\w*|\bkill(?:ing)? (?:my|him|her|them)sel(?:f|ves)\b|\bend (?:my|his|her|their) (?:own )?life\b|"
    r"\b(?:want|wanna|going|wish) to die\b|\bdon" + _A + r"t want to (?:live|be alive|wake up)\b|"
    r"\bno reason to live\b|\b(?:hurt|harm|cut)(?:ting)? (?:my|him|her)self\b|\bself[- ]harm\w*|"
    r"\boverdos\w* on purpose\b|\btake all (?:my|of my|the) (?:pills|tablets|meds|medicines?)\b|"
    r"انتحار|انتحر|اقتل نفسي|اموت نفسي|(?:عايز|عاوز|نفسي|حابب|هموت) اموت|مش (?:عايز|عاوز) اعيش|"
    r"(?:اذي|ااذي|اجرح) نفسي|\b(?:3ayez|3awez|nefsy) amoo?t\b|\bant[e]?7ar\b"
)

# Current-state danger signs. Exempt only when framed as an information request
# AND no person is involved (e.g. "Is choking a listed side effect?").
DANGER_SIGNS = re.compile(
    # breathing
    r"\b(?:can ?not|can" + _A + r"t|cant|could ?n" + _A + r"t|unable to|struggling to|hard to|barely)\s+"
    r"(?:breathe?|breath(?:ing)?|catch (?:my|his|her|their) breath)\b|"
    r"\b(?:not|stopped|isn" + _A + r"t|wasn" + _A + r"t) breathing\b|\bgasping\b|\bchoking\b|"
    r"\bsevere (?:shortness of breath|breathlessness|difficulty breathing|asthma attack)\b|"
    r"\blips?\b.{0,30}\b(?:blue|grey|gray|purple)\b|\b(?:blue|grey|gray) (?:lips?|face)\b|\bturn(?:ed|ing)? blue\b|"
    # consciousness
    r"\b(?:unconscious|unresponsive|passed out|fainted|collapsed|not waking up|won" + _A + r"t wake up|"
    r"can" + _A + r"t wake (?:him|her|them))\b|"
    # swelling / anaphylaxis
    r"\b(?:throat|tongue|lips?|face|mouth)\b.{0,25}\b(?:swell\w*|swollen|closing|tight)\b|"
    r"\banaphyla\w*|\bsevere allergic reaction\b|"
    # chest pain / stroke
    r"\bchest (?:pain|tightness|pressure)\b|\bcrushing chest\b|\bhaving a (?:heart attack|stroke)\b|"
    r"\bface droop\w*|\bslurred speech\b|\bnumb\w* on one side\b|"
    # bleeding
    r"\b(?:vomit\w*|throw\w* up|cough\w* up|spit\w* up)\s+blood\b|\bblood in (?:my |his |her )?(?:vomit|stool)\b|"
    r"\bblack,? (?:tarry )?stools?\b|\btarry stools?\b|\bsevere bleeding\b|"
    r"\bbleeding (?:won" + _A + r"t|will not|doesn" + _A + r"t|does not) stop\b|"
    # seizure / poisoning / overdose
    r"\b(?:seizures?|seizing|convuls\w*)\b|\bpoison(?:ed|ing)\b|\boverdos(?:e|ed|ing)\b|"
    r"\b(?:very|dangerously) low (?:blood )?(?:sugar|glucose)\b|"
    # Arabic (Egyptian and MSA)
    r"مش (?:قادر|قادره|عارف|عارفه).{0,12}(?:اتنفس|يتنفس|تتنفس|اخد نفس|ياخد نفس|تاخد نفس)|"
    r"مبيتنفسش|مش بيتنفس|مش بتتنفس|بطل يتنفس|بطلت تتنفس|"
    r"(?:ضيق|صعوب[هة]) (?:في )?(?:التنفس|تنفس|النفس|نفس) شديد|ضيق نفس شديد|صعوب[هة] شديد[هة] (?:في )?التنفس|"
    r"بيتخنق|بتتخنق|اختناق|نهجان شديد|"
    r"(?:شفايف|شفاه|شفته|شفتها|شفتي|وشه|وشها|وشي|وجهه)\S*.{0,20}(?:زرق|ازرق|زرقا|زرقاء|ازرقت|ازرقاق)|"
    r"(?:الم|وجع) (?:شديد )?(?:في )?(?:صدري|الصدر|صدره|صدرها)|"
    r"اغمي|اغما عليه|مغمي|فاقد الوعي|فاقده الوعي|فقد الوعي|مش بيفوق|مش بتفوق|مش بيرد|مش بترد|"
    r"تشنجات|تشنج|بيتشنج|بتتشنج|"
    r"تورم (?:في )?(?:اللسان|الحلق|الزور|الوش|الوجه|الشفايف|الشفاه)|(?:زوري|زوره|حلقي|حلقه) (?:بيقفل|مقفول)|"
    r"حساسي[هة] شديد[هة]|"
    r"(?:بيرجع|برجع|بترجع|ترجيع|قيء|استفراغ|بيكح|بكح|كح[هة]) دم|براز (?:اسود|لونه اسود)|"
    r"نزيف (?:شديد|مش بيقف|مبيقفش)|تسمم|اتسمم|جرع[هة] زائد[هة]|"
    # Franco-Arabic
    r"\b(?:mesh|mish) (?:2ader|ader|3aref) (?:atnafas|yetnafes|akhod nafas)\b|\bshafayef\w* zar2\w*|\bet?samem\b"
)

_SUGAR = re.compile(r"(?:\bsugar|\bglucose|السكر|سكره|سكري|سكرها|سكرى)\D{0,12}(\d{2,3})(?!\d)")

_INFO_FRAME = re.compile(
    r"\b(?:side effects?|adverse|symptoms? of|signs? of|warnings?|cautions?|contraindicat\w*|listed|"
    r"according to|source|document|knowledge base|kb|explain|describe|define|definition|caus(?:e|es|ed|ing)|"
    r"what (?:is|are|does|do)|which|is .{1,40} a (?:side effect|symptom|sign|warning))\b|"
    r"المصدر|اعراض جانبي\w*|الاعراض الجانبي\w*|الاثار الجانبي\w*|موانع|تحذيرات|ايه اعراض|ما هي اعراض|اشرح|يعني ايه"
)
_PERSON = re.compile(
    r"\b(?:i|i" + _A + r"m|im|me|my|we|our|us|he|she|his|her|him|they|their|them|patient|child|kid|son|daughter|"
    r"baby|wife|husband|mother|father|mom|mum|dad|friend|someone|somebody|man|woman|boy|girl)\b|"
    r"عندي|عنده|عندها|ابني|بنتي|ابنى|جوزي|مراتي|امي|ابويا|المريض|العيان|عيان|انا|حد|طفل"
)

# --- reported ingestion events (never exempted by framing; hypothetical/negated/remote are not events)

_EVENT = (r"\b(?:took|taken|swallowed|ingested|ate|drank|had|chewed|got into|akhadt|a5adt|akhdet|bala3t|sherebt)\b|"
          r"(?<!\w)(?:اخدت|خدت|اخد|خد|اخذت|تناولت|بلعت|بلع|شربت|شرب|اكلت|اكل)(?!\w)")
_EXCESS = re.compile(
    r"\b(?:too (?:many|much)|way too|a lot of|lots of|loads of|a bunch of|(?:a )?handful|the whole|a whole|"
    r"an entire|all (?:of )?(?:the|my|his|her|their)|bottle|box|pack(?:et)?|strip|blister|extra|double dose|"
    r"more than (?:prescribed|recommended|the)|kteer|keteer|ketir|ziyada|zeyada)\b|"
    r"\b(?:many|several|numerous)\s+(?:[a-z-]+\s+){0,2}(?:pills?|tablets?|tabs|capsules?|caps)\b|"
    r"كتير|كثير|زياد[هة]|زايد[هة]?|زائد[هة]?|كمي[هة] كبير[هة]|علب[هة]|شريط|كل الحبوب|كل الدوا|اكتر من"
)
_UNSURE = re.compile(
    r"\b(?:not sure|unsure|don" + _A + r"t know|do not know|can" + _A + r"t remember|no idea)\b.{0,20}"
    r"\bhow (?:many|much)\b|\bunsure of the amount\b|\bunknown amount\b|\bmesh 3aref el kameya\b|"
    r"(?:مش فاكر|مش عارف|لا اعرف|ولا اعرف|مش متاكد).{0,25}(?:كام|الكمي[هة])"
)
_COUNT = re.compile(
    r"(?:" + _EVENT + r")\D{0,25}?(\d+)\s*(?:x\s*)?(?!(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|"
    r"seconds?|steps|photos?|pictures?|times?|دقيق\w*|ساع\w*|يوم|ايام|سن\w*)\b)(?:[a-zء-ي]+)"
)
_NON_MEDICAL = re.compile(
    r"\b(?:time|food|water|tea|coffee|juice|pizza|pasta|rice|bread|meal|lunch|dinner|breakfast|care|photos?|"
    r"photographs?|pictures?|notes|steps|breaks|effort|money|sleep|wa2t|mayya|akl)\b|"
    r"(?<!\w)(?:وقت|صور|خطوات|اكل|طعام|رز|مياه|مي[هة]|ماء|شاي|قهو[هة]|عصير|عيش)(?!\w)"
)
_MEDICINE = re.compile(
    r"\b(?:pills?|tablets?|tabs|capsules?|caps|medicines?|medications?|meds|drugs?|doses?|syrup|"
    r"7obob|7abayat|dawa|dwa)\b|حبوب|حباي\w*|حبات|اقراص|قرص|كبسول\w*|دواء|الدوا|دوا|شراب"
)
_CHILD_INGESTION = re.compile(
    r"\b(?:child|kid|son|daughter|baby|toddler|boy|girl)\b.{0,40}\b(?:swallowed|ate|took|drank|got into|chewed)\b|"
    r"(?:ابني|بنتي|ابنى|الطفل|طفل|العيل|البيبي|ابنها|ابنه|بنته|بنتها).{0,30}(?:بلع|بلعت|اكل|اكلت|خد|خدت|شرب|شربت)"
)
_NEGATED = re.compile(
    r"\b(?:did not|didn" + _A + r"t|have not|haven" + _A + r"t|has not|hasn" + _A + r"t|never|not)\s+"
    r"(?:take|took|taken|swallow|swallowed|ingest|ingested|overdose|overdosed|eat|drink)\b|\bno overdose\b|"
    r"(?:ما|مش)\s*(?:اخدتش|اخدش|خدتش|بلعتش|شربتش)|لم\s+(?:اتناول|ابتلع|ياخذ|يتناول|تتناول)|"
    r"\bma (?:akhadtesh|a5adtesh|bala3tesh)\b"
)
_HYPOTHETICAL = re.compile(
    r"^(?:(?:(?:can you )?explain )?what (?:would happen|happens|happened|could happen) if|if\b|hypothetically|"
    r"suppose|imagine|in case|لو\b|ماذا (?:قد )?يحدث (?:لو|اذا)|law\b)"
)
_REMOTE_RESOLVED = re.compile(
    r"(?:years ago|as a child|long ago|سنين|سنوات|men seneen).{0,160}"
    r"(?:recovered|hospital care|treated|تعافيت|اتعالجت|et3alegt)"
)
_CLAUSES = re.compile(r"[.!?؟;\n]+|\bbut\b|لكن|بس(?!\w)")


def _reported_ingestion(text: str) -> bool:
    if _CHILD_INGESTION.search(text) and _MEDICINE.search(text) and not _NEGATED.search(text):
        return True
    for clause in _CLAUSES.split(text):
        clause = clause.strip(" ,،")
        if (not clause or _HYPOTHETICAL.search(clause) or _REMOTE_RESOLVED.search(clause)
                or _NEGATED.search(clause) or not re.search(_EVENT, clause)):
            continue
        if _UNSURE.search(clause):
            return True
        for count in _COUNT.finditer(clause):
            if int(count.group(1)) >= 5:
                return True
        # Each "and"/"و" part is its own exposure: "drank water and swallowed pills".
        for part in re.split(r"\band\b|\bw\b|(?<!\w)و(?=\s)", clause):
            if re.search(_EVENT, part) and _EXCESS.search(part) and not _NON_MEDICAL.search(part):
                return True
    return False


def _danger_sign(text: str) -> bool:
    found = DANGER_SIGNS.search(text)
    sugar = any(int(value) < 55 for value in _SUGAR.findall(text))
    if not found and not sugar:
        return False
    if found and not sugar:
        educational = _INFO_FRAME.search(text) and not _PERSON.search(text)
        hypothetical = _HYPOTHETICAL.search(text.strip()) and re.match(r"overdos", found.group())
        if educational or hypothetical or (re.match(r"overdos", found.group()) and _NEGATED.search(text)):
            return False
    return True


DANGEROUS = re.compile(
    r"\b(?:lethal|fatal) (?:dose|amount)\b|\bpoison (?:someone|somebody|him|her|them|my|a person)\b|"
    r"\b(?:harm|hurt|kill) (?:someone|somebody|him|her|them|my (?:wife|husband|mother|father|boss))\b|"
    r"\bdrug (?:someone|somebody|a person|my)\b|\bget(?:ting)? high\b|\brecreational\b|"
    r"\bwithout (?:them|him|her|their|his) (?:knowing|knowledge)\b|"
    r"اسمم|اسَمّم|اقتل (?:حد|جوزي|مراتي)|اؤذي حد|اخدر حد|من غير ما (?:يعرف|تعرف)|اتسطل|\b(?:atsetel|asmem)\b"
)

DOSING = re.compile(
    r"\bhow (?:many|much)\b.{0,40}\b(?:mg|mcg|ml|milligrams?|grams?|units?|tablets?|pills?|capsules?|puffs?|"
    r"drops?|doses?|times)\b|\bhow (?:many|much)\b.{0,30}\b(?:take|give|use|administer|have)\b|"
    r"(?<!high )(?<!double )(?<!over)\bdos(?:e|es|age|ages|ing)\b|\bmaximum (?:daily )?(?:amount|dose)\b|"
    r"\b(?:mg|mcg|ml)\s*(?:per|a|/)\s*(?:day|kg)\b|\bhow (?:often|frequently)\b.{0,30}\b(?:take|give|use)\b|"
    r"\btitrat\w*|"
    r"(?:جرع[هة]|الجرع[هة]|جرعات)(?! زائد| زايد)|كام (?:قرص|حباي[هة]|حب[هة]|مجم|ملي|مل|مر[هة]|بخ[هة]|نقط[هة])|"
    r"(?:اخد|ياخد|تاخد|اديله|اديها|اديهم) كام|كم (?:قرص|حب[هة]|مجم|مر[هة])|"
    r"\b(?:gor3a|gor3et|kam 7aba\w*|kam 2ors|akhod kam|a5od kam)\b"
)


def safety_category(message: str) -> str | None:
    """Return "emergency", "dangerous", "dosing" or None. Order matters: emergencies first."""
    text = normalize_text(message)
    if SELF_HARM.search(text) or _reported_ingestion(text) or _danger_sign(text):
        return "emergency"
    if DANGEROUS.search(text):
        return "dangerous"
    if DOSING.search(text):
        return "dosing"
    return None


# ---------------------------------------------------------------- messages

MESSAGES = {
    "emergency": {
        "en": "This may be an emergency. Contact local emergency services now (in Egypt call 123) and get urgent "
              "medical help. Do not wait for a chat response. If someone may have taken an overdose, bring the "
              "medicine packaging with you.",
        "ar": "قد تكون هذه حالة طارئة. اتصل بالإسعاف فوراً (في مصر 123) واطلب مساعدة طبية عاجلة. لا تنتظر رد المحادثة. "
              "لو في احتمال جرعة زائدة، خُد علبة الدواء معاك.",
    },
    "dosing": {
        "en": "The approved knowledge base does not provide doses, and I can't recommend one. Please check the "
              "approved product leaflet or ask the prescribing doctor or pharmacist.",
        "ar": "قاعدة المعرفة المعتمدة لا تحتوي على جرعات، ولا أستطيع تحديد جرعة. من فضلك راجع النشرة المعتمدة للمستحضر "
              "أو اسأل الطبيب المعالج أو الصيدلي.",
    },
    "clinician": {
        "en": "This needs an individual clinical decision, which I can't make. Please consult the treating doctor "
              "or pharmacist. I can share what the approved knowledge base says about a specific medicine.",
        "ar": "ده محتاج قرار طبي فردي ومقدرش أاخده. من فضلك ارجع للطبيب المعالج أو الصيدلي. أقدر أقولك قاعدة المعرفة "
              "المعتمدة بتقول إيه عن دواء معين.",
    },
    "dangerous": {
        "en": "I can't help with that. If someone is in danger, contact local emergency services now (in Egypt call 123).",
        "ar": "مقدرش أساعد في ده. لو في حد في خطر، اتصل بالإسعاف فوراً (في مصر 123).",
    },
    "out_of_scope": {
        "en": "I can only answer questions covered by the approved drug knowledge base, and it doesn't cover this.",
        "ar": "أقدر أجاوب بس على الأسئلة اللي قاعدة المعرفة المعتمدة للأدوية بتغطيها، وهي مش بتغطي السؤال ده.",
    },
    "unverified": {
        "en": "I couldn't verify an answer to this against the approved knowledge base. Please rephrase or ask "
              "about a specific medicine.",
        "ar": "مقدرتش أتحقق من إجابة للسؤال ده من قاعدة المعرفة المعتمدة. ممكن تعيد صياغة السؤال أو تسأل عن دواء معين؟",
    },
}


def message(kind: str, language: str) -> str:
    return MESSAGES[kind]["ar" if language != "en" else "en"]
