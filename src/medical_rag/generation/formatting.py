"""Presentation of already verified facts; no clinical inference or translation."""
import re


def format_verified_excerpts(checks, question, excerpts, language):
    if checks.intent(question) == 'ingredient':
        subjects = checks.subjects(question)
        for line in excerpts.splitlines():
            if '|' not in line:
                continue
            name, value = (part.strip() for part in line.split('|', 1))
            if subjects == checks.words(name):
                return (f'The active ingredient in {name} is {value}.' if language == 'en'
                        else f'المادة الفعالة في {name} هي {value}.')
            if '+' not in value and subjects == checks.words(value):
                return (f'{value} is itself an active ingredient. The source lists {name} as a brand containing {value}.'
                        if language == 'en' else
                        f'{value} هو نفسه مادة فعالة. المصدر بيذكر {name} كاسم تجاري يحتوي على {value}.')
    lines = list(dict.fromkeys(line.strip() for line in excerpts.splitlines() if line.strip()))
    if checks.intent(question) == 'brands':
        prefix = ('The source lists these brands with the same active ingredient. This is not a recommendation to switch medicines:'
                  if language == 'en' else
                  'المصدر بيذكر الأسماء دي بنفس المادة الفعالة. دي قائمة من المصدر، مش توصية بتبديل الدواء:')
        return prefix + '\n\n' + '\n'.join('• ' + line for line in lines)
    prefix = ('The source lists:' if language == 'en' else
              'المصدر بيذكر الآتي (المصطلحات الطبية محفوظة زي ما وردت فيه):')
    return prefix + '\n\n' + '\n'.join('• ' + line for line in lines)


def format_verified_answer(answer):
    """Only add visible list markers; never rewrite verified medical wording."""
    lines = answer.splitlines()
    if len([line for line in lines if line.strip()]) < 3:
        return answer
    return '\n'.join(('• ' + line if line.strip() and not re.match(r'\s*(?:[-•*]|\d+[.)])', line)
                       and index > 0 and not line.rstrip().endswith(':') else line)
                     for index, line in enumerate(lines))
