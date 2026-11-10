import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

// Synthetic presentation fixtures are isolated to browser tests. No runtime mock path.
const answered = {
  answer: 'Synthetic evidence-backed answer for interface testing.', status: 'answered', language: 'en',
  citations: [{ doc_id: 'TEST-01', title: 'Synthetic source document', section: 'Synthetic section', snippet: 'An exact synthetic test excerpt.' }],
}
const screenshotDir = '../reports/frontend/screenshots'

test('friendly replies and clarifications have a neutral heading while scope limits remain explicit', async ({ page }) => {
  await page.goto('/')
  for (const answer of ['Hello! What would you like to know?', 'Please confirm the medicine name.',
    'The retrieved knowledge-base passages do not provide enough verified information to answer this question.']) {
    await mock(page, { ...answered, status: 'out_of_scope', citations: [], answer })
    await submit(page, 'Synthetic conversation request')
    await expect(page.locator('.answer-status').last()).toHaveText('Response')
    await expect(page.locator('.answer-text').last()).toHaveText(answer)
    await expect(page.locator('.sources')).toHaveCount(0)
  }
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
})

test('source-only fallback preserves bullet lines, paragraph spacing, and deduplicated source count', async ({ page }) => {
  const answer = 'المصدر بيذكر الآتي:\n\n• Severe renal impairment: eGFR أقل من 30.\n• Metabolic acidosis.'
  await mock(page, { ...answered, answer, language: 'mixed' })
  await page.goto('/')
  await submit(page)
  await expect(page.locator('.answer-text')).toBeVisible()
  expect(await page.locator('.answer-text').textContent()).toBe(answer)
  await expect(page.locator('.answer-text .bidi-line')).toHaveCount(4)
  await expect(page.locator('.source-count')).toHaveText('1')
  expect(await page.locator('.answer-text .bidi-line').nth(1).evaluate(el => el.getBoundingClientRect().height)).toBeGreaterThan(0)
  await page.setViewportSize({ width: 375, height: 812 })
  await noOverflow(page)
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
})

async function mock(page: Page, response = answered) {
  await page.route('**/api/chat', route => route.fulfill({ status: 200, json: response }))
}
async function submit(page: Page, question = 'A synthetic medical question') {
  await page.getByLabel('Your medical question').fill(question)
  await page.getByRole('button', { name: 'Send question', exact: true }).click()
}
async function noOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
}

test('starter question fills the composer without sending, and whitespace is validated', async ({ page }) => {
  let requests = 0
  await page.route('**/api/chat', async route => { requests++; await route.fulfill({ json: answered }) })
  await page.goto('/')
  await page.getByRole('button', { name: 'Send question' }).click()
  await expect(page.getByLabel('Your medical question')).toBeFocused()
  await expect(page.getByText('Write a question before sending.')).toBeVisible()
  await page.getByRole('button', { name: /Ingredients.*Cataflam/ }).click()
  await expect(page.getByLabel('Your medical question')).toHaveValue('What is the active ingredient in Cataflam?')
  expect(requests).toBe(0)
})

test('exact API request, genuine metadata, expandable citations, and text-only rendering', async ({ page }) => {
  const html = '<script>window.unsafe = true</script>'
  await page.route('**/api/chat', async route => {
    expect(route.request().postDataJSON()).toEqual({ message: 'Test subject', session_id: expect.any(String) })
    await route.fulfill({ json: { ...answered, answer: html } })
  })
  await page.goto('/')
  await submit(page, 'Test subject')
  await expect(page.locator('.answer-text')).toHaveText(html)
  await expect(page.getByText('From the knowledge base')).toBeVisible()
  await expect(page.getByText('Synthetic source document')).not.toBeVisible()
  await page.locator('summary').click()
  await expect(page.getByText('Synthetic source document')).toBeVisible()
  await expect(page.getByText('TEST-01', { exact: true })).toBeVisible()
  await expect(page.getByText('An exact synthetic test excerpt.')).toBeVisible()
  expect(await page.evaluate(() => 'unsafe' in window)).toBe(false)
})

test('loading blocks rapid duplicate submits; Shift+Enter and IME Enter do not send', async ({ page }) => {
  let requests = 0
  let release!: () => void
  const ready = new Promise<void>(resolve => { release = resolve })
  await page.route('**/api/chat', async route => { requests++; await ready; await route.fulfill({ json: answered }) })
  await page.goto('/')
  const input = page.getByLabel('Your medical question')
  await input.fill('Synthetic question')
  await input.press('Shift+Enter')
  await input.dispatchEvent('keydown', { key: 'Enter', isComposing: true })
  expect(requests).toBe(0)
  await input.press('Enter')
  await expect(page.getByRole('button', { name: 'Reviewing your question…' })).toBeDisabled()
  await input.fill('Another question being drafted')
  await input.press('Enter')
  await expect(page.locator('.pending-response')).toBeVisible()
  expect(requests).toBe(1)
  release()
  await expect(page.locator('.answer-text')).toBeVisible()
  await expect(input).toHaveValue('Another question being drafted')
})

for (const status of ['out_of_scope', 'escalated'] as const) {
  test(`${status} preserves exact backend text, never collapses instructions, and shows no citations`, async ({ page }) => {
    const answer = status === 'escalated' ? 'Synthetic urgent instructions.\nSecond instruction, always visible.' : 'Synthetic source insufficiency notice.'
    await mock(page, { ...answered, status, answer, citations: [] })
    await page.goto('/')
    await submit(page)
    await expect(page.locator('.answer-text')).toHaveText(answer)
    await expect(page.locator('details')).toHaveCount(0)
    if (status === 'escalated') {
      await expect(page.getByText('Professional or urgent help needed')).toBeVisible()
      await expect(page.locator('.assistant-message--escalated')).toBeVisible()
    }
  })
}

for (const status of [422, 500, 502]) {
  test(`HTTP ${status} is a technical error and retry does not duplicate the question`, async ({ page }) => {
    let calls = 0
    await page.route('**/api/chat', route => {
      calls++
      return route.fulfill(calls === 1 ? { status, json: { error: 'Synthetic server failure' } } : { status: 200, json: answered })
    })
    await page.goto('/')
    await submit(page)
    await expect(page.getByRole('alert')).toBeVisible()
    await expect(page.locator('.answer-status')).toHaveCount(0)
    await page.getByRole('button', { name: 'Try again' }).click()
    await expect(page.locator('.answer-text')).toBeVisible()
    await expect(page.locator('.user-message')).toHaveCount(1)
    expect(calls).toBe(2)
  })
}

test('network failure and malformed successful responses cannot become medical answers', async ({ page }) => {
  await page.route('**/api/chat', route => route.abort('failed'))
  await page.goto('/')
  await submit(page)
  await expect(page.getByText(/Check your connection/)).toBeVisible()
  await page.unroute('**/api/chat')
  await page.route('**/api/chat', route => route.fulfill({ json: { ...answered, citations: [] } }))
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByText(/unexpected response/)).toBeVisible()
  await expect(page.locator('.answer-text')).toHaveCount(0)
})

test('request timeout is reported as a technical failure', async ({ page }) => {
  await page.route('**/api/chat', () => {})
  await page.goto('/')
  await page.clock.install()
  await submit(page)
  await page.clock.fastForward(40_100)
  await expect(page.getByText('The service took too long to respond. Please try again.')).toBeVisible()
})

test('new conversation confirms clearing, aborts pending work, and ignores late responses', async ({ page }) => {
  let release!: () => void
  const ready = new Promise<void>(resolve => { release = resolve })
  await page.route('**/api/chat', async route => { await ready; await route.fulfill({ json: answered }).catch(() => {}) })
  await page.goto('/')
  await submit(page)
  await page.getByRole('button', { name: 'New conversation', exact: true }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.locator('.user-message')).toHaveCount(1)
  await page.getByRole('button', { name: 'New conversation', exact: true }).click()
  await page.getByRole('button', { name: 'Start new conversation', exact: true }).click()
  release()
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Understand the medicine.\nRead the evidence.')
  await expect(page.locator('.answer-text')).toHaveCount(0)
  await expect(page.getByLabel('Your medical question')).toBeFocused()
})

test('Arabic interface and mixed answer retain text order, isolated medicine names, and keyboard source disclosure', async ({ page }) => {
  const answer = 'هذا نص اختباري يحتوي على Metformin و Cataflam.\nThe second line is English.'
  await mock(page, { ...answered, answer, language: 'mixed', citations: [{ ...answered.citations[0], section: '1. English topic – عنوان عربي > 2. Non-Steroidal Anti-Inflammatory Drugs' }] })
  await page.goto('/')
  await page.getByRole('button', { name: 'Switch interface to Arabic' }).click()
  await expect(page.locator('html')).toHaveAttribute('dir', 'rtl')
  await expect(page.locator('html')).toHaveAttribute('lang', 'ar')
  await page.getByLabel('سؤالك الطبي').fill('ايه المادة الفعالة في Cataflam؟')
  await page.getByRole('button', { name: 'إرسال السؤال', exact: true }).click()
  expect(await page.locator('.answer-text').textContent()).toBe(answer)
  await expect(page.locator('.answer-text bdi').filter({ hasText: 'Metformin' })).toHaveAttribute('dir', 'ltr')
  await page.locator('summary').focus()
  await page.keyboard.press('Enter')
  await expect(page.getByText('Synthetic source document')).toBeVisible()
  await expect(page.locator('.answer-text .bidi-line').nth(0)).toHaveAttribute('dir', 'rtl')
  await expect(page.locator('.answer-text .bidi-line').nth(1)).toHaveAttribute('dir', 'ltr')
  await expect(page.locator('.source-section .bidi-line')).toHaveAttribute('dir', 'ltr')
  await expect(page.locator('.source-section bdi').filter({ hasText: 'Non-Steroidal Anti-Inflammatory Drugs' })).toHaveAttribute('dir', 'ltr')
  await noOverflow(page)
})

test('copy copies the exact answer and reload clears local conversation', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await mock(page)
  await page.goto('/')
  await submit(page)
  await page.getByRole('button', { name: 'Copy answer' }).click()
  await expect(page.getByRole('button', { name: 'Copied', exact: true })).toBeVisible()
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(answered.answer)
  await page.reload()
  await expect(page.locator('.user-message')).toHaveCount(0)
})

test('skip link, visible focus, and reduced-motion feedback remain usable', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/')
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: 'Skip to conversation' })).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.locator('#workspace')).toBeFocused()
  await page.keyboard.press('Tab')
  expect(await page.locator(':focus').evaluate(element => getComputedStyle(element).outlineStyle)).toBe('solid')
  expect(await page.locator('.example').first().evaluate(element => getComputedStyle(element).transitionDuration)).toBe('0s')
})

for (const width of [320, 375, 768, 1024, 1440]) {
  test(`responsive and accessible empty/answered/escalated layouts at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: width < 768 ? 900 : 1000 })
    await page.goto('/')
    await page.evaluate(() => document.fonts.ready)
    await noOverflow(page)
    const check = async () => {
      const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa']).analyze()
      expect(result.violations).toEqual([])
    }
    await check()
    if (width === 1440 || width === 375) await page.screenshot({ path: `${screenshotDir}/empty-${width}.png`, fullPage: true })
    const longAnswer = `${'Synthetic long answer for responsive testing. '.repeat(100)}\n${'اختبار عرض نص عربي طويل. '.repeat(30)}\n${'LongUnbrokenText'.repeat(45)}`
    await mock(page, { ...answered, answer: longAnswer, language: 'mixed' })
    await submit(page)
    await page.locator('summary').click()
    await noOverflow(page)
    await check()
    if (width === 375) await page.screenshot({ path: `${screenshotDir}/long-answer-mobile-test.png`, fullPage: true })
    await page.unroute('**/api/chat')
    await mock(page, { ...answered, status: 'escalated', citations: [], answer: 'Synthetic urgent guidance. This text must remain visible.' })
    await submit(page, 'Synthetic emergency question')
    await check()
    await noOverflow(page)
    await page.getByRole('button', { name: 'Switch interface to Arabic' }).click()
    await check()
    await noOverflow(page)
  })
}
