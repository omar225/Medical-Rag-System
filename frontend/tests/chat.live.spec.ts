import { expect, test } from '@playwright/test'
import type { ChatResponse } from '../src/api'
import AxeBuilder from '@axe-core/playwright'

test('real greetings, a typo, and mixed and Arabic questions', async ({ page }) => {
  test.setTimeout(120_000)
  await page.goto('/')
  const observations: object[] = []
  for (const [question, status] of [
    ['goodmorning', 'out_of_scope'],
    ['can i ask you a question', 'out_of_scope'],
    ['goodmorning. i need to ask a question about metformin', 'out_of_scope'],
    ['What is the active ingredient in Ibprofen?', 'answered'],
    ['Metformin ممنوع امتى؟', 'answered'],
    ['ما المادة الفعالة في لوسيك؟', 'answered'],
  ]) {
    await page.getByLabel('Your medical question').fill(question)
    const received = page.waitForResponse(r => r.url().endsWith('/api/chat') && r.request().method() === 'POST')
    const started = Date.now()
    await page.getByRole('button', { name: 'Send question', exact: true }).click()
    const response = await received
    expect(response.status()).toBe(200)
    const data: ChatResponse = await response.json()
    expect(data.status).toBe(status)
    await expect(page.locator('.answer-text').last()).toBeVisible()
    expect(await page.locator('.answer-text').last().textContent()).toBe(data.answer)
    if (question.startsWith('Metformin')) {
      expect(data.citations.length).toBeGreaterThan(0)
      await page.locator('summary').last().click()
      await page.setViewportSize({ width: 390, height: 844 })
      await page.getByRole('button', { name: 'Switch interface to Arabic' }).click()
      await page.locator('.turn').last().scrollIntoViewIfNeeded()
      await page.screenshot({ path: '../reports/frontend/screenshots/arabic-mobile-citations.png' })
      expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
      await page.getByRole('button', { name: 'Switch interface to English' }).click()
    }
    if (status === 'out_of_scope') await expect(page.locator('.answer-status').last()).toHaveText('Response')
    observations.push({ question, response: data, seconds: (Date.now() - started) / 1000 })
  }
  await test.info().attach('real-responses', { body: JSON.stringify(observations, null, 2), contentType: 'application/json' })
})

test('real FastAPI proxy integration: supported English, Arabic, and Franco questions, abstention, and escalation', async ({ page, request }) => {
  const health = await request.get('/api/health')
  expect(health.status()).toBe(200)
  expect(await health.json()).toMatchObject({ status: 'ok' })
  await page.goto('/')
  const cases = [
    { question: 'What is the active ingredient in Cataflam?', status: 'answered' },
    { question: 'ما المادة الفعالة في كتافلام؟', status: 'answered' },
    { question: 'eh el mada el fa3ala fe Cataflam?', status: 'answered' },
    { question: 'Which drugs are used to treat malaria?', status: 'out_of_scope' },
    { question: 'عيان عنده ضيق نفس شديد وشفايفه زرقا، أجيله Ventolin ؟', status: 'escalated' },
  ]
  const observations: object[] = []
  for (const [index, item] of cases.entries()) {
    const started = Date.now()
    await page.getByLabel('Your medical question').fill(item.question)
    const received = page.waitForResponse(response => response.url().endsWith('/api/chat') && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Send question', exact: true }).click()
    const response = await received
    expect(response.status()).toBe(200)
    const data: ChatResponse = await response.json()
    expect(data.status).toBe(item.status)
    const answer = page.locator('.answer-text').nth(index)
    await expect(answer).toBeVisible()
    expect(await answer.textContent()).toBe(data.answer)
    if (data.status === 'answered') {
      await expect(page.locator('.sources').last()).toBeVisible()
      await page.locator('summary').last().click()
      await expect(page.locator('.source-document').last()).toContainText(data.citations[0].doc_id)
    }
    observations.push({ question: item.question, response: data, browserSeconds: (Date.now() - started) / 1000 })
    if (index === 0) await page.screenshot({ path: '../reports/frontend/screenshots/live-answer-desktop.png', fullPage: true })
    if (index === 1) {
      await page.setViewportSize({ width: 390, height: 844 })
      await page.getByRole('button', { name: 'Switch interface to Arabic' }).click()
      await page.locator('.turn').last().scrollIntoViewIfNeeded()
      await page.screenshot({ path: '../reports/frontend/screenshots/live-arabic-mobile-answer.png' })
      await page.getByRole('button', { name: 'Switch interface to English' }).click()
      await page.setViewportSize({ width: 1440, height: 1000 })
    }
  }
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: 'Switch interface to Arabic' }).click()
  await page.locator('.assistant-message--escalated').scrollIntoViewIfNeeded()
  await page.screenshot({ path: '../reports/frontend/screenshots/live-arabic-mobile.png' })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await test.info().attach('real-chat-responses', { body: JSON.stringify(observations, null, 2), contentType: 'application/json' })
})
