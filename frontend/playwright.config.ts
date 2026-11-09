import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests', testMatch: '**/*.spec.ts', testIgnore: '**/*.live.spec.ts',
  fullyParallel: false, workers: 1, retries: 0,
  outputDir: 'test-results/offline',
  reporter: [['list'], ['json', { outputFile: '../reports/frontend/browser-tests.json' }]],
  use: { baseURL: 'http://127.0.0.1:5173', channel: process.env.BROWSER_CHANNEL || 'chrome',
    headless: true, screenshot: 'only-on-failure', trace: 'retain-on-failure', viewport: { width: 1440, height: 1000 } },
  webServer: { command: 'npm run dev', url: 'http://127.0.0.1:5173', reuseExistingServer: true, timeout: 30_000 },
})
