import { defineConfig } from '@playwright/test'
import config from './playwright.config'

export default defineConfig({ ...config, testIgnore: [], testMatch: '**/*.live.spec.ts', timeout: 60_000,
  outputDir: 'test-results/live',
  reporter: [['list'], ['json', { outputFile: '../reports/frontend/live-browser-tests.json' }]],
})
