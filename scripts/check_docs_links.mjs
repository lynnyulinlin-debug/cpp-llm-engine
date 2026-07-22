#!/usr/bin/env node

import { spawnSync } from 'node:child_process'

const result = spawnSync('npm', ['run', 'docs:build'], {
  stdio: 'inherit',
})

if (result.error) {
  console.error(result.error.message)
  process.exit(1)
}

process.exit(result.status ?? 1)
