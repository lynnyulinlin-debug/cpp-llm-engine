#!/usr/bin/env node

import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'

const roots = ['source/chapter00_quickstart', 'source/chapter01_tensor_loader',
  'source/chapter02_tensor_ops', 'source/chapter03_matmul', 'source/chapter04_attention',
  'source/chapter05_generation', 'source/chapter06_quantization', 'source/chapter07_benchmark',
  'source/chapter08_profiling', 'source/foundation', 'source/reference']
const patterns = [
  /##\s+这一页要解决什么/, /##\s+本节概览/, /这一页/, /本页/, /这一层/, /本层/
]

function walk(dir) {
  const files = []
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) files.push(...walk(path))
    else if (path.endsWith('.md')) files.push(path)
  }
  return files
}

const findings = []
for (const root of roots) {
  for (const file of walk(root)) {
    const lines = readFileSync(file, 'utf8').split(/\r?\n/)
    lines.forEach((line, index) => {
      if (patterns.some((pattern) => pattern.test(line))) {
        findings.push(`${file}:${index + 1}: ${line.trim()}`)
      }
    })
  }
}

if (findings.length > 0) {
  console.warn('Documentation style findings (non-blocking):')
  findings.forEach((finding) => console.warn(`- ${finding}`))
} else {
  console.log('Documentation style check passed: no generic page/layer references found.')
}

process.exit(0)
