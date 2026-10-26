#!/usr/bin/env node

import { existsSync, lstatSync, readdirSync, readFileSync } from 'node:fs'
import { extname, join } from 'node:path'

const roots = ['README.md', 'docs', 'source', 'exercises']
const skippedDirectories = new Set(['.vitepress', 'build', 'node_modules'])
const tableSeparator = /^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$/
const numberedLearningTask = /(?:\btask\s*\d+\b|任务\s*[0-9一二三四五六七八九十]+)/i
const chapterSource = /^source\/chapter(?:0[0-9])_[^/]+\/.+\.md$/
const maintainerOnlyLink = /(?:source\/maintenance\/|(?:^|[.]{0,2}\/+)maintenance\/)/
const maintainerOnlyStatus = /(?:当前就绪度|后续补强顺序|审核结论与补强顺序|English coverage status)/i

function isMaintainerFile(file) {
  return file.startsWith('source/maintenance/')
}

function markdownFiles(path) {
  if (!existsSync(path)) return []

  const entry = lstatSync(path)
  if (entry.isSymbolicLink()) return []
  if (entry.isFile()) return extname(path) === '.md' ? [path] : []

  const files = []
  for (const child of readdirSync(path, { withFileTypes: true })) {
    if (child.isDirectory() && skippedDirectories.has(child.name)) continue
    files.push(...markdownFiles(join(path, child.name)))
  }
  return files
}

const findings = []
let checkedFiles = 0

for (const root of roots) {
  for (const file of markdownFiles(root)) {
    checkedFiles += 1
    const lines = readFileSync(file, 'utf8').split(/\r?\n/)
    for (let index = 0; index < lines.length; index += 1) {
      if (chapterSource.test(file) && numberedLearningTask.test(lines[index])) {
        findings.push(`${file}:${index + 1}: chapter content must not depend on a numbered learning task`)
      }
      if (!isMaintainerFile(file) && maintainerOnlyLink.test(lines[index])) {
        findings.push(`${file}:${index + 1}: learner content must not link to maintainer-only documentation`)
      }
      if (!isMaintainerFile(file) && maintainerOnlyStatus.test(lines[index])) {
        findings.push(`${file}:${index + 1}: project maintenance status must not appear in learner content`)
      }

      if (!tableSeparator.test(lines[index])) continue

      const header = lines[index - 1] ?? ''
      const firstRow = lines[index + 1] ?? ''
      if (!header.trimStart().startsWith('|')) {
        findings.push(`${file}:${index + 1}: table separator has no header row`)
      }
      if (!firstRow.trimStart().startsWith('|')) {
        findings.push(`${file}:${index + 2}: table body must immediately follow its separator`)
      }
    }
  }
}

if (findings.length > 0) {
  console.error('Documentation structure check failed:')
  for (const finding of findings) console.error(`- ${finding}`)
  process.exit(1)
}

console.log(`Documentation structure check passed: ${checkedFiles} Markdown files checked.`)
