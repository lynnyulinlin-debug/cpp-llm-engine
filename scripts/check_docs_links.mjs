#!/usr/bin/env node

import { spawnSync } from 'node:child_process'
import { existsSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs'
import { dirname, extname, join, resolve } from 'node:path'

const distDir = 'docs/.vitepress/dist'
const sourceDir = 'source'

rmSync(distDir, { recursive: true, force: true })

const result = spawnSync('npm', ['run', 'docs:build'], {
  stdio: 'inherit',
})

if (result.error) {
  console.error(result.error.message)
  process.exit(1)
}

if ((result.status ?? 1) !== 0) {
  process.exit(result.status ?? 1)
}

const htmlFiles = []

function walk(dir) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) {
      walk(path)
    } else if (path.endsWith('.html')) {
      htmlFiles.push(path)
    }
  }
}

const sourceMarkdownFiles = []

function walkMarkdown(dir) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) {
      walkMarkdown(path)
    } else if (path.endsWith('.md')) {
      sourceMarkdownFiles.push(path)
    }
  }
}

walkMarkdown(sourceDir)

const sourceMissing = []
const markdownImagePattern = /!\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)/g

for (const file of sourceMarkdownFiles) {
  const markdown = readFileSync(file, 'utf8')
  for (const match of markdown.matchAll(markdownImagePattern)) {
    const image = match[1]
    if (/^(https?:|data:)/.test(image)) continue

    const target = resolve(dirname(file), image)
    if (!existsSync(target)) {
      sourceMissing.push(`${file}: image ${image} -> ${target}`)
    }
  }
}

if (sourceMissing.length > 0) {
  console.error('\nBroken source Markdown images:')
  for (const item of sourceMissing) {
    console.error(`- ${item}`)
  }
  process.exit(1)
}

function hrefToFile(href, fromFile) {
  if (!href || href.startsWith('#')) return null
  if (/^(https?:|mailto:|tel:|data:)/.test(href)) return null

  const [rawPath] = href.split('#')
  if (!rawPath || rawPath.startsWith('javascript:')) return null

  const normalized = decodeURIComponent(rawPath)

  const basePath = normalized.startsWith('/')
    ? join(distDir, normalized.slice(1))
    : resolve(dirname(fromFile), normalized)

  if (normalized === '/') return join(distDir, 'index.html')
  if (normalized.endsWith('/')) return join(basePath, 'index.html')
  if (extname(basePath)) return basePath
  return `${basePath}.html`
}

walk(distDir)

const missing = []
const hrefPattern = /\bhref="([^"]+)"/g
const imagePattern = /\bsrc="([^"]+)"/g

for (const file of htmlFiles) {
  const html = readFileSync(file, 'utf8')
  for (const match of html.matchAll(hrefPattern)) {
    const target = hrefToFile(match[1], file)
    if (target && !existsSync(target)) {
      missing.push(`${file}: ${match[1]} -> ${target}`)
    }
  }
  for (const match of html.matchAll(imagePattern)) {
    const target = hrefToFile(match[1], file)
    if (target && !existsSync(target)) {
      missing.push(`${file}: image ${match[1]} -> ${target}`)
    }
  }
}

if (missing.length > 0) {
  console.error('\nBroken generated internal links:')
  for (const item of missing) {
    console.error(`- ${item}`)
  }
  process.exit(1)
}

console.log(
  `Checked ${sourceMarkdownFiles.length} source Markdown files and ${htmlFiles.length} generated HTML files for internal links.`,
)
