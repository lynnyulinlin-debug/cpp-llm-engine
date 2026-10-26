# Summary Template 章节小结兼容页

中文主线不再把 `summary.md` 作为独立学习页面。章节结论、验证和下一步统一写在 `intro.md`，保证学习者只有一个章节入口。

迁移期间保留旧文件，避免历史链接失效。兼容页只保留标题和一个指向 `intro.md` 的链接，不重新复制正文内容。

## 写作约束

- 不在兼容页重复 intro、框架对照表或源码入口
- 不在兼容页新增背景、推导、复盘或下一步导航
- 需要详细系统说明时链接到 `reference/`，不在兼容页展开

## Checks

- `npm run docs:check-links`
- `npm run docs:build`
- `git diff --check`
