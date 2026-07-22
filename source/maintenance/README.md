# Maintenance 维护文档

这里放的是教程维护者和贡献者需要关心的规范、模板和检查标准。普通读者通常不需要看这里。

## 内容范围

- [tutorial_maintenance.md](/maintenance/tutorial_maintenance)：教程维护文档
- [tutorial_writing_guidelines.md](/maintenance/tutorial_writing_guidelines)：教程编写规范
- [chapter_template.md](/maintenance/chapter_template)：章节模板
- [summary_template.md](/maintenance/summary_template)：章节收口模板

## 怎么用

- 修改教程结构、写新章节或维护现有章节时，先看这里
- 只学习主线内容时，可以跳过这里
- 这些页面主要约束写法和流程，不承担主线教学

## 检查顺序

- 修改文档或链接后，先跑 `npm run docs:check-links`
- 再跑 `npm run docs:build`
- 如果构建失败，先查站点路由和锚点，再查正文内容
