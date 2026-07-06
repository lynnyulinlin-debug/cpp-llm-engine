# F05 OOP and Design Basics OOP 和设计基础

## 目标

把主教程会反复出现的对象设计原则先对齐，避免后面把所有权、边界和状态管理写散。

## 读完你应该能做什么

- 理解类是“状态 + 行为”的组合
- 理解封装，以及为什么公开接口应该尽量小
- 理解继承和组合，以及它们各自适合什么场景
- 理解为什么这套教程更偏向简单、明确的对象，而不是花哨抽象

## 为什么重要

这套教程会用 C++ 类来表达 `Tensor`、`ModelLoader`、operator、cache、runtime state 和 benchmark helper。对象设计不清楚时，读者通常不是卡在数学上，而是卡在“谁持有什么、谁负责什么、谁能改什么”。

这不是一门通用 OOP 课，而是主线代码的设计预热。后面的对象大多都只有一个职责：要么描述数据，要么描述行为，要么描述一段明确的运行时状态。这个页要解决的，就是把这种写法先变成默认直觉。

## 后面会用到什么

- `Tensor` 和 `ModelLoader` 的设计
- operator 和 backend 类
- decoder layer 组合
- cache 和 runtime state 对象

## 职责映射

| 类 | 职责类型 | 说明 |
| --- | --- | --- |
| `Tensor` | 数据容器 | 持有存储和元数据 |
| `ModelLoader` | 文件入口 | 解析并构造数据对象 |
| `MatMul` | 计算节点 | 执行具体运算 |
| `KVCache` | 状态壳 | 管理运行时缓存 |
| `Runtime` | 状态壳 | 管理推理状态 |
| `DecoderLayer` | 组合对象 | 组合 attention、norm 和 FFN |

## 这一页的核心内容

### 1. 对象先分职责，再谈实现

一个类最好只承担一类责任。教程里最常见的拆法是：

- 数据容器：`Tensor`
- 文件入口：`ModelLoader`
- 计算节点：operator / backend
- 状态壳：cache / runtime state

这样拆的目的不是“更面向对象”，而是让后续章节能单独验证每一层。

### 2. 接口只暴露下一步需要的能力

接口越小，越容易理解，也越容易测试。真正有价值的接口不是“功能最多”，而是“当前章节刚好够用”。

这会直接影响教程写法：

- 上一章先把对象边界定住
- 下一章只新增必要方法
- 不把未来可能会用到的东西提前暴露出来

### 3. 状态、行为和资源所有权要一起设计

只要对象持有内存、文件句柄、device buffer 或共享 storage，它的生命周期就属于设计的一部分。

实践上要记住三点：

- 谁创建，谁负责初始化到可用状态
- 谁拥有，谁负责释放或转移所有权
- 谁共享，谁必须把共享边界写清楚

这也是为什么教程里会反复强调 RAII、`shared_ptr`、`unique_ptr` 和构造/析构语义。

### 4. 组合优于深层继承

这套教程更偏向组合，而不是深层继承树。原因很工程化：

- 组合更容易替换局部实现
- 组合更容易让单层验证成立
- 组合更容易把 decoder、operator 和 backend 拆开

当一个结构可以通过“拼装”说明时，不要先把它做成复杂父子层级。

```cpp
class DecoderLayer {
public:
    Tensor forward(const Tensor& input) {
        auto x = attention_->forward(input);
        x = norm_->forward(x);
        return ff_->forward(x);
    }

private:
    std::unique_ptr<Attention> attention_;
    std::unique_ptr<LayerNorm> norm_;
    std::unique_ptr<FeedForward> ff_;
};
```

这个例子说明三件事：

- 每个组件都可以单独测试
- 替换 Attention 实现不需要改 DecoderLayer 的调用方式
- 所有权是清楚的，`DecoderLayer` 持有自己的组件

### 5. 继承只在边界明确时使用

教程里的继承边界应该尽量窄，优先用于接口抽象，而不是用于代码复用。

可以记住这几条规则：

- ✅ 纯虚基类：例如 `Backend` 基类，CPU / CUDA 分别实现
- ✅ 策略抽象：不同算子作为可替换策略
- ❌ 深层继承树：超过两层就要警惕
- ❌ 为复用代码而继承：优先用组合

如果不能用一句话说清“子类比父类多了什么”，通常就不该用继承。

### 6. 设计要服务于验证

如果一个对象设计得很漂亮，但不能单独验证，那它对教程来说就不够好。

这里的标准很明确：

- 一个对象能不能单独构造
- 一个对象能不能单独检查状态
- 一个对象能不能把错误尽早暴露出来

## 设计检查表

写主线代码时，可以直接用下面这组问题检查对象设计：

- 这个类只有一个明确职责吗
- 这个类的公开接口足够小吗
- 这个类的所有权关系能一眼看出来吗
- 这个类能不能被单独测试
- 这个类是否把“将来可能会用到”的东西提前暴露了

如果这五个问题里有两个以上答不清，就说明这个对象设计还不够收敛。

## 最小示例

```cpp
class Counter {
public:
    explicit Counter(int value) : value_(value) {}

    void add(int delta) { value_ += delta; }
    int value() const { return value_; }

private:
    int value_;
};
```

这个小例子说明了基本思路：

- 状态是私有的
- 行为是显式的
- 接口是窄的

如果把它往主线对象上投射，就能得到更接近教程的设计直觉：

- `Tensor` 要把元数据和 storage 放在同一个对象边界里
- `ModelLoader` 要把解析职责和数据对象构造分开
- operator / backend 要把计算职责和状态保存分开

接口设计也遵循同样的原则：当前章节刚好够用就行，不要提前把未来可能会用到的方法一次性加满。

```cpp
// ❌ 过度设计：提前暴露太多方法
class Tensor {
public:
    void reshape(...);
    void slice(...);
    void transpose(...);
    void broadcast(...);
    void permute(...);
};

// ✅ 刚好够用：先保留当前需要的最小接口
class Tensor {
public:
    std::size_t size() const;
    float& operator[](std::size_t idx);
    const float& operator[](std::size_t idx) const;
};
```

这类对比最重要的点不是“少写方法”，而是“方法只在需要时出现”。

## 怎么验证

- 解释这个对象的数据属于什么
- 解释这个对象的行为属于什么
- 解释为什么这个接口对后面的章节来说足够小
- 解释对象的所有权关系由谁持有、谁转移、谁共享
- 解释哪些地方应该用组合，哪些地方才考虑继承
- 解释一个子类比父类多了什么

## 常见问题

- 把太多逻辑塞进一个类
- 直接暴露内部状态
- 明明组合就够了，却用了继承
- 把 OOP 当成语法，而不是设计
- 先加接口再想职责
- 先堆抽象再补验证
- 把继承当成默认复用手段
- 接口一次性开太大

实用结论是：对象设计是架构的一部分，不只是为了让编译器通过。

## 总结

读完这一章后，读者应该能看出主教程里的 `Tensor`、`ModelLoader` 和 operator 类为什么会这样组织。

如果你能解释一个对象为什么存在、它拥有什么、以及它刻意不拥有什么，那么这一章就完成了任务。
