# 词义消歧与语义角色标注系统 (WSD-SRL NLP System)

一个基于Python Streamlit框架的交互式自然语言处理Web应用，集成了词义消歧(WSD)和语义角色标注(SRL)两个功能模块。

## 🌐 在线体验

访问线上应用：[https://wsd-srl-nlp.streamlit.app](https://wsd-srl-nlp.streamlit.app)

（如果链接不可用，请查看下方本地运行说明）

## ✨ 功能模块

### 📖 模块一：词义消歧 (WSD - Word Sense Disambiguation)

- **Lesk算法**：使用WordNet进行传统词义消歧
- **BERT上下文向量**：提取768维的上下文词向量
- **余弦相似度**：对比多个句子中词义的语义相似性
- **可视化**：向量分布直方图展示

**示例**：
- 第一句："I went to the bank to deposit my money."（金融机构）
- 第二句："I sat by the river bank."（河岸）
- 系统计算"bank"在两句中的语义相似度

### 🔤 模块二：语义角色标注 (SRL - Semantic Role Labeling)

- **依存句法分析**：使用spaCy分析句子结构
- **启发式规则提取**：自动识别语义角色
  - A0（施事者）：主语
  - Predicate（谓词）：动词
  - A1（受事者）：宾语
  - AM-LOC（地点）：地点修饰语
  - AM-TMP（时间）：时间修饰语
- **可视化展示**：结构化表格 + 依存关系图

**示例**：
- 输入："Apple is manufacturing new smartphones in China this year."
- 输出：
  | 角色 | 文本 |
  |------|------|
  | A0 | Apple |
  | Predicate | manufacturing |
  | A1 | smartphones |
  | AM-LOC | in China |
  | AM-TMP | this year |

## 🛠️ 技术栈

- **框架**：[Streamlit](https://streamlit.io/) - Python Web应用框架
- **NLP库**：
  - `nltk` - 自然语言处理工具包
  - `spacy` - 工业级NLP库
  - `transformers` - Hugging Face预训练模型
  - `torch` - 深度学习框架
- **数据处理**：`pandas`, `numpy`
- **可视化**：`matplotlib`, `spacy-displacy`
- **计算**：`scikit-learn` - 余弦相似度计算

## 📋 系统要求

- **Python**：3.8 或更高版本
- **内存**：至少4GB（用于模型加载）
- **存储**：约1GB（用于模型和依赖）

## 🚀 快速开始

### 方式一：本地运行

#### 1. 克隆项目
```bash
git clone https://github.com/pekabo1015/wsd-srl-nlp.git
cd wsd-srl-nlp
```

#### 2. 创建虚拟环境
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

#### 4. 下载模型（首次需要）
```bash
# 下载spaCy英文模型
python -m spacy download en_core_web_sm

# NLTK会在首次运行应用时自动下载wordnet等资源
```

#### 5. 运行应用
```bash
streamlit run week5.py
```

应用会自动在浏览器打开：`http://localhost:8501`

### 方式二：Docker运行

```bash
# 构建镜像
docker build -t wsd-srl-app .

# 运行容器
docker run -p 8501:8501 wsd-srl-app
```

## 📁 项目结构

```
wsd-srl-nlp/
├── week5.py                 # 主应用程序
├── requirements.txt         # Python依赖
├── .gitignore              # Git忽略文件
├── Dockerfile              # Docker配置（可选）
├── 记录.md                 # 项目开发记录
├── README.md               # 本说明文件
└── 其他数据文件
```

## 🎯 使用示例

### WSD模块示例

1. 打开应用，进入"WSD 词义消歧"标签页
2. 输入第一个句子：`"I went to the bank to deposit my money."`
3. 输入目标词：`bank`
4. 点击"第一阶段分析"
5. 查看结果：
   - Lesk算法的WordNet Synset
   - BERT词向量的统计信息和分布
6. 展开第一阶段，查看详细结果
7. 输入第二个句子：`"I sat by the river bank."`
8. 点击"计算余弦相似度"
9. 查看两个上下文中"bank"的语义相似度

### SRL模块示例

1. 进入"SRL 语义角色标注"标签页
2. 默认句子：`"Apple is manufacturing new smartphones in China this year."`
3. 点击"执行语义角色标注"
4. 查看结果：
   - 结构化的语义角色表格
   - 依存关系树状图
   - 提取规则说明

## 📊 性能与限制

- **WSD模块**：处理时间 2-5秒（取决于句子长度）
- **SRL模块**：处理时间 1-3秒
- **模型大小**：BERT模型约350MB，spaCy模型约40MB
- **并发用户**：Streamlit Cloud免费版支持3-5个并发用户

## 🐛 故障排除

### 错误：ModuleNotFoundError: No module named 'spacy'
**解决**：运行 `pip install -r requirements.txt`

### 错误：spaCy model 'en_core_web_sm' not found
**解决**：运行 `python -m spacy download en_core_web_sm`

### 应用启动很慢
**原因**：首次加载BERT模型（约350MB）
**解决**：耐心等待3-5分钟，后续启动会快很多（使用缓存）

### 依存图不显示
**解决**：刷新页面或重启应用

## 📝 开发与贡献

### 本地开发流程
```bash
# 修改代码后测试
streamlit run week5.py

# 推送到GitHub
git add .
git commit -m "描述修改内容"
git push origin main
```

### Streamlit Cloud自动部署
GitHub push后，Streamlit Cloud会自动检测并重新部署应用（约2-3分钟）

## 📚 参考资源

- [Streamlit文档](https://docs.streamlit.io/)
- [spaCy文档](https://spacy.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NLTK文档](https://www.nltk.org/)
- [WordNet](https://wordnet.princeton.edu/)

## 📄 许可证

MIT License - 供学习和研究使用

## 👨‍💻 作者

- **项目名**：词义消歧与语义角色标注系统
- **开发时间**：2026年4月
- **GitHub**：[pekabo1015](https://github.com/pekabo1015)

## 📧 反馈与改进

如有问题或建议，欢迎提出Issue或联系开发者。

---

**祝您使用愉快！** 🎉
