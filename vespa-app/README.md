# Vespa Application Package 部署指南

## 📁 应用包结构

```
vespa-app/
├── services.xml                          # Vespa 服务配置
├── hosts.xml                             # 主机配置
├── schemas/
│   └── multi_vector.sd                   # Multi-Vector Schema 定义
└── search/
    └── query-profiles/
        └── default.xml                   # 默认查询配置
```

## 📋 Schema 说明

### multi_vector.sd

定义了多向量文档结构：

**字段**:
- `document_id` (string): 文档ID，带索引
- `chunk_number` (int): 块编号
- `content` (string): 文档内容
- `chunk_metadata` (string): 元数据JSON
- `embeddings` (tensor<float>(d0[128])): 128维向量张量

**Rank Profiles**:
- `default`: 基于文本的默认排序
- `max_sim`: 多向量最大相似度排序（Max-Sim算法）
- `bm25`: BM25文本相似度排序

## 🚀 部署方法

### 方法1: 使用部署脚本（推荐）

#### Windows (PowerShell)
```powershell
# 确保 Vespa 已启动
docker-compose up -d vespa

# 运行部署脚本
.\deploy_vespa_app.ps1
```

#### Linux/Mac (Bash)
```bash
# 确保 Vespa 已启动
docker-compose up -d vespa

# 添加执行权限并运行
chmod +x deploy_vespa_app.sh
./deploy_vespa_app.sh
```

### 方法2: 使用 Docker 命令

```bash
# 1. 复制应用包到容器
docker cp vespa-app vespa:/opt/vespa/morphik-app

# 2. 部署应用
docker exec vespa vespa-deploy prepare /opt/vespa/morphik-app
docker exec vespa vespa-deploy activate

# 3. 验证部署
curl http://localhost:8080/ApplicationStatus
```

### 方法3: 使用 Vespa CLI（需要安装）

```bash
# 安装 Vespa CLI
# macOS: brew install vespa-cli
# Linux: 参考 https://docs.vespa.ai/en/vespa-cli.html

# 部署应用
vespa deploy vespa-app --target http://localhost:8080
```

### 方法4: 手动 HTTP API 部署

```bash
# 打包应用
cd vespa-app
zip -r ../morphik-app.zip *
cd ..

# 部署
curl -X POST \
  http://localhost:8080/application/v2/tenant/default/prepareandactivate \
  -H "Content-Type: application/zip" \
  --data-binary @morphik-app.zip
```

## ✅ 验证部署

### 1. 检查应用状态

```bash
curl http://localhost:8080/ApplicationStatus
```

预期输出包含 `"active"` 状态。

### 2. 插入测试文档

```bash
curl -X POST \
  http://localhost:8080/document/v1/default/multi_vector/docid/test_1 \
  -H "Content-Type: application/json" \
  -d '{
    "fields": {
      "document_id": "test_doc",
      "chunk_number": 0,
      "content": "This is a test document",
      "chunk_metadata": "{}",
      "embeddings": {
        "values": [0.1, 0.2, 0.3, ...(128个值)]
      }
    }
  }'
```

### 3. 查询文档

```bash
curl -X POST \
  http://localhost:8080/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "yql": "select * from multi_vector where true",
    "hits": 10
  }'
```

### 4. 运行 Python 测试

```bash
python core/tests/test_vespa_multi_vector_store.py
```

## 🔧 修改 Schema

如果需要修改 schema：

1. 编辑 `vespa-app/schemas/multi_vector.sd`
2. 重新部署：`.\deploy_vespa_app.ps1`
3. Vespa 会自动重新索引数据

## 📊 Schema 详细说明

### 向量字段配置

```
field embeddings type tensor<float>(d0[128]) {
    indexing: summary | attribute
    attribute {
        distance-metric: angular  # 使用余弦相似度
    }
}
```

- `d0[128]`: 定义为128维向量
- `angular`: 使用角度距离（余弦相似度）
- 可以改为其他维度，如 `d0[256]` 或 `d0[512]`

### Max-Sim Rank Profile

```
rank-profile max_sim {
    inputs {
        query(q) tensor<float>(d0[128])
    }
    
    first-phase {
        expression {
            sum(
                reduce(
                    max(
                        cosine(query(q), attribute(embeddings)),
                        d0
                    ),
                    sum,
                    d1
                )
            )
        }
    }
}
```

**工作原理**:
1. 计算查询向量和文档所有向量的余弦相似度
2. 对每个查询向量取最大相似度
3. 求和得到最终分数

## 🐛 故障排查

### 问题1: 部署失败

**检查日志**:
```bash
docker logs vespa
```

**常见原因**:
- Schema 语法错误
- 端口冲突
- 内存不足

**解决方法**:
```bash
# 重启 Vespa
docker-compose restart vespa

# 等待服务就绪
sleep 10

# 重新部署
.\deploy_vespa_app.ps1
```

### 问题2: Schema 不生效

**重新激活应用**:
```bash
docker exec vespa vespa-deploy activate
```

### 问题3: 查询返回空结果

**检查文档数量**:
```bash
curl http://localhost:8080/document/v1/default/multi_vector/docid?selection=true&wantedDocumentCount=10
```

**检查 Schema**:
```bash
curl http://localhost:8080/application/v2/tenant/default/application/default
```

## 📚 相关文档

- [Vespa 官方文档](https://docs.vespa.ai/)
- [Schema 参考](https://docs.vespa.ai/en/schemas.html)
- [Rank Profiles](https://docs.vespa.ai/en/ranking.html)
- [Tensor 操作](https://docs.vespa.ai/en/tensor-user-guide.html)

## 🔄 更新应用

修改后重新部署：

```bash
# 修改 schema 或配置后
.\deploy_vespa_app.ps1

# 或使用 Docker
docker cp vespa-app vespa:/opt/vespa/morphik-app
docker exec vespa vespa-deploy prepare /opt/vespa/morphik-app
docker exec vespa vespa-deploy activate
```

## 📝 注意事项

1. **向量维度**: 当前配置为128维，需要与代码中的维度一致
2. **内存需求**: Vespa 需要至少2GB内存
3. **数据持久化**: 确保配置了 volume 映射
4. **性能调优**: 生产环境建议增加 `redundancy` 和配置集群

## 🎯 下一步

部署完成后：

1. ✅ 运行测试验证
2. ✅ 查看 `docs/vespa_multi_vector_store.md` 了解 API 使用
3. ✅ 阅读 `VESPA_QUICKSTART.md` 快速开始
4. ✅ 集成到你的应用中

