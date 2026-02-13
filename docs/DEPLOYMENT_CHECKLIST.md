# Deployment Checklist

Use this checklist to ensure a smooth deployment of the image embedding service.

## Pre-Deployment

### Infrastructure
- [ ] Docker installed and running
- [ ] Docker Compose installed (v1.29+)
- [ ] NVIDIA GPU drivers installed (if using GPU)
- [ ] NVIDIA Container Toolkit installed (if using GPU)
- [ ] Sufficient disk space (min 20GB recommended)
- [ ] Network ports available: 8080, 8000-8002, 19530

### Model Preparation
- [ ] Embedding model exported to ONNX format
- [ ] Model placed in correct directory structure
- [ ] Triton config file created and validated
- [ ] Model tested locally

### Configuration
- [ ] `config.yaml` customized for your environment
- [ ] Environment variables set (if using)
- [ ] Batch sizes tuned for your hardware
- [ ] Index type selected (IVF_FLAT/HNSW/FLAT)
- [ ] Collection parameters configured

## Deployment Steps

### 1. Start Services

```bash
# Pull latest images
docker-compose pull

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

- [ ] All containers running
- [ ] No error messages in logs
- [ ] Health checks passing

### 2. Verify Services

**Milvus**
```bash
curl http://localhost:19530
```
- [ ] Milvus responding

**Triton**
```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/embedding_model/ready
```
- [ ] Triton server healthy
- [ ] Model loaded and ready

**API Service**
```bash
curl http://localhost:8080/health
```
- [ ] API service healthy
- [ ] All components reporting healthy

### 3. Functional Testing

**Create collection**
```bash
curl -X POST http://localhost:8080/collections/create \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "dim": 512}'
```
- [ ] Collection created successfully

**Insert test data**
```bash
curl -X POST http://localhost:8080/insert \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": ["/path/to/test/image.jpg"],
    "ids": ["test_1"],
    "collection_name": "test"
  }'
```
- [ ] Insert successful
- [ ] No errors in logs

**Search**
```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_input": "/path/to/test/image.jpg",
    "topk": 5,
    "collection_name": "test"
  }'
```
- [ ] Search returns results
- [ ] Response time acceptable

**Cleanup**
```bash
curl -X DELETE http://localhost:8080/collections/test
```
- [ ] Test collection deleted

## Performance Validation

### Run Benchmarks

```bash
python examples/batch_processing.py
```

- [ ] Preprocessing performance meets targets
- [ ] Inference performance meets targets
- [ ] Insert throughput meets targets
- [ ] Search latency meets targets

### Expected Performance

**Single GPU (RTX 3090)**
- [ ] Preprocessing: ≥ 40 images/sec
- [ ] Inference: ≥ 100 images/sec
- [ ] Insert: ≥ 5000 vectors/sec
- [ ] Search: < 50ms (TopK=10)

Adjust targets based on your hardware.

## Monitoring Setup

### Metrics Collection
- [ ] Triton metrics endpoint accessible
- [ ] Application logs being collected
- [ ] Resource usage monitored

### Alerts
- [ ] Service health alerts configured
- [ ] Performance degradation alerts set
- [ ] Disk space alerts configured
- [ ] Memory usage alerts set

### Dashboards
- [ ] Grafana/monitoring dashboard set up
- [ ] Key metrics visible
- [ ] Alerts integrated

## Security

### Network
- [ ] Firewall rules configured
- [ ] Only necessary ports exposed
- [ ] SSL/TLS configured for production
- [ ] Rate limiting enabled

### Authentication
- [ ] API authentication enabled (if required)
- [ ] Milvus authentication configured (if required)
- [ ] Strong passwords/tokens used
- [ ] Secrets not committed to git

### Data
- [ ] Data persistence configured
- [ ] Backup strategy in place
- [ ] Data retention policy defined
- [ ] PII handling reviewed

## Documentation

- [ ] Deployment architecture documented
- [ ] Configuration parameters documented
- [ ] Runbook created for common issues
- [ ] Contact information for support

## Production Readiness

### High Availability
- [ ] Multiple replicas configured
- [ ] Load balancer set up
- [ ] Health checks configured
- [ ] Auto-restart enabled

### Backup & Recovery
- [ ] Database backup automated
- [ ] Model files backed up
- [ ] Configuration backed up
- [ ] Recovery procedure tested

### Scaling Plan
- [ ] Horizontal scaling strategy defined
- [ ] Resource limits configured
- [ ] Auto-scaling rules set (if applicable)
- [ ] Capacity planning done

### Operations
- [ ] Log rotation configured
- [ ] Monitoring alerts tested
- [ ] Incident response plan ready
- [ ] On-call rotation set up

## Post-Deployment

### Day 1
- [ ] Monitor logs for errors
- [ ] Verify metrics are normal
- [ ] Test with real traffic
- [ ] Document any issues

### Week 1
- [ ] Review performance metrics
- [ ] Tune configuration as needed
- [ ] Address any bottlenecks
- [ ] Update documentation

### Month 1
- [ ] Analyze usage patterns
- [ ] Plan capacity upgrades
- [ ] Review and update alerts
- [ ] Conduct post-mortem if issues occurred

## Rollback Plan

If deployment fails:

1. **Stop new service**
   ```bash
   docker-compose down
   ```

2. **Check logs**
   ```bash
   docker-compose logs
   ```

3. **Restore previous version**
   ```bash
   git checkout <previous-version>
   docker-compose up -d
   ```

4. **Verify**
   - [ ] Old version running
   - [ ] Services healthy
   - [ ] Data intact

## Sign-Off

- [ ] Development team approval
- [ ] QA team approval
- [ ] Operations team approval
- [ ] Security team approval (if applicable)

---

**Deployment Date**: _____________

**Deployed By**: _____________

**Version**: _____________

**Notes**: _____________
