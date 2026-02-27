from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


def create_collection_no_index(config):
    connections.connect(host=config["MILVUS_HOST"], port=config["MILVUS_PORT"])
    collection_name = config["COLLECTION_NAME"]

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    type_map = {
        "INT64":            DataType.INT64,
        "INT32":            DataType.INT32,
        "FLOAT":            DataType.FLOAT,
        "DOUBLE":           DataType.DOUBLE,
        "BOOL":             DataType.BOOL,
        "FLOAT_VECTOR":     DataType.FLOAT_VECTOR,
        "BINARY_VECTOR":    DataType.BINARY_VECTOR,
        "VARCHAR":          DataType.VARCHAR,
    }

    fields = []
    for field in config["COLLECTION_FIELDS"]:
        field_name = field.get("name")
        field_type = field.get("type")
        if not field_name or not field_type:
            continue

        dtype = type_map.get(field_type)
        if dtype is None:
            raise ValueError(f"Unsupported field type: {field_type}")

        if field_type in ("FLOAT_VECTOR", "BINARY_VECTOR"):
            dim = field.get("dim") or config["COLLECTION_VECTOR_DIM"]
            fields.append(
                FieldSchema(
                    name=field_name,
                    dtype=dtype,
                    dim=int(dim),
                )
            )
        else:
            fields.append(
                FieldSchema(
                    name=field_name,
                    dtype=dtype,
                    is_primary=bool(field.get("primary_key", False)),
                    auto_id=bool(field.get("auto_id", False)),
                )
            )

    schema = CollectionSchema(fields)
    return Collection(collection_name, schema)


def build_index_and_load(config):
    connections.connect(host=config["MILVUS_HOST"], port=config["MILVUS_PORT"])
    collection = Collection(config["COLLECTION_NAME"])

    index_field = config.get("COLLECTION_INDEX_FIELD")
    index_cfg = config.get("COLLECTION_INDEX_CFG") or {}
    if index_field and index_cfg:
        index_params = {
            "index_type":   index_cfg.get("type",   "GPU_IVF_PQ"),
            "metric_type":  index_cfg.get("metric", "IP"),
            "params":       index_cfg.get("params", {}),
        }
        collection.create_index(index_field, index_params)

    collection.load()
    return collection


class MilvusInserter:
    def __init__(self, host, port, collection_name, auto_id):
        connections.connect(host=host, port=port)
        self.collection = Collection(collection_name)
        self.auto_id = bool(auto_id)

    def insert(self, ids, vectors):
        if self.auto_id:
            self.collection.insert([vectors.tolist()])
        else:
            self.collection.insert([ids, vectors.tolist()])
