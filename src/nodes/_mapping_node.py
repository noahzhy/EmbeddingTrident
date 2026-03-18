import ray
from ray import serve


@serve.deployment
class MappingNode:
    def __init__(self, file_path: str):
        # format as: pms_id,cls_id -> 124532,0
        with open(file_path, "r") as f:
            self.id_map = {
                line.strip().split(',')[1]: line.strip().split(',')[0] 
                for line in f if "," in line
            }

    def map_id(self, result_index: str):
        # input 0 -> return 124532
        return self.id_map.get(str(result_index))


if __name__ == "__main__":
    _class = MappingNode.func_or_class
    
    node = _class("data/mapping_table.txt")
    print(f"Result for 0: {node.map_id('0')}")
    print(f"Result for 0: {node.map_id('1')}")
    print(f"Result for 0: {node.map_id('2')}")
