from poetry_dataset import PoetryNERDataset
from transformers import BertTokenizer
from config import BERT_MODEL_PATH, ALLUSION_DICT_PATH
from train import load_allusion_dict


def parse_line(line):
        """解析单行数据，提取诗句和标签"""
        parts = line.strip().split('\t')
        try:
            # 获取基本信息
            text = parts[0].strip()
            
            # 处理有典故的情况
            allusion_info = parts[6].strip()
            allusion_parts = allusion_info.split(';')
            
            # 初始化标签序列
            position_labels = ['O'] * len(text)
            type_labels = ['O'] * len(text) # 'O'表示非典故
            
            # 处理每个典故
            for part in allusion_parts:
                part = part.strip()
                if not part:  # 跳过空字符串
                    continue
                
                part = part.strip('[]')
                items = [item.strip() for item in part.split(',')]
                positions = [int(pos) for pos in items[:-1]]
                allusion_type = items[-1]
                
                # 设置位置标签
                if positions:
                    position_labels[positions[0]] = 'B'
                    for pos in positions[1:]:
                        position_labels[pos] = 'I'
                    
                    # 设置类型标签
                    for pos in positions:
                        type_labels[pos] = allusion_type
                        
            return text, position_labels, type_labels
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line}")
            print(f"Error details: {str(e)}")
            return None
        
def test_parse_line():
    # 初始化必要的组件

        
    # 测试数据
    test_line ="高皇冷笑重瞳客，盖世拔山何所为。	徐夤	偶书	拔山（项羽，虞姬）	1	[[10, 11]]	[10, 11,拔山（项羽，虞姬）]\n"
    
    # 解析并打印结果
    print("\n=== 解析结果 ===")
    result = parse_line(test_line)
        
    text, position_labels, type_labels = result
    
    print(f"\n1. 文本内容（长度：{len(text)}）：")
    print(text)
    
    print(f"\n2. 位置标签（长度：{len(position_labels)}）：")
    print("位置: ", end='')
    for i, label in enumerate(position_labels):
        print(f"{i:2d}", end=' ')
    print("\n文字: ", end='')
    for char in text:
        print(f"{char:2s}", end=' ')
    print("\n标签: ", end='')
    for label in position_labels:
        print(f"{label}", end=' ')
        
    print(f"\n\n3. 类型标签（长度：{len(type_labels)}）：")
    print("位置: ", end='')
    for i in range(len(type_labels)):
        print(f"{i}", end=' ')
    print("\n文字: ", end='')
    for char in text:
        print(f"{char:2s}", end=' ')
    print("\n标签: ", end='')
    for label in type_labels:
        print(f"{label}", end=' ')
    
    # 检查长度匹配
    print("\n\n=== 长度检查 ===")
    print(f"文本长度: {len(text)}")
    print(f"位置标签长度: {len(position_labels)}")
    print(f"类型标签长度: {len(type_labels)}")
    
    if len(text) != len(position_labels) or len(text) != len(type_labels):
        print("\n警告：长度不匹配！")

if __name__ == "__main__":
    test_parse_line() 