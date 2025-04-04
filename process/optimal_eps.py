from difflib import SequenceMatcher
import numpy as np

# 计算两个字符串之间的相似度
#ATTENTION 这个算法是在写报告时更新的！在模型中使用的是原来的算法，原来只对sorted==sorted额外处理
def string_similarity(str1, str2):
    """计算两个字符串之间的相似度
    
    Args:
        str1: 第一个字符串
        str2: 第二个字符串
    Returns:
        float: 相似度分数 (0-1)
    """
    # 如果字符串完全相同
    if str1 == str2:
        return 1.0
        
    # 计算原始顺序的相似度
    original_similarity = SequenceMatcher(None, str1, str2).ratio()
    
    # 计算排序后的相似度
    sorted_similarity = SequenceMatcher(None, ''.join(sorted(str1)), ''.join(sorted(str2))).ratio()
    
    # 组合两种相似度，给排序后的相似度一个较小的权重（如0.3）
    final_similarity = original_similarity * 0.7 + sorted_similarity * 0.3
    
    return final_similarity

# 选择每个聚类的代表词
def select_representative(cluster):
    # 计算每个词与其他词的平均相似度
    avg_similarities = {}
    for word1 in cluster:
        total_sim = 0
        for word2 in cluster:
            if word1 != word2:
                total_sim += string_similarity(word1, word2)
        avg_similarities[word1] = total_sim / (len(cluster) - 1) if len(cluster) > 1 else 0
    
    # 选择与其他词平均相似度最高的词作为代表词
    return max(avg_similarities.items(), key=lambda x: x[1])[0]

def calculate_optimal_eps_for_all_groups(major_groups):
    """为所有组计算最优eps值，考虑所有组间的相似度"""
    # 存储所有组内相似度和组间相似度
    intra_similarities = []  # 组内相似度
    inter_similarities = []  # 组间相似度
    
    # 获取所有小组及其代表词
    all_groups = []
    all_representatives = []
    for major_group in major_groups:
        for group in major_group:
            all_groups.append(group)
            all_representatives.append(select_representative(group))
    
    # 计算组内相似度
    for i, group in enumerate(all_groups):
        rep = all_representatives[i]
        for word in group:
            if word != rep:
                intra_similarities.append(string_similarity(rep, word))
    
    # 计算所有组间相似度
    for i in range(len(all_groups)):
        for j in range(i + 1, len(all_groups)):
            rep_i = all_representatives[i]
            for word in all_groups[j]:
                inter_similarities.append(string_similarity(rep_i, word))
    
    # 将相似度转换为numpy数组并排序
    intra_similarities = np.array(intra_similarities)
    inter_similarities = np.array(inter_similarities)
    
    # 合并所有相似度并排序，用作候选eps值
    all_similarities = np.unique(np.concatenate([intra_similarities, inter_similarities]))
    all_similarities.sort()
    
    # 寻找最优eps
    best_eps = None
    best_score = -1
    
    for eps in all_similarities:
        # 计算在当前eps下的指标
        intra_included = np.sum(intra_similarities >= eps)  # 包含的组内相似度数量
        inter_excluded = np.sum(inter_similarities < eps)   # 排除的组间相似度数量
        
        # 计算准确率和召回率
        intra_recall = intra_included / len(intra_similarities) if len(intra_similarities) > 0 else 0
        inter_precision = inter_excluded / len(inter_similarities) if len(inter_similarities) > 0 else 0
        
        # 计算F1分数
        if intra_recall + inter_precision > 0:
            f1_score = 2 * (intra_recall * inter_precision) / (intra_recall + inter_precision)
        else:
            f1_score = 0
            
        # 更新最优值
        if f1_score > best_score:
            best_score = f1_score
            best_eps = eps
    
    return best_eps if best_eps is not None else 0.5


if __name__ == "__main__":
    # 示例分组（多个大组，每个大组包含多个小组）
    major_groups = [
        # 第一个大组
        [
            ['开三径', '三径', '开竹径'],
            ['羊求', '求仲', '羊仲', '求羊径'],
            ['蒋诩径', '蒋生径']
        ],
        # 第二个大组
        [
             ['隐墙东','避世墙东','墙东隐','墙东客','卧墙东','老墙东','高卧墙东','侩墙东','墙东'],
            ['侩牛', '牛侩','学刽牛'],
            ['避世贤']
        ],
        # 第三个大组
        [
            ['濯尘缨', '濯我缨','解尘缨','濯缨','思濯缨','缨濯','尘缨欲濯','净濯兰缨','濯楚臣缨'],
            ['濯沧浪','沧浪','沧浪濯缨','濯沧浪缨','沧浪之水','之水','濯发沧浪','沧浪未濯'],
            ['逢渔父','渔父笑','渔父濯沧浪','渔父足','值渔父','渔父'],
            ['孺子歌','鼓枻歌','鼓枻翁'],
            ['濯足','濯冠巾','濯','清濯','欲濯'],
            ['论浊清']
        ],
        # 第四个大组
        [
            ['高山流水','流水曲','流水琴','山水在琴','山水调','山水','流水'],
            ['伯牙琴','伯牙弦','牙弦','弄琴牙','伯牙'],
            ['钟期耳','钟殁师废琴','子期','钟期辨','钟期','子期知音'],
            ['绝弦','朱弦断','弦断','弦琴肯重闻','弦绝', '琴弦'],
            ['罢琴']
        ],
        # 第五个大组
        [
            ['杜鹃啼血','鹃啼血','啼成血','杜宇啼血','口流血','鹃血','口血','血','啼血'],
            ['魂作杜鹃','杜鹃积恨','泣杜鹃','啼鹃','杜鹃魂','子规','杜魂','杜魄','杜宇','冤魂化禽','魄'],
            ['古帝魂','蜀帝魂','蜀天子','蜀王遗魄','蜀王作鸟','蜀帝王','蜀王','昔帝','昔帝恨','悲蜀帝'],
            ['望帝春心','望帝啼鹃','望帝','望帝归魂','望帝愁魂'],
            ['蜀鸟', '蜀魂','蜀羽怨红啼断','蜀魄']
        ],
        # 第六个大组
        [
            ['长生药','偷药','药娥','姮娥捣药','羿妻窃药','窃药'],
            ['月中人','嫦娥奔月','月娥孀独','奔月偶桂','奔月成仙','奔月','月中孀','月里人','月娥孤'],
            ['孀娥','恒娥','姮娥寡','姱娥','素娥','伴娥孤另','常娥','嫦娥','姮娥','娥'],
            ['羿妻','后羿寻','后羿'],
            ['素']
        ],
        # 第七个大组
        [
            ['馆娃宫','馆娃歌舞','馆娃','名娃金屋','娃宫'],
            ['浣纱人','浣纱石','浣纱神女','浣纱'],
            ['采香径','香径'],
            ['苧罗人','苧萝'],
            ['吴宫妃','西施','西施倾吴','西子']
        ],
        # 第八个大组
        [
            ['鲛人泪','鲛泪','鲛人珠','鲛丝','鲛人织绡','鲛人','鲛绡','鲛人泣珠','鲛工', '鲛宫物','鲛帘','鲛盘千点','泪有鲛人','鲛人雨泣','鲛珠','蛟人珠','蛟女绢','蛟人'],
            ['珠有泪','泣珠','泪成珠','泣珠人','泪作珠','洒泪成珠','泉客泪','泉客泣','泉客珠','珠进泪','珠泪','玉盘进泪','泪','波泪','成珠','泣珠报恩'],
            ['卷绡人','泪绡','海上得绡','冰绡清泪','绡夺烟雾']
        ],
        # 第九个大组
        [
            ['卧南阳','南阳卧','南阳高卧','高卧南阳'],
            ['卧龙', '龙卧','葛龙卧','诸葛号龙','葛龙','龙如诸葛'],
            ['诸葛庐','草庐无人顾','诸葛']
        ],
        # 第十个大组
        [
            ['羲皇上','羲皇','羲和人','傲羲皇','傲羲轩','羲皇人','羲皇上人','上皇人','白日羲皇','羲皇情','羲皇侣','羲皇以上人','笑羲皇','直到羲皇世','身致羲皇上','高枕晤羲皇','高卧偃羲皇','闭门寻羲皇','人似上皇','羲皇叟','卧羲皇','慕羲皇','上皇','到羲皇'],
            ['北窗风','卧北窗','北窗眠','北窗凉','北窗高卧','北窗卧','北窗高枕','北窗一枕','北窗卧羲皇', '北窗羲皇','北窗寄傲','老北窗','宇宙一北窗','北牖羲皇','北窗睡美','靖节窗风'],
            ['风期结陶叟','陶窗','陶令塌','陶令羲皇','北窗']
        ],
        
        # 第十一大组
        [
            ['精卫填海','填沧海','填海精卫','填海心','禽海填','填渤澥','投石填海','石填大海','女娃东海','填海','心平海','禽填海','填瀚海','填'],
            ['含石','精卫衔石','衔木石','精卫衔芦','费木石','衔石冤禽','衔木','口衔山石','衔木鸟','衔石','帝女衔石'],
            ['精卫恨','精卫苦','精卫怒','精卫','化精卫','魂化精卫'],
            ['沧海鸟','沈冤鸟口','帝子衔冤','冤禽','衔土怨'],
            ['帝女灵','碧海乾', '帝女填']
        ],
        # 第十二个大组
        [
            ['羲车','羲和辔','羲和驾','羲和车','朱羲','羲和驭日','驻羲和','羲和日驭', '羲和辀','羲和轮','羲和乌','羲和失鞭','羲轮','羲驭'],
            ['回日驭', '日车','鞭日','驱日月','鞭挞日月','鞭白日','日御','日驭']
        ],
        # 第十三个大组
        [
            ['孟邻','孟家邻','孟母邻','孟子邻','邻'],
            ['孟母三迁','孟母教','孟母迁邻','三迁教养','三徙'],
            ['孟子','孟']
        ],
        # 第十四个大组
        [
            ['舌存','张仪舌','存舌','吾舌在','留舌','仪舌','张仪舌在','仪舌在','舌在何忧','纵横舌','有舌','恒视舌','留舌示妻','舌问妻孥','舌在'],
            ['疑璧' '盗璧','韫璧', '璧非真盗'],
            ['诬张仪','张仪']
        ]
    ]
    
    # 1. 计算最优eps
    optimal_eps = calculate_optimal_eps_for_all_groups(major_groups)
    print(f"\n最优eps值: {optimal_eps:.3f}")
