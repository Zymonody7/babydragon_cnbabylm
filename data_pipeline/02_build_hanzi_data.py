"""
Build hanzi-specific training data for the HANZI track.
Includes: radical decomposition, polyphone contexts, homophone disambiguation, structure descriptions.
"""
import json
import random
from pathlib import Path
from pypinyin import pinyin, Style

random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "supplementary" / "hanzi_corpus"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LQ = "「"  # 「
RQ = "」"  # 」


def load_chaizi():
    chaizi = {}
    path = DATA_DIR / "supplementary" / "chaizi" / "chaizi-jt.txt"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                char = parts[0]
                decompositions = [d.strip() for d in parts[1:]]
                chaizi[char] = decompositions
    return chaizi


def load_polyphones():
    path = DATA_DIR / "supplementary" / "chinese-dictionary" / "character" / "polyphone.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["char"]: item["pinyin"] for item in data}


GRADE1 = "一二三四五六七八九十百千万上下左右大小多少前后里外天地人你我他她它的了不在有是这那个们好会走看说话到来去过也都很还可以就着吃住坐行跑飞听写读书字词句文学生老师同对家门口手头目耳鼻足身心日月水火山石土木花草虫鱼鸟马牛羊猫狗鸡鸭鹅白黑红绿黄蓝色明亮暗东西南北风雨雪云开关出入长短高低远近快慢新旧男女父母子朋友春夏秋冬早午晚年时分秒"
GRADE2 = "城市村庄桥路船车站飞机因为所以如果已经正非常特别第每次回答问题动物植物世界国家民族历史故事事情感觉应该知道认识发现解决需要准备完成比较重要简单复杂方法结果原因变化影响保护环境自然科学技术发展进步力量勇气希望梦想幸福快乐安全健康成长努力坚持相信"
ALL_COMMON = list(set(GRADE1 + GRADE2))


def gen_chaizi_texts(chaizi, n=3000):
    texts = []
    templates = [
        LQ + "{char}" + RQ + "字可以拆分成{parts}。",
        "汉字" + LQ + "{char}" + RQ + "由{parts}组成。",
        "{char}，拆开来看是{parts}。",
        "学写" + LQ + "{char}" + RQ + "字：它是由{parts}组合而成的。",
        "认识" + LQ + "{char}" + RQ + "字：这个字的结构是{parts}。",
    ]
    chars = [c for c in ALL_COMMON if c in chaizi]
    for char in chars:
        decomp = chaizi[char][0]
        parts_str = "、".join(decomp.split())
        t = random.choice(templates)
        texts.append(t.format(char=char, parts=parts_str))
    extra = [c for c in chaizi if c not in set(ALL_COMMON)]
    random.shuffle(extra)
    for char in extra[: n - len(texts)]:
        decomp = chaizi[char][0]
        parts_str = "、".join(decomp.split())
        t = random.choice(templates)
        texts.append(t.format(char=char, parts=parts_str))
    return texts


# Polyphone sentence data: char -> [(pinyin, sentence), ...]
_POLY_DATA = {
    "行": [("hang2", "银行是存钱和取钱的地方。"), ("xing2", "行走在马路上要注意安全。"),
           ("hang2", "他在这一行干了二十年。"), ("xing2", "这个办法行得通。")],
    "长": [("zhang3", "他是我们的班长。"), ("chang2", "这条路很长，要走一个小时。"),
           ("zhang3", "小树慢慢长大了。"), ("chang2", "长江是中国最长的河流。")],
    "乐": [("le4", "六一儿童节，小朋友们很快乐。"), ("yue4", "音乐课上，大家一起唱歌。"),
           ("le4", "助人为乐是一种美德。"), ("yue4", "他是一位音乐家。")],
    "了": [("le", "我吃完饭了。"), ("liao3", "这件事了不起。"),
           ("le", "天黑了，回家吧。"), ("liao3", "他受不了这种痛苦。")],
    "还": [("hai2", "他还没有回来。"), ("huan2", "借的书要还给图书馆。"),
           ("hai2", "今天还要上一节课。"), ("huan2", "欠钱要还钱。")],
    "地": [("di4", "大地上长满了青草。"), ("de", "他高兴地跳了起来。"),
           ("di4", "这块地可以种庄稼。"), ("de", "她轻轻地关上了门。")],
    "得": [("de2", "他得了一百分。"), ("de", "他跑得很快。"),
           ("dei3", "我得走了，不然要迟到了。")],
    "都": [("dou1", "大家都到齐了。"), ("du1", "北京是中国的首都。"),
           ("dou1", "这些苹果都很甜。"), ("du1", "成都是四川省的省会。")],
    "数": [("shu4", "数学是一门重要的学科。"), ("shu3", "小朋友在数星星。"),
           ("shu4", "这个数字是多少？"), ("shu3", "数一数，一共有几个苹果？")],
    "干": [("gan4", "他干活很认真。"), ("gan1", "衣服晒干了。"),
           ("gan4", "你在干什么呢？"), ("gan1", "这块木头很干燥。")],
    "觉": [("jue2", "我觉得今天天气很好。"), ("jiao4", "他睡了一个好觉。")],
    "发": [("fa1", "老师发作业本。"), ("fa4", "奶奶的头发白了。"),
           ("fa1", "春天来了，树木发芽了。")],
    "种": [("zhong3", "这是一种新的花。"), ("zhong4", "春天到了，农民伯伯开始种地。"),
           ("zhong3", "各种各样的动物住在森林里。"), ("zhong4", "我在花盆里种了一棵小树。")],
    "着": [("zhe", "门开着，窗户也开着。"), ("zhao2", "着火了，快打119！"),
           ("zhe", "妈妈看着我笑了。"), ("zhao2", "今晚我怎么也睡不着。")],
    "重": [("zhong4", "这个箱子很重，搬不动。"), ("chong2", "我要重新做一遍。"),
           ("zhong4", "体重增加了。"), ("chong2", "重复的事情做多了就熟练了。")],
    "只": [("zhi1", "树上有一只小鸟。"), ("zhi3", "我只有一块钱。"),
           ("zhi1", "两只小猫在玩耍。"), ("zhi3", "他只吃了一碗饭。")],
    "少": [("shao3", "今天来的人很少。"), ("shao4", "少年强则国强。"),
           ("shao3", "这里的水太少了。"), ("shao4", "少先队员戴着红领巾。")],
    "为": [("wei4", "这是为你准备的礼物。"), ("wei2", "他的行为值得表扬。"),
           ("wei4", "为什么天会下雨？"), ("wei2", "他被选为班长。")],
    "好": [("hao3", "今天天气真好。"), ("hao4", "她爱好画画。"),
           ("hao3", "这本书写得很好。"), ("hao4", "他爱好读书。")],
    "空": [("kong1", "天空中飘着白云。"), ("kong4", "我有空的时候喜欢看书。"),
           ("kong1", "教室里空无一人。"), ("kong4", "请填写空格。")],
    "中": [("zhong1", "中国是一个伟大的国家。"), ("zhong4", "他中了大奖。"),
           ("zhong1", "花园在学校中间。")],
    "背": [("bei1", "他背着书包上学去。"), ("bei4", "老师让我们背课文。"),
           ("bei1", "妈妈背着弟弟走路。"), ("bei4", "他在背后说别人坏话是不对的。")],
    "相": [("xiang1", "他们互相帮助。"), ("xiang4", "照相机可以拍照。"),
           ("xiang1", "相信自己能做好。"), ("xiang4", "丞相是古代的大官。")],
    "教": [("jiao1", "老师教我们写字。"), ("jiao4", "教室里很安静。"),
           ("jiao1", "妈妈教我做饭。"), ("jiao4", "他信仰佛教。")],
    "难": [("nan2", "这道题很难。"), ("nan4", "遇到灾难要互相帮助。"),
           ("nan2", "写好作文并不难。"), ("nan4", "在困难面前不要退缩。")],
    "调": [("diao4", "这首歌的曲调很好听。"), ("tiao2", "妈妈在调电视频道。"),
           ("diao4", "他被调到了新的学校。"), ("tiao2", "要注意调节自己的情绪。")],
    "差": [("cha4", "他的成绩和别人差很多。"), ("chai1", "老师派他出差了。"),
           ("cha4", "还差一点就到了。"), ("ci1", "参差不齐的树木。")],
    "当": [("dang1", "他当上了班长。"), ("dang4", "这件事不能当真。"),
           ("dang1", "我长大了要当医生。")],
    "落": [("luo4", "树叶落了下来。"), ("la4", "他把书落在教室里了。"),
           ("luo4", "日落的时候天边很美。")],
    "场": [("chang3", "操场上同学们在跑步。"), ("chang2", "他赶了一场集。"),
           ("chang3", "电影院正在放一场好电影。")],
    "朝": [("chao2", "教室的窗户朝南。"), ("zhao1", "一日之计在于朝。"),
           ("chao2", "唐朝是中国历史上的一个朝代。"), ("zhao1", "朝气蓬勃的少年。")],
    "弹": [("tan2", "她在弹钢琴。"), ("dan4", "子弹飞得很快。"),
           ("tan2", "弹吉他是他的爱好。")],
    "参": [("can1", "他参加了学校的运动会。"), ("shen1", "人参是一种珍贵的药材。"),
           ("can1", "参观博物馆可以学到很多知识。")],
}


def gen_polyphone_texts():
    texts = []
    for char, usages in _POLY_DATA.items():
        group = []
        for py, sentence in usages:
            group.append(LQ + char + RQ + "在这里读" + py + "：" + sentence)
        intro = LQ + char + RQ + "是一个多音字，有不同的读法。"
        texts.append(intro + "".join(group))
        for _, sentence in usages:
            texts.append(sentence)
    return texts


def gen_pinyin_texts(n=2000):
    texts = []
    chars = list(set(GRADE1))
    templates = [
        LQ + "{char}" + RQ + "的拼音是{py}。",
        "汉字" + LQ + "{char}" + RQ + "读作{py}。",
        "{char}（{py}）。",
    ]
    for char in chars:
        try:
            py_list = pinyin(char, style=Style.TONE)
            py_str = py_list[0][0]
            t = random.choice(templates)
            texts.append(t.format(char=char, py=py_str))
        except Exception:
            continue
    return texts[:n]


_HOMOPHONES = [
    ("做", "作", "zuo4", "「做」多用于具体动作，如做作业、做饭。「作」多用于抽象意义，如作文、工作。"),
    ("的", "地", "de", "「的」用在名词前面，如美丽的花。「地」用在动词前面，如快乐地跳。"),
    ("在", "再", "zai4", "「在」表示存在或正在，如我在学校。「再」表示又一次，如再见、再来一次。"),
    ("它", "他", "ta1", "「他」指男性的人，「它」指动物或事物。"),
    ("坐", "座", "zuo4", "「坐」是动词，表示坐下的动作。「座」是名词，表示座位。"),
    ("以", "已", "yi3", "「以」表示用、拿，如以后、可以。「已」表示已经，如已经完成。"),
    ("像", "象", "xiang4", "「像」表示相似，如好像。「象」指大象这种动物。"),
    ("带", "戴", "dai4", "「带」指携带或带子，如带上书包。「戴」指穿戴在身上，如戴帽子。"),
    ("近", "进", "jin4", "「近」表示距离短，如离学校很近。「进」表示进入，如进门。"),
    ("圆", "园", "yuan2", "「圆」表示圆形，如圆圈。「园」表示园子，如花园、公园。"),
    ("青", "清", "qing1", "「青」指青色，如青草。「清」指清楚或干净，如清水。"),
    ("望", "忘", "wang4", "「望」表示看或希望，如看望。「忘」表示忘记，如忘了带书。"),
]


def gen_homophone_texts():
    texts = []
    for c1, c2, py, explanation in _HOMOPHONES:
        text = (LQ + c1 + RQ + "和" + LQ + c2 + RQ +
                "的读音相近，都读" + py + "，但意思不同。" + explanation)
        texts.append(text)
    return texts


_STRUCTURES = [
    ("明", "左右结构", "左边是「日」，右边是「月」，日月合在一起就是明亮的意思。"),
    ("休", "左右结构", "一个人靠在树旁休息，所以「休」表示休息。"),
    ("林", "左右结构", "两棵树并排在一起就是树林。"),
    ("森", "上下结构", "三棵树在一起就是森林。"),
    ("看", "上下结构", "把手放在眼睛上方向远处看。"),
    ("男", "上下结构", "在田里出力干活的人。"),
    ("花", "上下结构", "草字头下面加「化」，植物变化开放就是花。"),
    ("想", "上下结构", "上面是「相」，下面是「心」，心里所想。"),
    ("好", "左右结构", "女子在一起就是好。"),
    ("妈", "左右结构", "左边是「女」，右边是「马」。"),
    ("从", "左右结构", "两个人一前一后就是跟从。"),
    ("众", "上下结构", "三个人在一起就是众多。"),
    ("尖", "上下结构", "上面小下面大就是尖。"),
    ("笔", "上下结构", "竹字头加毛就是笔。"),
    ("灭", "独体结构", "火上面盖一横就是灭火。"),
    ("回", "全包围结构", "大口套小口就是回。"),
    ("闪", "半包围结构", "人在门里一闪而过。"),
    ("问", "半包围结构", "门里有口就是问。"),
    ("闻", "半包围结构", "门里有耳就是听闻。"),
    ("闷", "半包围结构", "心在门里出不去就是闷。"),
    ("鲜", "左右结构", "鱼和羊在一起就是鲜美。"),
    ("尘", "上下结构", "小小的土就是灰尘。"),
    ("泪", "左右结构", "眼睛里流出的水就是泪。"),
    ("岩", "上下结构", "山上的石头就是岩石。"),
]


def gen_structure_texts():
    texts = []
    for char, struct, desc in _STRUCTURES:
        texts.append(LQ + char + RQ + "字是" + struct + "，" + desc)
    return texts


def main():
    chaizi = load_chaizi()

    all_texts = []

    print("生成部首拆解文本...")
    t = gen_chaizi_texts(chaizi, n=3000)
    all_texts.extend(t)
    print(f"  {len(t)} 条")

    print("生成多音字语境文本...")
    t = gen_polyphone_texts()
    all_texts.extend(t)
    print(f"  {len(t)} 条")

    print("生成拼音知识文本...")
    t = gen_pinyin_texts(n=2000)
    all_texts.extend(t)
    print(f"  {len(t)} 条")

    print("生成同音字辨析文本...")
    t = gen_homophone_texts()
    all_texts.extend(t)
    print(f"  {len(t)} 条")

    print("生成汉字结构描述文本...")
    t = gen_structure_texts()
    all_texts.extend(t)
    print(f"  {len(t)} 条")

    random.shuffle(all_texts)

    output_path = OUT_DIR / "hanzi_corpus.jsonl"
    total_chars = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            total_chars += len(text)
            record = {
                "text": text,
                "category": "hanzi-knowledge",
                "data-source": "self-built",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n已保存到 {output_path}")
    print(f"总条数: {len(all_texts)}")
    print(f"总字符数: {total_chars:,}")
    est_tokens = int(total_chars * 0.6)
    print(f"估计token数: ~{est_tokens:,}")


if __name__ == "__main__":
    main()
