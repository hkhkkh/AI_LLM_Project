import json
import os
import sqlite3

# 导入统一配置
from config import DB_FILE, TRAIN_DATA_LARGE

"""
阶段六：构建更大的企业知识库数据库 + JSONL 训练集
----------------------------------------------------
这个脚本会：
1. 重建 company_data.db（覆盖旧的 FAQ 内容，生成更丰富的知识库）
2. 向 FAQ 表写入 45 条覆盖多个业务场景的问答
3. 导出更大的训练数据集 train_data_large.jsonl
运行：python step6_create_large_db.py
"""

db_file = DB_FILE
jsonl_file = TRAIN_DATA_LARGE

# 如已存在旧文件，先删除（方便反复演示）
if os.path.exists(db_file):
    os.remove(db_file)

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS faq (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL
    )
    """
)

# 更丰富的企业知识库（可根据需要再扩展）
dataset = [
    # 公司制度
    ("General", "FutureAI的使命是什么？", "FutureAI致力于打造面向中小企业的本地化AI解决方案，让每一家企业都拥有自己的智能助手。"),
    ("General", "FutureAI的核心价值观有哪些？", "坦诚协作、快速迭代、以客户价值为中心、对结果负责。"),
    ("General", "FutureAI的办公地点在哪？", "总部位于上海浦东张江AI产业园，另在深圳与成都设有研发中心。"),
    ("General", "FutureAI每周的All Hands是什么时候？", "每周三下午4点，通过Teams直播，同时在上海总部设现场会场。"),
    # 人力资源
    ("HR", "员工到岗时间怎么规定？", "弹性工作制，核心时间10:00-16:00，在此时间段需保证在线与可联络。"),
    ("HR", "试用期多长？", "所有正式员工试用期为3个月，表现优秀可提前转正。"),
    ("HR", "绩效考核频率是什么？", "实行季度绩效评估，年度末会综合评估晋升机会。"),
    ("HR", "年终奖怎么发放？", "年终奖基于年度绩效结果与公司利润情况，通常在春节前发放。"),
    ("HR", "员工可以申请远程办公吗？", "支持部分岗位长期远程，但需直属主管批准，并在HR系统备案。"),
    ("HR", "新员工入职需要准备什么？", "请提前提交身份证复印件、银行账户信息，并完成线上入职培训。"),
    # 福利与假期
    ("Benefits", "公司提供哪些补贴？", "提供交通补贴、午餐补贴、手机通讯补贴及年度体检。"),
    ("Benefits", "年假制度是怎样的？", "入职一年及以下10天，之后每满一年增加1天，最高20天。"),
    ("Benefits", "加班如何申请调休？", "需在OA系统提交加班申请并由主管审批，通过后可在两个月内调休。"),
    ("Benefits", "公司有节日礼品吗？", "重要传统节日及员工生日会发放礼品或购物卡。"),
    # 财务与报销
    ("Finance", "差旅报销流程是什么？", "在OA系统填写差旅报销单，上传发票照片，经主管与财务审核后，款项将在5个工作日内到账。"),
    ("Finance", "报销单最晚多久提交？", "请在费用发生后30天内提交，否则需要额外说明。"),
    ("Finance", "可报销的打车费用标准是什么？", "市内打车建议单次不超过200元，超出需说明原因并附截图。"),
    ("Finance", "采购笔记本电脑需要走什么流程？", "请提交IT采购申请表，审批完成后由IT统一下单采购。"),
    ("Finance", "员工福利如何计税？", "按照国家相关政策，由财务统一计算个税并在工资中体现。"),
    # IT支持
    ("IT", "FutureAI公司的Wifi密码是多少？", "访客Wifi名为'FutureAI_Guest'，密码为'Welcome2025'。员工请使用个人账号连接'FutureAI_Secure'。"),
    ("IT", "GitLab访问不了怎么办？", "请先确认VPN已连接；若仍失败，提交IT工单附上报错截图。"),
    ("IT", "邮箱密码忘了怎么重置？", "在内网自助服务门户点击“忘记密码”，通过手机号或工牌号验证即可。"),
    ("IT", "VPN账号申请流程是什么？", "新员工默认开通VPN权限，如需额外地域节点，请向IT申请。"),
    ("IT", "公司默认使用什么代码托管平台？", "统一使用内部部署的GitLab，访问地址为git.futureai.local。"),
    ("IT", "提交IT工单平均多久响应？", "工作时间内2小时内响应，非工作时间会在下一个工作日跟进。"),
    # 安全与合规
    ("Security", "访客进入办公室需要哪些步骤？", "需提前在OA预约，前台核验身份证后发放临时胸牌，由员工全程陪同。"),
    ("Security", "员工离职后账号多久关闭？", "人力在系统确认离职后24小时内，IT会统一停用账号与门禁。"),
    ("Security", "数据泄露应急流程是什么？", "第一时间通知直属主管与安全应急小组，并填写事故报告，24小时内提交初步分析。"),
    ("Security", "可以将代码同步到个人Git仓库吗？", "禁止私自拷贝或同步公司代码到外部仓库，如有特殊需求需CTO审批。"),
    # 办公与行政
    ("Facilities", "工位怎么预定？", "使用“FutureSeat”小程序提前一天预约，默认每人可预定两个时段。"),
    ("Facilities", "会议室最多可提前多久预定？", "可提前两周在Outlook中预定，超过2小时的会议需行政审批。"),
    ("Facilities", "空调温度谁负责调节？", "各区域空调受中央控制，如需调节请联系行政值班。"),
    ("Facilities", "夜间加班可以申请班车吗？", "21:30后可在OA申请夜班车，行政会统一安排。"),
    # 文化活动
    ("Culture", "FutureAI有哪些兴趣小组？", "目前有篮球、羽毛球、摄影、读书以及AI黑客松小组，欢迎加入。"),
    ("Culture", "公司吉祥物叫什么？", "我们的吉祥物是一只名叫'CodeCat'的机械猫，它最喜欢吃显卡。"),
    ("Culture", "员工分享会在哪里报名？", "在钉钉群“FutureAI Talk”里填写报名表，运维同事会安排档期。"),
    ("Culture", "内推奖励政策是什么？", "推荐候选人通过试用期后，可获得5000元现金奖励或等值礼品卡。"),
    # 业务与产品
    ("Product", "FutureAI最主要的产品形态有哪些？", "包括本地部署AI助手、行业知识库构建工具以及LoRA微调训练平台。"),
    ("Product", "客户常见的部署周期多长？", "标准项目4-6周可上线，复杂定制平均8-10周。"),
    ("Product", "支持哪些数据库类型接入？", "支持PostgreSQL、MySQL、SQL Server、SQLite以及部分国产数据库。"),
    ("Product", "和客户数据同步如何保障安全？", "采用VPC专线+TLS加密传输，敏感字段支持加密脱敏。"),
    ("Product", "可以部署在客户离线环境吗？", "可以，我们提供离线安装包并支持自定义源。"),
    ("Product", "和客户现有CRM系统如何集成？", "通过REST API或消息队列进行数据同步，支持Webhook回调。"),
    # 组织结构
    ("Org", "FutureAI的创始人是谁？", "FutureAI由神秘的开发者'Y'于2024年创立，目前担任CEO。"),
    ("Org", "CTO负责哪些团队？", "CTO负责AI研究、平台工程、基础架构与DevOps团队。"),
    ("Org", "数据科学团队目前有多少人？", "共有15人，涵盖NLP、CV、时序预测与数据分析方向。"),
    ("Org", "客户成功团队的主要职责是什么？", "负责客户培训、上线支持以及NPS满意度回访。"),
]

cursor.executemany(
    "INSERT INTO faq (category, question, answer) VALUES (?, ?, ?)",
    dataset
)
conn.commit()
print(f"✅ 已写入 {len(dataset)} 条知识库记录到 {db_file}")

# 导出 JSONL 训练集（更大的微调数据）
with open(jsonl_file, "w", encoding="utf-8") as f:
    for category, question, answer in dataset:
        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "你是FutureAI公司的智能助手，必须根据内部知识库准确回答员工问题。"
                },
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ],
            "category": category
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 训练数据已生成: {jsonl_file}")
print("-" * 30)
print("数据预览 (前3条):")
with open(jsonl_file, "r", encoding="utf-8") as f:
    for _ in range(3):
        print(f.readline().strip())
print("-" * 30)
print("阶段六完成！请使用 train_data_large.jsonl 进行微调，或直接用数据库做 RAG。")

conn.close()
