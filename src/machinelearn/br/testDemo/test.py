import json
import os
from datetime import datetime
from typing import Dict, List, Any

def main():
    """
    主函数：处理支出交易数据并进行分类分析

    @return: 分类结果字典
    """
    # 从content.input1.支出类提取支出数据
    data = params.get("input1", {})
    # 获取总费用金额
    expense_total_amount_str = params.get("expense_total_amount", "0")
    try:
        expense_total_amount = float(expense_total_amount_str)
    except (ValueError, TypeError):
        expense_total_amount = 0.0

    # 调用分类函数，传入总金额用于计算占比
    result = classify_expenses(data, expense_total_amount)
    return {"result": result}


def parse_amounts(amount_point: str) -> List[float]:
    """
    解析金额字符串，将分号分隔的金额转换为浮点数列表

    @param amount_point: 包含分号分隔金额的字符串，例如 "60.00;123.10"
    @return: 金额列表，例如 [60.0, 123.1]
    """
    if not amount_point:
        return []
    amounts = amount_point.split(';')
    return [float(amount.strip()) for amount in amounts if amount.strip()]


def is_large_expense(amount_point: str) -> bool:
    """
    判断是否为大额支出（单笔金额 >= 50000元）

    @param amount_point: 金额字符串
    @return: 如果存在单笔金额 >= 50000元，返回True，否则返回False
    """
    amounts = parse_amounts(amount_point)
    return any(amount >= 50000 for amount in amounts)


def matches_keywords(value: str, keywords: List[str]) -> bool:
    """
    检查交易描述中是否包含指定的关键词

    @param value: 交易描述字符串
    @param keywords: 关键词列表
    @return: 如果包含任一关键词，返回True，否则返回False
    """
    value_lower = value.lower()
    return any(keyword.lower() in value_lower for keyword in keywords)


def is_investment_expense(value: str) -> bool:
    """
    判断是否为投资理财支出

    @param value: 交易描述字符串
    @return: 如果匹配投资理财关键词，返回True，否则返回False
    """
    investment_keywords = [
        '证券', '证券账户转入', '证券有限公司',
        '投资', '理财', '基金', '账户转入',
        '华泰证券', '国信证券', '广发证券', '招商证券', '海通证券'
    ]
    return matches_keywords(value, investment_keywords)


def is_fixed_expense(value: str) -> bool:
    """
    判断是否为固定支出

    @param value: 交易描述字符串
    @return: 如果匹配固定支出关键词，返回True，否则返回False
    """
    fixed_keywords = ['房租', '房贷', '物业费', '住房租赁']
    return matches_keywords(value, fixed_keywords)


def is_variable_expense(value: str) -> bool:
    """
    判断是否为变动支出

    @param value: 交易描述字符串
    @return: 如果匹配变动支出关键词，返回True，否则返回False
    """
    variable_keywords = [
        # 餐饮相关
        '蜜雪冰城', '餐饮', '外卖',
        # 交通相关
        '滴滴出行', '打车', '交通',
        # 购物娱乐
        '购物', '娱乐', '消费',
        # 服务咨询
        '咨询服务费', '咨询'
    ]
    return matches_keywords(value, variable_keywords)


def calculate_total_amount(items: List[Dict[str, str]]) -> float:
    """
    计算交易列表的总金额

    @param items: 交易字典列表
    @return: 总金额（浮点数）
    """
    total = 0.0
    for item in items:
        amounts = parse_amounts(item.get('amount_point', ''))
        total += sum(amounts)
    return total


def classify_expenses(expense_transactions: List[Dict[str, str]], total_amount: float = 0.0) -> Dict[str, Any]:
    """
    将支出交易分类为四种类型：大额支出、变动支出、固定支出、投资理财支出

    @param expense_transactions: 支出交易列表
    @param total_amount: 总金额，用于计算占比（如果为0则使用各类别金额之和）
    @return: 包含分类结果和统计信息的字典
    """
    large_expense_items = []
    variable_expense_items = []
    fixed_expense_items = []
    investment_expense_items = []

    # 遍历所有交易进行分类
    for transaction in expense_transactions:
        value = transaction.get('value', '')
        amount_point = transaction.get('amount_point', '')

        # 优先判断大额支出（单笔 >= 50000元，且不是投资理财）
        if is_large_expense(amount_point) and not is_investment_expense(value):
            large_expense_items.append(transaction)
        # 判断投资理财支出
        elif is_investment_expense(value):
            investment_expense_items.append(transaction)
        # 判断固定支出
        elif is_fixed_expense(value):
            fixed_expense_items.append(transaction)
        # 判断变动支出
        elif is_variable_expense(value):
            variable_expense_items.append(transaction)
        # 默认归类为变动支出
        else:
            variable_expense_items.append(transaction)

    # 计算各类别的总金额
    large_total = calculate_total_amount(large_expense_items)
    variable_total = calculate_total_amount(variable_expense_items)
    fixed_total = calculate_total_amount(fixed_expense_items)
    investment_total = calculate_total_amount(investment_expense_items)

    # 使用传入的总金额计算占比，如果未传入或为0，则使用各类别金额之和
    grand_total = total_amount if total_amount > 0 else (large_total + variable_total + fixed_total + investment_total)

    # 计算各类别占比
    large_percentage = (large_total / grand_total * 100) if grand_total > 0 else 0
    variable_percentage = (variable_total / grand_total * 100) if grand_total > 0 else 0
    fixed_percentage = (fixed_total / grand_total * 100) if grand_total > 0 else 0
    investment_percentage = (investment_total / grand_total * 100) if grand_total > 0 else 0

    def format_amount(amount: float) -> str:
        """格式化金额为字符串，保留两位小数"""
        return f"{amount:.2f}"

    def format_percentage(percentage: float) -> str:
        """格式化百分比为字符串，保留两位小数"""
        return f"{percentage:.2f}%"

    # 构建结果字典
    result = {
        "large_expense": {
            "items": large_expense_items,
            "total_amount": format_amount(large_total),
            "percentage": format_percentage(large_percentage),
            "chinese_description": f"大额支出：共{len(large_expense_items)}笔，总金额{format_amount(large_total)}元，占比{format_percentage(large_percentage)}"
        },
        "variable_expense": {
            "items": variable_expense_items,
            "total_amount": format_amount(variable_total),
            "percentage": format_percentage(variable_percentage),
            "chinese_description": f"变动支出：共{len(variable_expense_items)}笔，总金额{format_amount(variable_total)}元，占比{format_percentage(variable_percentage)}"
        },
        "fixed_expense": {
            "items": fixed_expense_items,
            "total_amount": format_amount(fixed_total),
            "percentage": format_percentage(fixed_percentage),
            "chinese_description": f"固定支出：共{len(fixed_expense_items)}笔，总金额{format_amount(fixed_total)}元，占比{format_percentage(fixed_percentage)}"
        },
        "investment_expense": {
            "items": investment_expense_items,
            "total_amount": format_amount(investment_total),
            "percentage": format_percentage(investment_percentage),
            "chinese_description": f"投资理财支出：共{len(investment_expense_items)}笔，总金额{format_amount(investment_total)}元，占比{format_percentage(investment_percentage)}"
        }
    }

    return result





if __name__ == "__main__":
    # 从testDemo/input.json读取参数
    input_file = os.path.join(os.path.dirname(__file__), "", "input.json")
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 创建模拟的params字典
    global params
    params = test_data.get("content", {})

    # 调用main函数
    print("--- 开始计算支出费用 ---")
    result = main()

    # 打印结果
    print("\n--- 计算结果 ---")
    print(f"\n完整结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
