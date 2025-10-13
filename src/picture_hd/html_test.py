from bs4 import BeautifulSoup
import json

html_content = """
<html>
        <head>
            <title>小企业财务报表利润表</title>
            <meta charset="UTF-8"/>
            <meta charset="UTF-8"/>
            <link rel="stylesheet" href="../../../sbbcx/styles/forView.css"/>
        </head>
        <body>
            <div>
                <table width="780" align="center">
                    <tr>
                        <td width='100' align="right"></td>
                        <td width='580'>
                            <div align="center">
                                <b class="bbbt">利　　润　　表</b>
                            </div>
                        </td>
                    </tr>
                </table>
                <table width="780" align="center">
                    <tr>
                        <td width="50%">纳税人识别号：91441900MA53U1N60Y</td>
                        <td width="40%">税款所属期：2022-01-01 至 2022-12-31</td>
                        <td align="right">会小企02表</td>
                    </tr>
                    <tr>
                        <td align="left">纳税人名称：东莞市圣煜服饰有限公司</td>
                        <td>报送日期：2023-01-11</td>
                        <td align="right">单位：元</td>
                    </tr>
                </table>
                <table id="tabList" width="780" class="border" cellspacing="0" cellpadding="0" align="center">
                    <tr>
                        <td width="340" align="center">项目</td>
                        <td width="100" align="center" class="nowrap">行次</td>
                        <td width="170" align="center">本年累计金额</td>
                        <td width="170" align="center">上年金额</td>
                    </tr>
                    <tr>
                        <td>一、营业收入</td>
                        <td align="center">1</td>
                        <td align="right" class="jebg">1,116,578.14</td>
                        <td align="right" class="jebg">95,752.23</td>
                    </tr>
                    <tr>
                        <td>　　减：营业成本</td>
                        <td align="center">2</td>
                        <td align="right" class="jebg">1,074,752.81</td>
                        <td align="right" class="jebg">57,632.28</td>
                    </tr>
                    <tr>
                        <td>　　税金及附加</td>
                        <td align="center">3</td>
                        <td align="right" class="jebg">1,732.80</td>
                        <td align="right" class="jebg">167.90</td>\t
                    </tr>
                    <tr>
                        <td>　　其中：消费税</td>
                        <td align="center">4</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　营业税</td>
                        <td align="center">5</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　城市维护建设税</td>
                        <td align="center">6</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　资源税</td>
                        <td align="center">7</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　土地增值税</td>
                        <td align="center">8</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　城镇土地使用税、房产税、车船税、印花税</td>
                        <td align="center">9</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　教育费附加、矿产资源补偿费、排污费</td>
                        <td align="center">10</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　销售费用</td>
                        <td align="center">11</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　其中：商品维修费</td>
                        <td align="center">12</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　广告费和业务宣传费</td>
                        <td align="center">13</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　管理费用</td>
                        <td align="center">14</td>
                        <td align="right" class="jebg">55,480.00</td>
                        <td align="right" class="jebg">54,000.00</td>
                    </tr>
                    <tr>
                        <td>　　其中：开办费</td>
                        <td align="center">15</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　业务招待费</td>
                        <td align="center">16</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　研究费用</td>
                        <td align="center">17</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　财务费用</td>
                        <td align="center">18</td>
                        <td align="right" class="jebg">1,179.64</td>
                        <td align="right" class="jebg">1,037.18</td>
                    </tr>
                    <tr>
                        <td>　　其中：利息费用（收入以“-”号填列）</td>
                        <td align="center">19</td>
                        <td align="right" class="jebg">-86.36</td>
                        <td align="right" class="jebg">-22.82</td>
                    </tr>
                    <tr>
                        <td>　　加：投资收益（损失以“-”号填列）</td>
                        <td align="center">20</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>二、营业利润（亏损以“-”号填列）</td>
                        <td align="center">21</td>
                        <td align="right" class="jebg">-16,567.11</td>
                        <td align="right" class="jebg">-17,085.13</td>
                    </tr>
                    <tr>
                        <td>　　加：营业外收入</td>
                        <td align="center">22</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　其中：政府补助</td>
                        <td align="center">23</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　减：营业外支出</td>
                        <td align="center">24</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　其中：坏账损失</td>
                        <td align="center">25</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　无法收回的长期债券投资损失</td>
                        <td align="center">26</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　无法收回的长期股权投资损失</td>
                        <td align="center">27</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　自然灾害等不可抗力因素造成的损失</td>
                        <td align="center">28</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>　　税收滞纳金</td>
                        <td align="center">29</td>
                        <td align="right" class="jebg">0.00</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>三、利润总额（亏损总额以“-”号填列）</td>
                        <td align="center">30</td>
                        <td align="right" class="jebg">-16,567.11</td>
                        <td align="right" class="jebg">-17,085.13</td>
                    </tr>
                    <tr>
                        <td>　　减：所得税费用</td>
                        <td align="center">31</td>
                        <td align="right" class="jebg">550.58</td>
                        <td align="right" class="jebg">0.00</td>
                    </tr>
                    <tr>
                        <td>四、净利润（净亏损以“-”号填列）</td>
                        <td align="center">32</td>
                        <td align="right" class="jebg">-17,117.69</td>
                        <td align="right" class="jebg">-17,085.13</td>
                    </tr>
                </table>
            </div>
            <meta charset="UTF-8"/>
            <div style="text-align: center;padding: 10px 0;" id="printClickBtn"></div>
        </body>
    </html>
"""

# 解析 HTML 内容
soup = BeautifulSoup(html_content, 'lxml')

# 找到表格
table = soup.find('table', {'id': 'tabList'})

# 初始化结果列表
result = []
projectNameCode = 120108
# 遍历表格行
for row in table.find_all('tr')[1:]:  # 跳过表头
    cols = row.find_all('td')
    project_name = cols[0].get_text(strip=True)
    sequence = cols[1].get_text(strip=True)
    current_year_accumulative_amount = cols[2].get_text(strip=True).replace(',', '')
    last_year_amount = cols[3].get_text(strip=True).replace(',', '')

    # 构建 JSON 对象
    json_obj = {
        "projectName": project_name,
        "projectNameCode" : str(projectNameCode),
        "sequence": int(sequence),
        "columnSequence": int(sequence)
    }
    result.append(json_obj)
    projectNameCode = projectNameCode + 1

# 转换为 JSON 字符串
json_result = json.dumps(result, ensure_ascii=False, indent=4)
print(json_result)