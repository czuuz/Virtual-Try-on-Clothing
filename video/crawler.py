# pip install DrissionPage
# pip install requests

import time
import csv
import os
import requests
import random
from DrissionPage._pages.chromium_page import ChromiumPage
import datetime as dt
page = ChromiumPage()


def timestamp_to_timestr(timestamp):
    return dt.datetime.fromtimestamp(timestamp)


def video():
    os.makedirs('./file', exist_ok=True)
    for res in page.listen.steps(timeout=5):
        try:
            datas = res.response.body.get('data')
            if datas is None:
                continue
            else:
                for i in datas:
                    aweme_info = i['aweme_info']
                    author = aweme_info['author']['nickname']
                    enterprise_verify_reason = aweme_info['author']['enterprise_verify_reason']
                    custom_verify = aweme_info['author']['custom_verify']
                    try:
                        signature = aweme_info['author']['signature'].replace('\n', '')
                    except:
                        signature = ''
                    author_user_id = aweme_info['author_user_id']
                    aweme_id = aweme_info['aweme_id']
                    video_url = f'https://www.douyin.com/video/{aweme_id}'
                    sec_uid = aweme_info['author']['sec_uid']
                    user_url = f'https://www.douyin.com/user/{sec_uid}?relation=0&vid={aweme_id}'
                    follower_count = aweme_info['author']['follower_count']
                    desc = aweme_info['desc'].replace('\n', '')
                    tim = timestamp_to_timestr(aweme_info['create_time'])
                    collect_count = aweme_info['statistics']['collect_count']
                    comment_count = aweme_info['statistics']['comment_count']
                    digg_count = aweme_info['statistics']['digg_count']
                    download_count = aweme_info['statistics']['download_count']
                    share_count = aweme_info['statistics']['share_count']
                    try:
                        video_url = aweme_info['video']['play_addr']['url_list'][-1]
                    except:
                        video_url = ''
                    with open('数据.csv', 'a+', encoding='utf-8-sig', newline='') as fi:
                        fi = csv.writer(fi)
                        fi.writerow(
                            [author] + [enterprise_verify_reason] + [video_url] + [custom_verify] + [signature] + [author_user_id] + [user_url] + [aweme_id] +
                            [follower_count] + [desc] + [str(tim)] + [collect_count] + [comment_count] + [digg_count] +
                            [download_count] + [share_count] + [video_url]
                        )
        except Exception as e:
            print(e)
            continue

def start():
    # 注释：使用你自己的链接
    page.get('https://www.douyin.com/search/%E5%A5%B3%E8%A3%85?aid=37485c76-d20d-4909-923d-8dd7699d5d4a&type=video')
    time.sleep(1)
    for num in range(3):
        page.ele('xpath://span[text()=\'筛选\']').click(by_js=True)
        time.sleep(1)
    page.ele('xpath://span[text()=\'5分钟以上\']').click(by_js=True)  # 注释：可以改为\'1-5分钟\'
    page.listen.start('ttps://www.douyin.com/aweme/v1/web/search/item/?device_platform=webapp&aid=')
    # 分类
    trs = page.eles('xpath://div[@class="CbguaXge"]/div/div')
    if trs != None and trs != []:
        for tr in trs:
            tr.click(by_js=True)
            video()
            if '搜索结果为空' in page.html:
                break
            else:
                for num in range(1, 5): # 注释：控制爬取最多的页数，这里是5页
                    page.scroll.to_bottom()
                    time.sleep(10)
                    video()
                    if '暂时没有更多了' in page.html:
                        break

def get_csv():
    a = {}
    cd = csv.reader(open('数据.csv', 'r', encoding='utf-8'))
    for i in cd:
        url = i[2]
        a.update({url: i}) # 去重，字典的key不能重复
    for i in a.items():
        with open('清洗后数据.csv', 'a+', encoding='utf-8-sig', newline='') as fi:
            fi = csv.writer(fi)
            fi.writerow(i[1])

def down_video():
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36 Edg/92.0.902.78',
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.73",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:86.0) Gecko/20100101 Firefox/86.0",
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
        'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/82.0.4051.0 Safari/537.36 Edg/82.0.425.0',
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ]
    with open('清洗后数据.csv', 'r', encoding='utf-8') as file:
        cd = list(csv.reader(file))
    for i in cd:
        headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-language": "zh-CN,zh;q=0.9",
            "cache-control": "max-age=0",
            "priority": "u=0, i",
            "sec-ch-ua-mobile": "?0",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": random.choice(USER_AGENTS)
        }
        url = i[-1]
        print(url)
        uid = i[7]
        if 'http' in url:
            for num in range(3):
                time.sleep(2)
                try:
                    response = requests.get(url, headers=headers, timeout=180)
                    print(response)
                    if response.status_code == 200:
                        with open(f'./file/{uid}.mp4', 'wb') as f:
                            f.write(response.content)
                        break
                except Exception as e:
                    print(e)
                    continue


# start()
# get_csv()
down_video()