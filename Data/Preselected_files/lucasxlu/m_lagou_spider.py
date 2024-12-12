"""
a web spider for mobile lagou
"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python
import datetime
import os
import random
import re
import sys
import time
import pandas as pd

import requests
from requests import Session

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from util.excel_helper import mkdirs_if_not_exists
from util.file_reader import parse_job_xml
from util import log

try:
    from urllib import parse as parse

except:
    import urllib as parse

    sys.reload()
    sys.setdefaultencoding('utf-8')


def init_cookies():
    """
    return the cookies after your first visit
    """
    headers = {
        'Upgrade-Insecure-Requests': '1',
        'Host': 'm.lagou.com',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1',
        'DNT': '1',
        'Cache-Control': 'max-age=0',
        'Referrer Policy': 'no-referrer-when-downgrade',
    }
    url = 'https://m.lagou.com/search.html'
    response = requests.get(url, headers=headers, timeout=10)

    return response.cookies


def visit_logo_static(logo_static_url):
    headers = {
        'DNT': '1',
        'Referer': 'https://m.lagou.com/search.html',
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1'
    }

    response = requests.get(logo_static_url, headers=headers, timeout=20, cookies=init_cookies())
    if response.status_code == 200:
        print('visit logo static URL successfully.')
    else:
        print('Error occurs during visiting logo static URL. The ERROR_CODE is {0}.'.format(response.status_code))


def crawl_jobs(positionName):
    """
    crawl the job info from lagou H5 web pages
    """
    JOB_DATA = list()
    max_page_number = get_max_pageNo(positionName)
    log.info("%s, There are %s pages, approximately %s records in total.", positionName, max_page_number,
             max_page_number * 15)

    session = Session()

    # init cookies
    cookie = init_cookies()
    # cookie = dict(
    #     cookies_are='')

    df = pd.read_excel('./Proxy.xlsx')
    df = df[df['Delay'] < 1]
    print(df)
    ips = df['IP']
    https = df['HTTP']
    ports = df['Port']

    for i in range(1, max_page_number + 1):
        request_url = 'https://m.lagou.com/search.json?city=%E5%85%A8%E5%9B%BD&positionName=' + parse.quote(
            positionName) + '&pageNo=' + str(i) + '&pageSize=15'
        headers = {
            'Accept': 'application/json',
            'DNT': '1',
            'Referer': 'https://m.lagou.com/search.html',
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1',
        }

        if not os.path.exists('./Proxy.xlsx'):
            print('Proxy.xlsx not exists. Please run proxy_crawler.py to collect IP proxy first!')
            sys.exit(0)

        idx = random.randint(0, len(ips))
        proxies = {
            https[idx].lower(): 'http://{0}:{1}'.format(ips[idx], ports[idx]) if https[idx].lower() == 'http'
            else 'https://{0}:{1}'.format(ips[idx], ports[idx])
        }

        print('========using Proxy: {0}========'.format(proxies))

        # response = session.get(url=request_url, headers=headers, cookies=cookie, proxies=proxies)
        response = requests.get(request_url, headers=headers, cookies=cookie, proxies=proxies)

        # update cookies after first visit
        # cookie = response.cookies
        # cookie = dict(cookies_are='')

        if response.status_code == 200:
            try:
                items = response.json()['content']['data']['page']['result']
                if len(items) > 0:
                    for each_item in items:
                        if "今天" in each_item['createTime']:
                            each_item['createTime'] = re.sub("今天.*", str(datetime.date.today()),
                                                             each_item['createTime'])
                        elif "昨天" in each_item['createTime']:
                            today = datetime.date.today()
                            oneday = datetime.timedelta(days=1)
                            yesterday = today - oneday
                            each_item['createTime'] = re.sub("昨天.*", str(yesterday), each_item['createTime'])

                        JOB_DATA.append([each_item['positionId'], each_item['positionName'], each_item['city'],
                                         each_item['createTime'], each_item['salary'], each_item['companyId'],
                                         each_item['companyName'], each_item['companyFullName']])
                        print(each_item)
                    print('crawling page %d done...' % i)
                    time.sleep(random.randint(5, 10))
                else:
                    break
            except Exception:
                print('Error occurs during visiting Lagou. The ERROR_CODE is {0}. Return: {1}'.format(
                    response.status_code, response.text))
                pass
        elif response.status_code == 403:
            log.error('request is forbidden by the server...')
        else:
            log.error(response.status_code)

    return JOB_DATA


def get_max_pageNo(positionName):
    """
    return the max page number of a specific job
    """
    request_url = 'https://m.lagou.com/search.json?city=%E5%85%A8%E5%9B%BD&positionName=' + parse.quote(
        positionName) + '&pageNo=1&pageSize=15'
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Host': 'm.lagou.com',
        'Referer': 'https://m.lagou.com/search.html',
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/600.1.3 (KHTML, like Gecko) '
                      'Version/8.0 Mobile/12A4345d Safari/600.1.4',
        'X-Requested-With': 'XMLHttpRequest',
        'Connection': 'keep-alive'
    }
    response = requests.get(request_url, headers=headers, cookies=init_cookies(), timeout=10)
    print("Getting data from %s successfully. URL: " % positionName + request_url)
    if response.status_code == 200:
        max_page_no = int(int(response.json()['content']['data']['page']['totalCount']) / 15 + 1)

        return max_page_no
    elif response.status_code == 403:
        log.error('request is forbidden by the server...')

        return 0
    else:
        log.error(response.status_code)

        return 0


if __name__ == '__main__':
    craw_job_list = parse_job_xml('../config/job.xml')
    for _ in craw_job_list:
        joblist = crawl_jobs(_)
        col = [
            u'职位编码',
            u'职位名称',
            u'所在城市',
            u'发布日期',
            u'薪资待遇',
            u'公司编码',
            u'公司名称',
            u'公司全称']
        df = pd.DataFrame(joblist, columns=col)
        dir = "./data/"
        mkdirs_if_not_exists(dir)
        df.to_excel(os.path.join(dir, _ + ".xlsx"), sheet_name=_, index=False)
