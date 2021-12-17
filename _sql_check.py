# -*- coding: utf-8 -*-

import ssl
import sys
import datetime
from datetime import timedelta
from class_mssql import *
from elasticsearch import Elasticsearch
from elasticsearch.connection import create_ssl_context

    ###################################시간 수정########################################

yesterday = (datetime.datetime.utcnow() + timedelta(days=-1)).strftime("%Y-%m-%d")
today = (datetime.datetime.utcnow()).strftime("%Y-%m-%d")

gv_mssql_insert = """
INSERT INTO dw.dbo.tdw_stat_tag_searched (
    base_date,
    company_seq,
    user_id,    
    keyword,
    url,
    reg_date
)
VALUES (
'{}', %s, %s, %s, %s, %s
)
""".format(yesterday)


def elastic_query():
    ssl_context = create_ssl_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    _userpw = "elastic:###"
    _elastic = [
        'https://{}@###'.format(_userpw),
        'https://{}@###'.format(_userpw),
        'https://{}@###'.format(_userpw)
    ]

    _obj_elastic = Elasticsearch(
        _elastic,
        verify_certs=False,
        cs_certs=False,
        ssl_context=ssl_context,
    )

    _query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": yesterday,
                                "lt": today
                            }
                        }
                    },
                    {
                        "match": {
                              "verb.display.en-us": "searched"
                        }
                    },
                    {
                        "exists": {
                            "field": "object.definition.extensions.keyword"
                        }
                    }
                ]
            }
        },
        "_source": [
            "@timestamp",
            "object.definition.extensions.company_cd",
            "object.definition.extensions.keyword",
            "object.definition.extensions.url",
            "actor.account.name"
        ]
    }

    _res = _obj_elastic.search(
        index='experience-api-b2b-*',
        body=_query,
        scroll='60m',
        _source=[
            "@timestamp",
            "object.definition.extensions.company_cd",
            "object.definition.extensions.keyword",
            "object.definition.extensions.url",
            "actor.account.name"
        ],
        size=1000
    )

    _scroll_id = _res['_scroll_id']
    _scroll_size = _res['hits']['total']['value']

    _data_list = []
    while _scroll_size > 0:
        for row in _res['hits']['hits']:
            _data_list.append(row['_source'])
        _res = _obj_elastic.scroll(scroll_id=_scroll_id, scroll='60m')
        _scroll_id = _res['_scroll_id']
        _scroll_size = len(_res['hits']['hits'])

    # print(_data_list)

    search_result = []
    for i in _data_list:
        # keyword = i["object"]["definition"]["extensions"]["keyword"].encode().decode()
        search_result.append((
            i["object"]["definition"]["extensions"]["company_cd"],
            i['actor']['account']['name'],
            i["object"]["definition"]["extensions"]["keyword"],
            i["object"]["definition"]["extensions"]["url"][0:100],
            i['@timestamp'])
        )

    return search_result


def fn_mssql_insert(cDB, rv):
    # print(gv_mysql_insert % rv[])
    try:
        cDB.db_insert_many(gv_mssql_insert, rv)

    except Exception as e:
        print("Database insert Error !!")
        conn_ms_sql.db_close()
        sys.exit()


if __name__ == "__main__":
    mssql_conn_list = ['MAIN-DB-PRIMARY']
    # mssql_conn_list = ['MAIN-STG']

    try:
        for target in mssql_conn_list:
            conn_ms_sql = cMSSQL(target)

    except Exception as e:
        print("Database Connect Error !!")
        conn_ms_sql.db_close()
        sys.exit()

    
        ###########################################################################################

    # 0순위로 테스트 테이블 생성 후 gv_mssql_insert = """I블}
    rv = elastic_query()
    # 1. print 주석 해제해서 rv 값으로 뭐가 들어오는 지 확인!
    print('rv data :', rv)
    # 2. class cMSSQL: 의 def db_insert_many(self, query, data) try except 주석해제하고 query, data print 찍어서 확인해본다.
    # airflow 실행 후 log를 확인해본다.
    # 근데 문제점을 못찾는다? mssql 버전을 확인해봐야한다. 
    fn_mssql_insert(conn_ms_sql, rv)
