# -*- coding: utf-8 -*-

import airflow
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.hooks.ssh_hook import SSHHook
from airflow.contrib.operators.ssh_operator import SSHOperator
from datetime import datetime, timedelta, date
from airflow.utils.trigger_rule import TriggerRule

import urllib3

urllib3.disable_warnings()

# DAGs 의 성공/실패 정보를 Kafka API 서버에 전달하는 Custom Operator import
try:
    from KafkaRestApiCallOperator import KafkaRestApiCallOperator
except ImportError:
    from airflow.operators import KafkaRestApiCallOperator


sshHook_imply_3113 = SSHHook(
    ssh_conn_id='imply_3113'
)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': airflow.utils.dates.days_ago(2),
    'email': [''],
    'email_on_failure': False,
    'email_on_retry': False,
}

# schedule_interval=timedelta(days=1)
# * * * * *
# 분 시 일 월 요일
dag = DAG(
    'elasticsearch_to_mssql',
    default_args=default_args,
    description='Elasticsearch xAPI 데이터 -> MainDB 적재',
    schedule_interval='20 17 * * *'
    # schedule_interval='52 07 * * *'
)


command_t1 = ""

command_t2 = ""

param_date = (datetime.utcnow() + timedelta(hours=+9)).strftime("%Y-%m-%d")

# 처음 시작되는 task의 시작 시간을 조회하여, xcom 영역에 저장한다.
def get_start_info(context):
    context['ti'].xcom_push(key="start_info", value={
        "dag_id": context['ti'].dag_id,
        "hostname": str(context['ti'].hostname),
        "start_time": context['ti'].start_date.strftime("%Y-%m-%dT%H:%M:%S")
    })


# 성공/실패 시  Kafka Api 서버에 결과 값을 전달 한다.
def result_msg(context):
    kafka_msg_send = KafkaRestApiCallOperator(task_id="kafka_restapi_msg_send", endpoint="/kafka_api")
    kafka_msg_send.execute(context)


t1 = SSHOperator(
    task_id='xapi_search_keyword',
    command=command_t1,
    ssh_hook=sshHook_imply_3113,
    on_failure_callback=result_msg,  # 실패 시 정보전달
    on_success_callback=get_start_info,  # 성공 시 시작시간 정보 저장
    params={'DATE': param_date},
    dag=dag)

t2 = SSHOperator(
    task_id='xapi_flow_object',
    command=command_t2,
    ssh_hook=sshHook_imply_3113,
    on_failure_callback=result_msg,  # 실패 시 정보전달
    on_success_callback=result_msg,
    dag=dag)

t1.set_downstream(t2)