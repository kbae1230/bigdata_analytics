#-*- coding:utf-8 -*-
#!/usr/bin/python
import pymssql  # MSSQL
from configparser import SafeConfigParser


gv_path = "###"


# database command
class cMSSQL:
    def __init__(self, target):
        """

        :rtype:
        """
        host, port, user, passwd, dbname = db_config_get(target)
        # self.db = pymssql.connect(server=host, port=int(port), user=user,
        # password=passwd, database=dbname, autocommit=True,
        # charset="ISO-8859-1");
        self.db = pymssql.connect(server=host, port=int(
            port), user=user, password=passwd, database=dbname, autocommit=True, charset="utf8")
        # use_unicode=True,
        self.cursor = self.db.cursor()
   
    def con(self):
        return self.db

    def disCon(self):
 
        self.db.close()

    def print_db_version(self):
        self.cursor.execute("select version()")
        row = self.cursor.fetchone()
        print("server version : %s" % row[0])

    def db_select(self, query):
        self.cursor.execute(query)
        row = self.cursor.fetchone()
        return row

    def db_colname_selects(self, query):
        self.cursor.execute(query)
        names_of_column = [e[0] for e in self.cursor.description]
        row = self.cursor.fetchall()
        return(names_of_column, row)

    def db_callproc_colname_selects(self, query, parameter):
        self.cursor.callproc(query, (parameter,))
        print("1")
        # names_of_column = [e[0] for e in self.cursor.description]
        print("2")
        row = self.cursor.fetchall()
        print("3")
        print(row)
        return(names_of_column, row)
        # return row

    def db_selects(self, query):
        self.cursor.execute(query)
        row = self.cursor.fetchall()
        return row

    def db_selects_rowcount(self, query):
        self.cursor.execute(query)
        row = self.cursor.fetchall()
        return row

    def db_selects_rowcount(self, query):
        self.cursor.execute(query)
        row = self.cursor.fetchall()
        rowcount = self.cursor.rowcount
        return row, rowcount

    def db_selects_date(self, query, data):
        self.cursor.execute(query, data)
        row = self.cursor.fetchall()
        return row

    def db_colname_selects_data(self, query, data):
        self.cursor.execute(query, data)
        names_of_column = [e[0] for e in self.cursor.description]
        row = self.cursor.fetchall()
        return(names_of_column, row)

    def db_command(self, query):
        try:
            self.cursor.execute(query)

            return "1"
        except Exception as e:
            return str(e)

    # 한 행 입력 할 때 사용하자.
    def db_insert(self, query, data):
        self.cursor.execute(query, data)
 
    ###########################################################################################
    # 여러 행 입력할 때 사용하자.
    def db_insert_many(self, query, data):
        # 2. try return 값 확인으로 cursor가 작동하는 지 확인한다.
        try :
            print('query', query)
            print('data', data)
            self.cursor.executemany(query, data)
            return 1
        except:
            return 0

    def db_close(self):
        self.cursor.close()
        self.db.close()

    ############################print 뒤에 쉼표 수정 ################################################

    # print function
    def printData(rows):
        for r in rows:
            print (r)


# config read


def db_config_get(target):
    cConf = SafeConfigParser()
    cConf.read(gv_path + '/config.ini')
    db_conf_list = []
    db_conf_list.append(cConf.get(target, 'ipaddress'))
    db_conf_list.append(cConf.get(target, 'port'))
    db_conf_list.append(cConf.get(target, 'username'))
    db_conf_list.append(cConf.get(target, 'password'))
    db_conf_list.append(cConf.get(target, 'dbname'))

    return tuple(db_conf_list)

def fn_example_class_call():
    ''' =========================================
    create table tb_test(a int auto_increment not null primary key, b varchar(100) ) ;
    insert into tb_test(b) values('ii');
    ========================================= '''
    # print get version
    mDB = cDatabase("local")
    mDB.print_db_version()
    # ret = mDB.db_command("drop table if exists tb_test")
    # print(ret)
    # ret = mDB.db_command("create table tb_test(a int auto_increment not null primary key, b varchar(100))")
    # print(ret)
    # ret = mDB.db_command("insert into tb_test(b) value('°¡³ª')")
    # print(ret)

    # ## select one_row & print
    # row = mDB.db_select_row("select * from tb_test  ")
    # printData(row)

    # ## select multi_row & print
    # rows = mDB.db_select_rows("select * from tb_test  ")
    # printData(rows)

    # ## command (insert, update, delete, create, drop...)
    # ret = mDB.db_command("insert into tb_test(b) values('´Ù¶ó')")
    # print(ret)  ## return 1 -> success, retrun 0 -> fail

    # ## insert_many
    # data = [('a', 'b'), ('c', 'd'), ('e', 'f'),]
    # mDB.db_command('drop table if exists tb_test2')
    # mDB.db_command('create table tb_test2 (a varchar(10), b varchar(10))')
    # ret = mDB.db_insert_many("insert into tb_test2(a,b) values(%s, %s)", data)
    # print (ret)

    # db close !!!!!!
    mDB.db_close()


if __name__ == "__main__":
    fn_example_class_call()